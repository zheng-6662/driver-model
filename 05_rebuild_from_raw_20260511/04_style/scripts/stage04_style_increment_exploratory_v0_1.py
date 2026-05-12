# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd


ROOT = Path(r"F:/data_set_process/data_process/05_rebuild_from_raw_20260511")
STAGE03_SCRIPT_DIR = ROOT / "03_baselines" / "scripts"
if str(STAGE03_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE03_SCRIPT_DIR))

import evaluate_stage3_vehicle_baselines as eval_utils  # noqa: E402
import stage03_vehicle_instability_clean_task_vehicle_baselines_v0_1 as clean_base  # noqa: E402


STYLE_TABLE = (
    ROOT
    / "04_style"
    / "stage04_continuous_style_protocol_v0_1"
    / "tables"
    / "style_feature_candidate_wide_trainz_session_split.csv"
)
OUT_ROOT = ROOT / "04_style" / "stage04_style_increment_exploratory_v0_1"
TABLE_DIR = OUT_ROOT / "tables"
FIG_DIR = OUT_ROOT / "figures"
LOG_DIR = OUT_ROOT / "logs"
REPORT_DIR = ROOT / "09_reports"
NOTES_DIR = ROOT / "00_project_notes"
DAILY_LOG = NOTES_DIR / "daily_logs" / "2026-05-13.md"

SPLIT_STRATEGY = "session_level_split"
TRACK_ID = "B_response3s_strict_core"
WINDOW_ID = "pre3_label3_response_coverage"
TASK_ROLE = "response3s_strict_core_candidate"
RBF_MODEL = "rbf_kernel_ridge_context_no_subject"
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
SHUFFLE_SEEDS = [20260513, 20260514, 20260515, 20260516, 20260517]
WINDOW_PREFIXES = ["prefix_until_guard3", "last120_guard3", "last60_guard3", "last30_guard3"]

PLOT_MODELS = [
    (RBF_MODEL, "RBF vehicle", "#1f77b4"),
    ("rbf_plus_style_last60_guard3_residual_ridge", "RBF+style60", "#2ca02c"),
    ("rbf_plus_style_all_windows_residual_ridge", "RBF+style all", "#9467bd"),
    ("rbf_plus_driver_id_residual_ridge", "RBF+driver ID", "#ff7f0e"),
    ("rbf_plus_style_last60_with_driver_id_residual_ridge", "RBF+style60+ID", "#d62728"),
]


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIG_DIR, LOG_DIR, REPORT_DIR, DAILY_LOG.parent]:
        path.mkdir(parents=True, exist_ok=True)


def load_b_track() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    manifest = pd.read_csv(clean_base.TASK_MANIFEST_PATH)
    cfg = {
        "window_config_id": WINDOW_ID,
        "task_sample_role": TASK_ROLE,
        "description_cn": "3秒响应覆盖严格核心候选。",
    }
    return clean_base.load_track(TRACK_ID, cfg, manifest)


def split_indices(meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split = meta[SPLIT_STRATEGY].astype(str).to_numpy()
    return np.where(split == "train")[0], np.where(split == "val")[0], np.where(split == "test")[0]


def numeric_style_columns(style: pd.DataFrame, prefixes: list[str]) -> list[str]:
    cols: list[str] = []
    allowed_suffixes = ("_mean", "_std", "_p10", "_p50", "_p90", "_abs_mean", "_abs_p95", "_rms")
    blocked_fragments = (
        "window_status",
        "style_duration_s",
        "style_row_count",
        "style_sampling_rate_est_hz",
        "valid_ratio",
        "overlaps_",
        "uses_post_anchor_future",
    )
    for col in style.columns:
        if not any(col.startswith(prefix + "__") for prefix in prefixes):
            continue
        if any(fragment in col for fragment in blocked_fragments):
            continue
        if not col.endswith(allowed_suffixes):
            continue
        if pd.api.types.is_numeric_dtype(style[col]):
            cols.append(col)
    return cols


def style_matrix(style: pd.DataFrame, cols: list[str]) -> np.ndarray:
    x = style[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def one_hot_from_train(values: pd.Series, train_idx: np.ndarray, prefix: str) -> tuple[np.ndarray, list[str]]:
    vals = values.astype(str).fillna("NA").reset_index(drop=True)
    cats = sorted(vals.iloc[train_idx].unique().tolist())
    x = np.zeros((len(vals), len(cats)), dtype=np.float64)
    cat_to_j = {cat: j for j, cat in enumerate(cats)}
    for i, val in enumerate(vals):
        j = cat_to_j.get(val)
        if j is not None:
            x[i, j] = 1.0
    return x, [f"{prefix}={cat}" for cat in cats]


def fit_ridge_residual(
    x: np.ndarray,
    y: np.ndarray,
    y_mask: np.ndarray,
    base_pred: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    residual = np.where(y_mask, y - base_pred, 0.0).astype(np.float64)
    x_aug = np.concatenate([np.ones((x.shape[0], 1), dtype=np.float64), x.astype(np.float64)], axis=1)
    xt = x_aug[train_idx]
    yt = residual[train_idx]
    eye = np.eye(xt.shape[1], dtype=np.float64)
    eye[0, 0] = 0.0
    coef = np.linalg.solve(xt.T @ xt + alpha * eye, xt.T @ yt)
    pred = base_pred.astype(np.float64) + x_aug @ coef
    train_abs = np.nanpercentile(np.abs(y[train_idx][y_mask[train_idx]]), 99)
    clip = max(float(train_abs) * 2.5, 2.0)
    pred = np.clip(pred, -clip, clip).astype(np.float32)
    return pred, coef


def choose_alpha(
    model_name: str,
    x: np.ndarray,
    y: np.ndarray,
    y_mask: np.ndarray,
    base_pred: np.ndarray,
    label_time: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> tuple[float, np.ndarray, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    best_alpha = ALPHAS[0]
    best_pred: np.ndarray | None = None
    best_rmse = math.inf
    for alpha in ALPHAS:
        pred, _ = fit_ridge_residual(x, y, y_mask, base_pred, train_idx, val_idx, alpha)
        val_rmse = eval_utils.rmse(y[val_idx], pred[val_idx], y_mask[val_idx])
        rows.append({"model_name": model_name, "alpha": alpha, "val_rmse": val_rmse})
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_alpha = alpha
            best_pred = pred
    assert best_pred is not None
    return best_alpha, best_pred, rows


def shuffle_indices_by_group(group_values: pd.Series, split_values: pd.Series, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    perm = np.arange(len(group_values))
    keys = pd.DataFrame({"group": group_values.astype(str).fillna("NA"), "split": split_values.astype(str).fillna("NA")})
    for _, idx in keys.groupby(["split", "group"], dropna=False).groups.items():
        idx_arr = np.asarray(list(idx), dtype=int)
        if idx_arr.size > 1:
            perm[idx_arr] = rng.permutation(idx_arr)
    return perm


def shuffle_indices_global(split_values: pd.Series, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    perm = np.arange(len(split_values))
    for _, idx in split_values.astype(str).fillna("NA").groupby(split_values.astype(str).fillna("NA")).groups.items():
        idx_arr = np.asarray(list(idx), dtype=int)
        if idx_arr.size > 1:
            perm[idx_arr] = rng.permutation(idx_arr)
    return perm


def make_model_feature_sets(style: pd.DataFrame, train_idx: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    last60_cols = numeric_style_columns(style, ["last60_guard3"])
    all_style_cols = numeric_style_columns(style, WINDOW_PREFIXES)
    x_last60 = style_matrix(style, last60_cols)
    x_all = style_matrix(style, all_style_cols)
    x_driver, driver_names = one_hot_from_train(style["subject"], train_idx, "subject")
    x_road_module, road_module_names = one_hot_from_train(style["road_design_module_name"], train_idx, "road_module")
    feature_sets = {
        "rbf_plus_style_last60_guard3_residual_ridge": x_last60,
        "rbf_plus_style_all_windows_residual_ridge": x_all,
        "rbf_plus_driver_id_residual_ridge": x_driver,
        "rbf_plus_road_module_residual_ridge": x_road_module,
        "rbf_plus_style_last60_with_driver_id_residual_ridge": np.concatenate([x_last60, x_driver], axis=1),
    }
    feature_names = {
        "rbf_plus_style_last60_guard3_residual_ridge": last60_cols,
        "rbf_plus_style_all_windows_residual_ridge": all_style_cols,
        "rbf_plus_driver_id_residual_ridge": driver_names,
        "rbf_plus_road_module_residual_ridge": road_module_names,
        "rbf_plus_style_last60_with_driver_id_residual_ridge": last60_cols + driver_names,
    }
    return feature_sets, feature_names


def evaluate_predictions(
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
    split_values = meta[SPLIT_STRATEGY].astype(str).to_numpy()
    for split_name in ["train", "val", "test"]:
        mask = split_values == split_name
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
                window_id=WINDOW_ID,
                large_thr=large_thr,
                difficult_thr=difficult_thr,
            )
            if sample_rows:
                part = pd.DataFrame(sample_rows)
                part["track_id"] = TRACK_ID
                rows.append(part)
    per_sample = pd.concat(rows, ignore_index=True)
    metrics = eval_utils.aggregate_metrics(per_sample)
    metrics["track_id"] = TRACK_ID
    return metrics, per_sample


def summarize_by(metrics_source: pd.DataFrame, by_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    test = metrics_source[metrics_source["split"] == "test"].copy()
    for (model_name, key), grp in test.groupby(["model_name", by_col], dropna=False):
        rows.append(
            {
                "model_name": model_name,
                by_col: key,
                "n_samples": int(len(grp)),
                "mean_sample_rmse": float(np.sqrt(np.mean(np.square(grp["sample_rmse"])))),
                "wrong_side_rate": float(grp["wrong_side"].mean()),
                "large_response_recall": float(grp.loc[grp["is_large_response"] == 1, "large_response_recalled"].mean())
                if (grp["is_large_response"] == 1).any()
                else float("nan"),
                "severe_amp_under_rate": float(grp["severe_amp_under"].mean()),
                "difficult_top20_rmse": float(np.sqrt(np.mean(np.square(grp.loc[grp["is_difficult_peak_top20"] == 1, "sample_rmse"]))))
                if (grp["is_difficult_peak_top20"] == 1).any()
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def plot_metric_summary(metrics: pd.DataFrame) -> Path:
    models = [name for name, _, _ in PLOT_MODELS] + [
        "rbf_plus_road_module_residual_ridge",
        "rbf_plus_style_last60_within_subject_shuffle_mean",
        "rbf_plus_style_last60_global_shuffle_mean",
        "rbf_plus_style_last60_road_balanced_shuffle_mean",
    ]
    test = metrics[(metrics["split"] == "test") & (metrics["model_name"].isin(models))].copy()
    test = test.set_index("model_name").reindex([m for m in models if m in set(test["model_name"])])
    labels = [
        {
            RBF_MODEL: "RBF vehicle",
            "rbf_plus_style_last60_guard3_residual_ridge": "style60",
            "rbf_plus_style_all_windows_residual_ridge": "style all",
            "rbf_plus_driver_id_residual_ridge": "driver ID",
            "rbf_plus_road_module_residual_ridge": "road module",
            "rbf_plus_style_last60_with_driver_id_residual_ridge": "style60+ID",
            "rbf_plus_style_last60_within_subject_shuffle_mean": "within-subj shuffle",
            "rbf_plus_style_last60_global_shuffle_mean": "global shuffle",
            "rbf_plus_style_last60_road_balanced_shuffle_mean": "road-balanced shuffle",
        }.get(idx, idx)
        for idx in test.index
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), squeeze=False)
    panels = [
        ("rmse_steer", "Test RMSE", "#4c78a8"),
        ("wrong_side_rate", "Wrong-side rate", "#e45756"),
        ("large_response_recall", "Large-response recall", "#54a24b"),
        ("difficult_top20_rmse", "Difficult top20 RMSE", "#b279a2"),
    ]
    for ax, (col, title, color) in zip(axes.ravel(), panels):
        vals = test[col].to_numpy(dtype=float)
        ax.barh(labels, vals, color=color)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
        ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    out = FIG_DIR / "style_increment_metric_summary_test.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def plot_shuffle_controls(metrics: pd.DataFrame) -> Path:
    test = metrics[metrics["split"] == "test"].copy()
    keep = test[test["model_name"].str.contains("style_last60", regex=False)].copy()
    keep = keep[keep["model_name"].str.contains("seed") | keep["model_name"].isin(["rbf_plus_style_last60_guard3_residual_ridge"])]
    if keep.empty:
        out = FIG_DIR / "style_increment_shuffle_controls_test.png"
        return out
    order = keep.groupby("model_name")["rmse_steer"].mean().sort_values().index.tolist()
    labels = [v.replace("rbf_plus_style_last60_", "").replace("_residual_ridge", "") for v in order]
    fig, ax = plt.subplots(figsize=(13, max(5, 0.35 * len(order))))
    ax.barh(labels, keep.set_index("model_name").loc[order, "rmse_steer"], color="#72b7b2")
    ax.axvline(float(test.loc[test["model_name"] == RBF_MODEL, "rmse_steer"].iloc[0]), color="#1f77b4", linestyle="--", label="RBF")
    ax.set_title("Style last60 true vs shuffled controls: test RMSE")
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out = FIG_DIR / "style_increment_shuffle_controls_test.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def plot_sample_grid(
    sample_ids: list[str],
    y: np.ndarray,
    y_mask: np.ndarray,
    label_time: np.ndarray,
    meta: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    out_path: Path,
    title: str,
) -> None:
    cols = 4
    rows = int(np.ceil(max(len(sample_ids), 1) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(18, max(3.2 * rows, 3.6)), squeeze=False)
    id_to_idx = {sid: i for i, sid in enumerate(meta["sample_id"].astype(str))}
    for ax in axes.ravel():
        ax.axis("off")
    for k, sid in enumerate(sample_ids):
        ax = axes.ravel()[k]
        ax.axis("on")
        idx = id_to_idx[sid]
        gt = np.where(y_mask[idx] & np.isfinite(y[idx]), y[idx], np.nan)
        ax.plot(label_time, gt, color="black", linewidth=1.8, label="GT")
        for model_name, label, color in PLOT_MODELS:
            if model_name in predictions:
                ax.plot(label_time, predictions[model_name][idx], color=color, linewidth=1.05, alpha=0.95, label=label)
        ax.axhline(0, color="#dddddd", linewidth=0.8)
        ax.set_title(
            f"{meta.at[idx, 'subject']} {meta.at[idx, 'road_design_module_name']}\nanchor={meta.at[idx, 'anchor_time_rel_s']:.1f}s peak={np.nanmax(np.abs(gt)):.2f}",
            fontsize=8,
        )
        ax.tick_params(labelsize=8)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=min(6, len(labels)), fontsize=8)
    fig.suptitle(title, fontsize=12, y=0.955)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def merge_shuffle_means(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.Series] = []
    test_cols = [c for c in metrics.columns if c not in {"model_name"}]
    for prefix, name in [
        ("rbf_plus_style_last60_within_subject_shuffle_seed", "rbf_plus_style_last60_within_subject_shuffle_mean"),
        ("rbf_plus_style_last60_global_shuffle_seed", "rbf_plus_style_last60_global_shuffle_mean"),
        ("rbf_plus_style_last60_road_balanced_shuffle_seed", "rbf_plus_style_last60_road_balanced_shuffle_mean"),
    ]:
        part = metrics[metrics["model_name"].str.startswith(prefix)].copy()
        if part.empty:
            continue
        for split_name, grp in part.groupby("split"):
            numeric = grp.select_dtypes(include=[np.number]).mean(numeric_only=True)
            row = pd.Series({col: grp.iloc[0][col] for col in test_cols if col in grp.columns and col not in numeric.index})
            for col, val in numeric.items():
                row[col] = val
            row["split"] = split_name
            row["model_name"] = name
            rows.append(row)
    if not rows:
        return metrics
    return pd.concat([metrics, pd.DataFrame(rows)], ignore_index=True, sort=False)


def build_gate_table(metrics: pd.DataFrame, permutation_summary: pd.DataFrame) -> pd.DataFrame:
    test = metrics[metrics["split"] == "test"].set_index("model_name")
    rbf = test.loc[RBF_MODEL]
    style = test.loc["rbf_plus_style_last60_guard3_residual_ridge"]
    driver = test.loc["rbf_plus_driver_id_residual_ridge"]
    style_id = test.loc["rbf_plus_style_last60_with_driver_id_residual_ridge"]
    shuffle_mean = permutation_summary[
        (permutation_summary["control_name"] == "within_subject_shuffle")
        & (permutation_summary["split"] == "test")
    ]
    shuffle_rmse = float(shuffle_mean["rmse_mean"].iloc[0]) if not shuffle_mean.empty else float("nan")
    rows = [
        {
            "gate_item": "fixed_vehicle_reference",
            "status": "pass_reference",
            "evidence": f"{RBF_MODEL} test RMSE={rbf['rmse_steer']:.6f}; n={int(rbf['n_samples'])}",
            "decision_cn": "固定 RBF/KRR 车辆-only 作为本轮连续风格增量对照底线。",
        },
        {
            "gate_item": "style_last60_beats_rbf_rmse",
            "status": "pass_exploratory" if style["rmse_steer"] < rbf["rmse_steer"] else "fail",
            "evidence": f"style60 test RMSE={style['rmse_steer']:.6f}; RBF test RMSE={rbf['rmse_steer']:.6f}",
            "decision_cn": "只作为探索性迹象；不能单独证明连续风格有效。",
        },
        {
            "gate_item": "style_not_only_driver_id",
            "status": "needs_more_evidence" if style["rmse_steer"] < driver["rmse_steer"] else "fail_or_driver_proxy_risk",
            "evidence": f"style60 RMSE={style['rmse_steer']:.6f}; driver ID RMSE={driver['rmse_steer']:.6f}; style+ID RMSE={style_id['rmse_steer']:.6f}",
            "decision_cn": "需继续用 subject-level 或留一被试验证；本轮 session-level 不能排除身份代理风险。",
        },
        {
            "gate_item": "shuffle_control_drop",
            "status": "pass_exploratory" if np.isfinite(shuffle_rmse) and style["rmse_steer"] < shuffle_rmse else "fail",
            "evidence": f"style60 true RMSE={style['rmse_steer']:.6f}; within-subject shuffle mean RMSE={shuffle_rmse:.6f}",
            "decision_cn": "置乱后若收益下降，说明存在样本-风格对应信号；仍需更多 split 和 seed。",
        },
        {
            "gate_item": "physical_metric_improvement_required",
            "status": "needs_review",
            "evidence": (
                f"wrong_side RBF={rbf['wrong_side_rate']:.6f}, style60={style['wrong_side_rate']:.6f}; "
                f"large_recall RBF={rbf['large_response_recall']:.6f}, style60={style['large_response_recall']:.6f}; "
                f"difficult RMSE RBF={rbf['difficult_top20_rmse']:.6f}, style60={style['difficult_top20_rmse']:.6f}"
            ),
            "decision_cn": "若只改善 RMSE 而不改善物理错误，不能升级为主线。",
        },
        {
            "gate_item": "style_effectiveness_claim_allowed",
            "status": "blocked",
            "evidence": "当前只完成 session-level 探索性残差对照；subject-level/跨被试验证、更多置乱和固定图复核尚未完成。",
            "decision_cn": "不能宣称连续风格有效，只能说形成或未形成下一步验证候选。",
        },
        {
            "gate_item": "physio_eeg_allowed",
            "status": "blocked",
            "evidence": "车辆+连续风格公平参照还没有完成多 split 验证。",
            "decision_cn": "生理/EEG 仍不进入有效性验证。",
        },
    ]
    return pd.DataFrame(rows)


def write_reports(metrics: pd.DataFrame, gate: pd.DataFrame, feature_summary: pd.DataFrame, figures: dict[str, str]) -> None:
    test = metrics[metrics["split"] == "test"].set_index("model_name")
    def m(name: str, col: str) -> float:
        return float(test.loc[name, col])

    user = f"""# 阶段 4 用户查看版：连续驾驶风格探索性增量对照 v0.1

## 这个阶段为什么做

前面已经把 B 轨道 270 个严格失稳响应样本、3 秒标签窗口和 RBF/KNN 类车辆-only 对照固定下来。这一步不是直接证明风格有效，而是先问一个更小的问题：在固定 RBF 车辆-only 预测之后，事件前 60 秒或更长的连续驾驶风格，能不能解释一部分剩余误差。

## 这个阶段检查了什么

- 主任务只用 B 轨道 `response3s_strict_core_candidate`，共 270 个样本，test 40 个。
- 主参照固定为 `{RBF_MODEL}`，不再用 Transformer 作为当前主参照。
- 连续风格只来自事件前，并且与直接车辆输入窗口 `[-3, 0]` 和标签窗口 `[0, 3]` 不重叠。
- 对照包括：RBF、RBF+last60 风格、RBF+全部风格窗口、RBF+驾驶员 ID、RBF+道路模块、RBF+风格+驾驶员 ID，以及多种置乱风格。

## 目前发现了什么

- RBF test RMSE：{m(RBF_MODEL, 'rmse_steer'):.6f}
- RBF+last60 风格 test RMSE：{m('rbf_plus_style_last60_guard3_residual_ridge', 'rmse_steer'):.6f}
- RBF+全部风格窗口 test RMSE：{m('rbf_plus_style_all_windows_residual_ridge', 'rmse_steer'):.6f}
- RBF+驾驶员 ID test RMSE：{m('rbf_plus_driver_id_residual_ridge', 'rmse_steer'):.6f}
- RBF+last60 风格+驾驶员 ID test RMSE：{m('rbf_plus_style_last60_with_driver_id_residual_ridge', 'rmse_steer'):.6f}

物理指标上还要看错侧率、大幅响应召回、困难样本 RMSE 和坏样本图，不能只看 RMSE。

## 哪些结果可信

可信的是：本轮风格特征来源是事件前，标准化参数只来自 train split，评估使用固定 RBF 参照，并且加入了驾驶员 ID 与置乱对照。可信范围是“探索性增量对照”，不是最终有效性结论。

## 哪些结果还不能下结论

还不能说连续风格有效，也不能说它提供了跨被试泛化信息。因为当前只完成 session-level split，没有完成 subject-level 或留一被试验证；如果收益和驾驶员 ID 接近，也可能只是身份或道路分布代理。

## 下一阶段是否可以继续

可以继续做阶段 4 的更严格验证，但生理和 EEG 仍然不进入。下一步应该补 subject-level/跨 session 风格对照，并用固定预测图和坏样本图确认收益是不是来自真实物理错误改善。

## 推荐优先查看

1. `{figures['metric_summary']}`
2. `{figures['fixed_predictions']}`
3. `{figures['bad_samples']}`
4. `{(TABLE_DIR / 'style_increment_gate_table.csv').as_posix()}`
5. `{(TABLE_DIR / 'style_increment_metrics.csv').as_posix()}`
"""
    (REPORT_DIR / "stage04_style_increment_exploratory_user_summary_cn.md").write_text(user, encoding="utf-8")

    technical = f"""# 阶段 4：连续驾驶风格探索性增量对照 v0.1

## 输入与边界

- 样本：B 轨道 `{TASK_ROLE}`，窗口 `{WINDOW_ID}`。
- 主参照：`{RBF_MODEL}`。
- 风格表：`{STYLE_TABLE.as_posix()}`。
- 标准化：沿用阶段 4 协议的 session-level train-only z-score。
- 模型：只做 RBF 残差 Ridge，不训练 Transformer，不使用生理、EEG、EMG、RESP 或驾驶员生理状态。

## 特征数量

```text
{feature_summary.to_string(index=False)}
```

## test 指标摘录

```text
{test.reset_index()[['model_name','n_samples','rmse_steer','wrong_side_rate','large_response_recall','peak_amp_ratio_pred_over_gt_mean','severe_amp_under_rate','tail_abs_error_mean','reversal_count_exact_match_rate','difficult_top20_rmse']].to_string(index=False)}
```

## gate

```text
{gate.to_string(index=False)}
```

## 输出

- 指标表：`{(TABLE_DIR / 'style_increment_metrics.csv').as_posix()}`
- 逐样本表：`{(TABLE_DIR / 'style_increment_per_sample_metrics.csv').as_posix()}`
- alpha 选择：`{(TABLE_DIR / 'style_increment_validation_selection.csv').as_posix()}`
- 置乱汇总：`{(TABLE_DIR / 'style_increment_permutation_summary.csv').as_posix()}`
- gate 表：`{(TABLE_DIR / 'style_increment_gate_table.csv').as_posix()}`
- 固定图：`{figures['fixed_predictions']}`
- 坏样本图：`{figures['bad_samples']}`

## 解释限制

本轮如果出现收益，只能说明“事件前连续风格候选值得继续做更严格验证”。它还不满足阶段 4 对有效性的完成标准：至少两类切分成立、置乱收益稳定下降、不是驾驶员 ID 替代品、物理错误或困难样本稳定改善。
"""
    (REPORT_DIR / "stage04_style_increment_exploratory_v0_1_cn.md").write_text(technical, encoding="utf-8")


def update_transparency(run_summary: dict[str, Any], gate: pd.DataFrame) -> None:
    status_text = f"""# R2E-Steering 项目总进度看板

## 最新更新：2026-05-13 05:40

- 当前阶段：阶段 4 连续驾驶风格探索性增量对照 v0.1 已完成，仍处于探索阶段。
- 当前正在做什么：已固定 RBF/KNN 类车辆-only 主参照，正在判断事件前连续风格是否值得进入更严格 split 验证。
- 已完成什么：新增并运行 `stage04_style_increment_exploratory_v0_1.py`；在 B 轨道 270 个严格核心失稳响应样本上完成 RBF 残差 Ridge、驾驶员 ID 对照、道路模块对照和多种置乱控制；生成指标表、逐样本表、固定预测图、坏样本图、置乱汇总、gate 表和中文报告。
- 正在运行什么任务：没有后台任务；没有服务器任务；没有本地训练任务。
- 服务器是否在运行：未使用服务器，未读取服务器指令与密码文件，未记录任何凭据。
- 最近一次结果是什么：RBF test RMSE={run_summary['rbf_test_rmse']:.6f}；RBF+last60 风格 test RMSE={run_summary['style60_test_rmse']:.6f}；风格有效性结论仍为 blocked。
- 当前最大风险是什么：session-level 探索性收益可能来自驾驶员身份、道路/场景分布或小样本偶然性；如果物理指标和坏样本图不改善，不能升级为主线。
- 下一步准备做什么：补 subject-level/跨 session 风格验证，继续比较真实风格、驾驶员 ID、同被试置乱、跨被试置乱、道路均衡置乱；生理/EEG 仍阻塞。
- 用户可以优先查看哪些文件：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage04_style_increment_exploratory_user_summary_cn.md`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_increment_exploratory_v0_1/tables/style_increment_gate_table.csv`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_increment_exploratory_v0_1/figures/style_increment_metric_summary_test.png`。
"""
    existing = (NOTES_DIR / "PROJECT_STATUS_CN.md").read_text(encoding="utf-8", errors="replace")
    marker = "# R2E-Steering 项目总进度看板"
    if existing.startswith(marker):
        rest = existing[len(marker):]
        (NOTES_DIR / "PROJECT_STATUS_CN.md").write_text(status_text + "\n" + rest, encoding="utf-8")
    else:
        (NOTES_DIR / "PROJECT_STATUS_CN.md").write_text(status_text + "\n\n" + existing, encoding="utf-8")

    task_text = f"""# 当前任务队列

## 最新更新：2026-05-13 05:40

### 正在做任务
- 阶段 4 连续风格探索性增量对照 v0.1 已完成；当前准备进入 subject-level/跨 session 复核。

### 已完成任务
- 已新增并运行 `stage04_style_increment_exploratory_v0_1.py`。
- 已生成 RBF+连续风格、RBF+驾驶员 ID、RBF+道路模块、RBF+风格+ID 和置乱控制的指标、逐样本表、固定预测图、坏样本图和 gate 表。
- 当前 RBF test RMSE={run_summary['rbf_test_rmse']:.6f}，RBF+last60 风格 test RMSE={run_summary['style60_test_rmse']:.6f}。

### 待做任务
- 做 subject-level 或留一被试风格验证，检查是否跨被试成立。
- 做跨 session 与道路均衡置乱的更严格复核。
- 逐图复核固定预测图和坏样本图，判断收益是否改善错侧、幅值、尾段、反向修正、多段修正或困难样本。

### 阻塞任务
- 连续风格有效性结论仍阻塞，直到多 split、置乱、驾驶员 ID 对照和物理指标复核完成。
- 生理、脑电有效性验证仍阻塞，直到车辆+连续风格公平参照形成。

### 可并行任务
- 风格 subject-level 输入表和 RBF 参照对齐。
- 风格置乱 seed 扩展。
- 坏样本图人工复核摘要。

### 需要服务器的任务
- 暂无。

### 不需要服务器的任务
- 当前阶段 4 风格轻量对照和报告整理均可本地完成。
"""
    existing_task = (NOTES_DIR / "TASK_QUEUE_CN.md").read_text(encoding="utf-8", errors="replace")
    if existing_task.startswith("# 当前任务队列"):
        rest = existing_task[len("# 当前任务队列"):]
        (NOTES_DIR / "TASK_QUEUE_CN.md").write_text(task_text + "\n" + rest, encoding="utf-8")
    else:
        (NOTES_DIR / "TASK_QUEUE_CN.md").write_text(task_text + "\n\n" + existing_task, encoding="utf-8")

    artifact_entry = f"""## 最新新增：阶段 4 连续驾驶风格探索性增量对照 v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage04_style_increment_exploratory_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage04_style_increment_exploratory_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/scripts/stage04_style_increment_exploratory_v0_1.py`
- 指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_increment_exploratory_v0_1/tables/style_increment_metrics.csv`
- 逐样本表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_increment_exploratory_v0_1/tables/style_increment_per_sample_metrics.csv`
- 置乱汇总：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_increment_exploratory_v0_1/tables/style_increment_permutation_summary.csv`
- gate 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_increment_exploratory_v0_1/tables/style_increment_gate_table.csv`
- 固定预测图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_increment_exploratory_v0_1/figures/style_increment_fixed_predictions_test.png`
- 坏样本图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_increment_exploratory_v0_1/figures/style_increment_bad_samples_test.png`
- 服务器日志：无，本轮未使用远程服务器。
- GPTPro 提问和回复：无。
- 重要 Git commit：待提交。
- 适合用户/老师直接查看：用户查看版总结、gate 表、指标概览图、固定预测图、坏样本图。
"""
    artifact_path = NOTES_DIR / "ARTIFACT_INDEX_CN.md"
    existing_artifact = artifact_path.read_text(encoding="utf-8", errors="replace")
    artifact_path.write_text(existing_artifact + "\n" + artifact_entry, encoding="utf-8")

    log_entry = f"""
## 05:40 阶段 4：连续驾驶风格探索性增量对照 v0.1

- 为什么做：在固定 RBF/KNN 类车辆-only 主参照后，先用轻量、可解释、无生理输入的方式判断事件前连续风格是否值得继续验证。
- 做了什么：新增并运行 `stage04_style_increment_exploratory_v0_1.py`，训练 RBF 残差 Ridge 风格对照、驾驶员 ID 对照、道路模块对照和多种置乱控制；生成指标表、逐样本表、固定预测图、坏样本图、置乱汇总、gate 表和中文报告。
- 用了哪些输入：B 轨道严格核心样本 manifest、阶段 3 clean task 车辆-only RBF 参照、阶段 4 train-only z-score 连续风格表。
- 生成了哪些输出：`04_style/stage04_style_increment_exploratory_v0_1/` 下的 tables、figures、logs，以及 `09_reports/stage04_style_increment_exploratory_user_summary_cn.md`。
- 当前结果如何：RBF test RMSE={run_summary['rbf_test_rmse']:.6f}；RBF+last60 风格 test RMSE={run_summary['style60_test_rmse']:.6f}；有效性结论仍 blocked。
- 是否遇到问题：当前只是 session-level 探索性结果，可能受驾驶员 ID、道路分布和小样本影响。
- 是否需要用户决策：暂不需要；下一步继续做 subject-level/跨 session 风格验证，生理/EEG 仍不进入。
"""
    with DAILY_LOG.open("a", encoding="utf-8") as f:
        f.write(log_entry)


def main() -> None:
    ensure_dirs()
    y, y_mask, input_values, input_time, label_time, meta = load_b_track()
    train_idx, val_idx, test_idx = split_indices(meta)
    if min(len(train_idx), len(val_idx), len(test_idx)) <= 0:
        raise RuntimeError("B track has empty train/val/test split")

    style = pd.read_csv(STYLE_TABLE)
    style = meta[["sample_id"]].merge(style, on="sample_id", how="left", validate="one_to_one")
    if style.isna().all(axis=1).any():
        raise RuntimeError("style table alignment failed")
    feature_sets, feature_names = make_model_feature_sets(style, train_idx)

    base_predictions, _ = clean_base.build_strong_predictions(
        TRACK_ID,
        WINDOW_ID,
        y,
        y_mask,
        input_values,
        input_time,
        label_time,
        meta,
        train_idx,
        val_idx,
    )
    rbf_pred = base_predictions[RBF_MODEL]
    predictions: dict[str, np.ndarray] = {RBF_MODEL: rbf_pred}
    selection_rows: list[dict[str, Any]] = []
    feature_summary_rows: list[dict[str, Any]] = []

    for model_name, x in feature_sets.items():
        best_alpha, pred, rows = choose_alpha(model_name, x, y, y_mask, rbf_pred, label_time, train_idx, val_idx)
        predictions[model_name] = pred
        selection_rows.extend(rows)
        feature_summary_rows.append(
            {
                "model_name": model_name,
                "n_features": int(x.shape[1]),
                "selected_alpha": float(best_alpha),
                "feature_source": "train-only standardized continuous style" if "style" in model_name else "control one-hot",
            }
        )

    x_last60 = feature_sets["rbf_plus_style_last60_guard3_residual_ridge"]
    split_series = style[SPLIT_STRATEGY]
    shuffle_plan = [
        ("within_subject_shuffle", lambda seed: shuffle_indices_by_group(style["subject"], split_series, seed)),
        ("global_shuffle", lambda seed: shuffle_indices_global(split_series, seed)),
        ("road_balanced_shuffle", lambda seed: shuffle_indices_by_group(style["road_design_module_name"], split_series, seed)),
    ]
    permutation_detail_rows: list[dict[str, Any]] = []
    for control_name, perm_fn in shuffle_plan:
        for seed in SHUFFLE_SEEDS:
            perm = perm_fn(seed)
            x_perm = x_last60[perm]
            model_name = f"rbf_plus_style_last60_{control_name}_seed{seed}"
            best_alpha, pred, rows = choose_alpha(model_name, x_perm, y, y_mask, rbf_pred, label_time, train_idx, val_idx)
            predictions[model_name] = pred
            selection_rows.extend(rows)
            permutation_detail_rows.append(
                {
                    "control_name": control_name,
                    "seed": seed,
                    "selected_alpha": float(best_alpha),
                    "same_subject_rate": float(np.mean(style["subject"].astype(str).to_numpy() == style["subject"].astype(str).to_numpy()[perm])),
                    "same_session_rate": float(np.mean(style["session_stamp"].astype(str).to_numpy() == style["session_stamp"].astype(str).to_numpy()[perm])),
                    "same_road_module_rate": float(np.mean(style["road_design_module_name"].astype(str).to_numpy() == style["road_design_module_name"].astype(str).to_numpy()[perm])),
                }
            )

    metrics, per_sample = evaluate_predictions(y, y_mask, label_time, meta, predictions, train_idx)
    metrics_with_means = merge_shuffle_means(metrics)

    perm_rows: list[dict[str, Any]] = []
    for control_name in ["within_subject_shuffle", "global_shuffle", "road_balanced_shuffle"]:
        prefix = f"rbf_plus_style_last60_{control_name}_seed"
        part = metrics[(metrics["split"].isin(["val", "test"])) & (metrics["model_name"].str.startswith(prefix))].copy()
        for split_name, grp in part.groupby("split"):
            perm_rows.append(
                {
                    "control_name": control_name,
                    "split": split_name,
                    "n_seeds": int(len(grp)),
                    "rmse_mean": float(grp["rmse_steer"].mean()),
                    "rmse_std": float(grp["rmse_steer"].std(ddof=0)),
                    "wrong_side_rate_mean": float(grp["wrong_side_rate"].mean()),
                    "large_response_recall_mean": float(grp["large_response_recall"].mean()),
                    "difficult_top20_rmse_mean": float(grp["difficult_top20_rmse"].mean()),
                }
            )
    permutation_summary = pd.DataFrame(perm_rows)
    feature_summary = pd.DataFrame(feature_summary_rows)
    selection = pd.DataFrame(selection_rows)
    permutation_detail = pd.DataFrame(permutation_detail_rows)
    gate = build_gate_table(metrics_with_means, permutation_summary)

    metrics_with_means.to_csv(TABLE_DIR / "style_increment_metrics.csv", index=False, encoding="utf-8-sig")
    per_sample.to_csv(TABLE_DIR / "style_increment_per_sample_metrics.csv", index=False, encoding="utf-8-sig")
    selection.to_csv(TABLE_DIR / "style_increment_validation_selection.csv", index=False, encoding="utf-8-sig")
    feature_summary.to_csv(TABLE_DIR / "style_increment_feature_summary.csv", index=False, encoding="utf-8-sig")
    permutation_summary.to_csv(TABLE_DIR / "style_increment_permutation_summary.csv", index=False, encoding="utf-8-sig")
    permutation_detail.to_csv(TABLE_DIR / "style_increment_permutation_detail.csv", index=False, encoding="utf-8-sig")
    gate.to_csv(TABLE_DIR / "style_increment_gate_table.csv", index=False, encoding="utf-8-sig")
    summarize_by(per_sample, "subject").to_csv(TABLE_DIR / "style_increment_subject_summary_test.csv", index=False, encoding="utf-8-sig")
    summarize_by(per_sample, "session_stamp").to_csv(TABLE_DIR / "style_increment_session_summary_test.csv", index=False, encoding="utf-8-sig")
    per_sample.merge(
        meta[["sample_id", "road_design_module_name", "road_design_instance_name"]],
        on="sample_id",
        how="left",
    ).pipe(summarize_by, "road_design_module_name").to_csv(
        TABLE_DIR / "style_increment_road_module_summary_test.csv", index=False, encoding="utf-8-sig"
    )

    metric_fig = plot_metric_summary(metrics_with_means)
    shuffle_fig = plot_shuffle_controls(metrics)
    test_meta = meta.loc[meta[SPLIT_STRATEGY].astype(str).to_numpy() == "test"].copy()
    fixed_ids = test_meta.sort_values(["subject", "anchor_time_rel_s"]).head(min(8, len(test_meta)))["sample_id"].astype(str).tolist()
    rbf_bad = per_sample[
        (per_sample["split"] == "test") & (per_sample["model_name"] == RBF_MODEL)
    ].sort_values("sample_rmse", ascending=False)
    bad_ids = rbf_bad.head(min(8, len(rbf_bad)))["sample_id"].astype(str).tolist()
    fixed_fig = FIG_DIR / "style_increment_fixed_predictions_test.png"
    bad_fig = FIG_DIR / "style_increment_bad_samples_test.png"
    plot_sample_grid(fixed_ids, y, y_mask, label_time, meta, predictions, fixed_fig, "Stage04 style increment: fixed test samples")
    plot_sample_grid(bad_ids, y, y_mask, label_time, meta, predictions, bad_fig, "Stage04 style increment: RBF bad test samples")

    figures = {
        "metric_summary": metric_fig.as_posix(),
        "shuffle_controls": shuffle_fig.as_posix(),
        "fixed_predictions": fixed_fig.as_posix(),
        "bad_samples": bad_fig.as_posix(),
    }
    write_reports(metrics_with_means, gate, feature_summary, figures)

    test = metrics_with_means[metrics_with_means["split"] == "test"].set_index("model_name")
    run_summary = {
        "run_time_local": "2026-05-13 05:40",
        "track_id": TRACK_ID,
        "window_config_id": WINDOW_ID,
        "task_sample_role": TASK_ROLE,
        "n_samples": int(len(meta)),
        "train_n": int(len(train_idx)),
        "val_n": int(len(val_idx)),
        "test_n": int(len(test_idx)),
        "rbf_test_rmse": float(test.loc[RBF_MODEL, "rmse_steer"]),
        "style60_test_rmse": float(test.loc["rbf_plus_style_last60_guard3_residual_ridge", "rmse_steer"]),
        "style_all_test_rmse": float(test.loc["rbf_plus_style_all_windows_residual_ridge", "rmse_steer"]),
        "driver_id_test_rmse": float(test.loc["rbf_plus_driver_id_residual_ridge", "rmse_steer"]),
        "style60_plus_id_test_rmse": float(test.loc["rbf_plus_style_last60_with_driver_id_residual_ridge", "rmse_steer"]),
        "feature_summary_path": (TABLE_DIR / "style_increment_feature_summary.csv").as_posix(),
        "metrics_path": (TABLE_DIR / "style_increment_metrics.csv").as_posix(),
        "per_sample_path": (TABLE_DIR / "style_increment_per_sample_metrics.csv").as_posix(),
        "gate_path": (TABLE_DIR / "style_increment_gate_table.csv").as_posix(),
        "figures": figures,
        "server_used": False,
        "server_access_file_read": False,
        "uses_physio": False,
        "uses_eeg": False,
        "uses_emg": False,
        "uses_resp": False,
        "uses_continuous_style": True,
        "style_effectiveness_claim_allowed": False,
        "raw_files_modified": False,
    }
    (LOG_DIR / "style_increment_exploratory_summary.json").write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    update_transparency(run_summary, gate)
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
