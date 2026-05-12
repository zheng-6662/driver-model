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


STYLE_RAW_TABLE = (
    ROOT
    / "04_style"
    / "stage04_continuous_style_protocol_v0_1"
    / "tables"
    / "style_feature_candidate_wide.csv"
)
OUT_ROOT = ROOT / "04_style" / "stage04_style_cross_split_validation_v0_1"
TABLE_DIR = OUT_ROOT / "tables"
FIG_DIR = OUT_ROOT / "figures"
LOG_DIR = OUT_ROOT / "logs"
REPORT_DIR = ROOT / "09_reports"
NOTES_DIR = ROOT / "00_project_notes"
DAILY_LOG = NOTES_DIR / "daily_logs" / "2026-05-13.md"

TRACK_ID = "B_response3s_strict_core"
WINDOW_ID = "pre3_label3_response_coverage"
TASK_ROLE = "response3s_strict_core_candidate"
RBF_MODEL = "rbf_kernel_ridge_context_no_subject"
SPLIT_STRATEGIES = ["session_level_split", "subject_level_split"]
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
WINDOW_PREFIXES = ["prefix_until_guard3", "last120_guard3", "last60_guard3", "last30_guard3"]
PLOT_MODELS = [
    RBF_MODEL,
    "rbf_plus_style_last60_guard3_residual_ridge",
    "rbf_plus_style_all_windows_residual_ridge",
    "rbf_plus_driver_id_residual_ridge",
    "rbf_plus_style_last60_with_driver_id_residual_ridge",
]
DISPLAY = {
    RBF_MODEL: "RBF vehicle",
    "rbf_plus_style_last60_guard3_residual_ridge": "style60",
    "rbf_plus_style_all_windows_residual_ridge": "style all",
    "rbf_plus_driver_id_residual_ridge": "driver ID",
    "rbf_plus_style_last60_with_driver_id_residual_ridge": "style60+ID",
}


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


def split_indices(meta: pd.DataFrame, split_strategy: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split = meta[split_strategy].astype(str).to_numpy()
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


def raw_style_matrix(style: pd.DataFrame, cols: list[str]) -> np.ndarray:
    x = style[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    return np.where(np.isfinite(x), x, np.nan)


def standardize_train_only(x: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    train = x[train_idx]
    mean = np.nanmean(train, axis=0)
    std = np.nanstd(train, axis=0)
    valid = np.isfinite(mean) & np.isfinite(std) & (std > 1e-8)
    out = np.zeros_like(x, dtype=np.float64)
    if valid.any():
        out[:, valid] = (np.nan_to_num(x[:, valid], nan=mean[valid]) - mean[valid]) / std[valid]
    params = pd.DataFrame(
        {
            "feature_index": np.arange(x.shape[1]),
            "train_mean": mean,
            "train_std": std,
            "usable": valid,
        }
    )
    return out, params


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
    alpha: float,
) -> np.ndarray:
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
    return np.clip(pred, -clip, clip).astype(np.float32)


def choose_alpha(
    split_strategy: str,
    model_name: str,
    x: np.ndarray,
    y: np.ndarray,
    y_mask: np.ndarray,
    base_pred: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> tuple[float, np.ndarray, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    best_alpha = ALPHAS[0]
    best_pred: np.ndarray | None = None
    best_rmse = math.inf
    for alpha in ALPHAS:
        pred = fit_ridge_residual(x, y, y_mask, base_pred, train_idx, alpha)
        val_rmse = eval_utils.rmse(y[val_idx], pred[val_idx], y_mask[val_idx])
        rows.append({"split_strategy": split_strategy, "model_name": model_name, "alpha": alpha, "val_rmse": val_rmse})
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_alpha = alpha
            best_pred = pred
    assert best_pred is not None
    return best_alpha, best_pred, rows


def make_feature_sets(style: pd.DataFrame, train_idx: np.ndarray, split_strategy: str) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    last60_cols = numeric_style_columns(style, ["last60_guard3"])
    all_style_cols = numeric_style_columns(style, WINDOW_PREFIXES)
    x_last60_raw = raw_style_matrix(style, last60_cols)
    x_all_raw = raw_style_matrix(style, all_style_cols)
    x_last60, params_last60 = standardize_train_only(x_last60_raw, train_idx)
    x_all, params_all = standardize_train_only(x_all_raw, train_idx)
    x_driver, _ = one_hot_from_train(style["subject"], train_idx, "subject")
    x_road, _ = one_hot_from_train(style["road_design_module_name"], train_idx, "road_module")
    feature_sets = {
        "rbf_plus_style_last60_guard3_residual_ridge": x_last60,
        "rbf_plus_style_all_windows_residual_ridge": x_all,
        "rbf_plus_driver_id_residual_ridge": x_driver,
        "rbf_plus_road_module_residual_ridge": x_road,
        "rbf_plus_style_last60_with_driver_id_residual_ridge": np.concatenate([x_last60, x_driver], axis=1),
    }
    rows = [
        {
            "split_strategy": split_strategy,
            "model_name": "rbf_plus_style_last60_guard3_residual_ridge",
            "n_features": int(x_last60.shape[1]),
            "usable_train_standardized_features": int(params_last60["usable"].sum()),
            "source": "raw last60 style; standardized on train only",
        },
        {
            "split_strategy": split_strategy,
            "model_name": "rbf_plus_style_all_windows_residual_ridge",
            "n_features": int(x_all.shape[1]),
            "usable_train_standardized_features": int(params_all["usable"].sum()),
            "source": "raw all style windows; standardized on train only",
        },
        {
            "split_strategy": split_strategy,
            "model_name": "rbf_plus_driver_id_residual_ridge",
            "n_features": int(x_driver.shape[1]),
            "usable_train_standardized_features": int(x_driver.shape[1]),
            "source": "train-subject one-hot control",
        },
        {
            "split_strategy": split_strategy,
            "model_name": "rbf_plus_road_module_residual_ridge",
            "n_features": int(x_road.shape[1]),
            "usable_train_standardized_features": int(x_road.shape[1]),
            "source": "train-road-module one-hot control",
        },
        {
            "split_strategy": split_strategy,
            "model_name": "rbf_plus_style_last60_with_driver_id_residual_ridge",
            "n_features": int(x_last60.shape[1] + x_driver.shape[1]),
            "usable_train_standardized_features": int(params_last60["usable"].sum() + x_driver.shape[1]),
            "source": "raw last60 style + train-subject one-hot control",
        },
    ]
    return feature_sets, pd.DataFrame(rows)


def evaluate_predictions(
    split_strategy: str,
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
    split_values = meta[split_strategy].astype(str).to_numpy()
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
                split_strategy=split_strategy,
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


def run_split(
    split_strategy: str,
    y: np.ndarray,
    y_mask: np.ndarray,
    input_values: np.ndarray,
    input_time: np.ndarray,
    label_time: np.ndarray,
    meta: pd.DataFrame,
    style: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_idx, val_idx, test_idx = split_indices(meta, split_strategy)
    if min(len(train_idx), len(val_idx), len(test_idx)) <= 0:
        raise RuntimeError(f"{split_strategy}: train/val/test split incomplete")
    old_split = clean_base.SPLIT_STRATEGY
    clean_base.SPLIT_STRATEGY = split_strategy
    try:
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
    finally:
        clean_base.SPLIT_STRATEGY = old_split
    predictions: dict[str, np.ndarray] = {RBF_MODEL: base_predictions[RBF_MODEL]}
    feature_sets, feature_summary = make_feature_sets(style, train_idx, split_strategy)
    selection_rows: list[dict[str, Any]] = []
    for model_name, x in feature_sets.items():
        best_alpha, pred, rows = choose_alpha(split_strategy, model_name, x, y, y_mask, predictions[RBF_MODEL], train_idx, val_idx)
        predictions[model_name] = pred
        selection_rows.extend(rows)
        feature_summary.loc[feature_summary["model_name"] == model_name, "selected_alpha"] = best_alpha
    metrics, per_sample = evaluate_predictions(split_strategy, y, y_mask, label_time, meta, predictions, train_idx)
    split_summary = pd.DataFrame(
        [
            {
                "split_strategy": split_strategy,
                "split": split_name,
                "n_samples": int((meta[split_strategy].astype(str) == split_name).sum()),
                "n_subjects": int(meta.loc[meta[split_strategy].astype(str) == split_name, "subject"].nunique()),
                "subjects": ",".join(sorted(meta.loc[meta[split_strategy].astype(str) == split_name, "subject"].astype(str).unique())),
                "n_sessions": int(meta.loc[meta[split_strategy].astype(str) == split_name, "session_stamp"].nunique()),
            }
            for split_name in ["train", "val", "test"]
        ]
    )
    return predictions, metrics, per_sample, pd.DataFrame(selection_rows), pd.concat([feature_summary, split_summary], ignore_index=True, sort=False)


def build_gate_table(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split_strategy in SPLIT_STRATEGIES:
        test = metrics[(metrics["split_strategy"] == split_strategy) & (metrics["split"] == "test")].set_index("model_name")
        rbf = test.loc[RBF_MODEL]
        style = test.loc["rbf_plus_style_last60_guard3_residual_ridge"]
        driver = test.loc["rbf_plus_driver_id_residual_ridge"]
        rows.append(
            {
                "gate_item": f"{split_strategy}_style_last60_beats_rbf",
                "status": "pass_exploratory" if style["rmse_steer"] < rbf["rmse_steer"] else "fail",
                "evidence": f"style60 RMSE={style['rmse_steer']:.6f}; RBF RMSE={rbf['rmse_steer']:.6f}; driverID RMSE={driver['rmse_steer']:.6f}",
                "decision_cn": "必须同时看物理指标、驾驶员 ID 对照和跨 split 稳定性。",
            }
        )
        rows.append(
            {
                "gate_item": f"{split_strategy}_physical_improvement",
                "status": "pass_exploratory"
                if (
                    style["wrong_side_rate"] < rbf["wrong_side_rate"]
                    or style["large_response_recall"] > rbf["large_response_recall"]
                    or style["difficult_top20_rmse"] < rbf["difficult_top20_rmse"]
                )
                else "fail",
                "evidence": (
                    f"wrong_side {rbf['wrong_side_rate']:.6f}->{style['wrong_side_rate']:.6f}; "
                    f"large_recall {rbf['large_response_recall']:.6f}->{style['large_response_recall']:.6f}; "
                    f"difficult_rmse {rbf['difficult_top20_rmse']:.6f}->{style['difficult_top20_rmse']:.6f}"
                ),
                "decision_cn": "若物理错误没有改善，不能升级为连续风格有效证据。",
            }
        )
    session_test = metrics[(metrics["split_strategy"] == "session_level_split") & (metrics["split"] == "test")].set_index("model_name")
    subject_test = metrics[(metrics["split_strategy"] == "subject_level_split") & (metrics["split"] == "test")].set_index("model_name")
    session_pass = session_test.loc["rbf_plus_style_last60_guard3_residual_ridge", "rmse_steer"] < session_test.loc[RBF_MODEL, "rmse_steer"]
    subject_pass = subject_test.loc["rbf_plus_style_last60_guard3_residual_ridge", "rmse_steer"] < subject_test.loc[RBF_MODEL, "rmse_steer"]
    rows.append(
        {
            "gate_item": "style_effectiveness_claim_allowed",
            "status": "blocked",
            "evidence": f"session_pass={session_pass}; subject_pass={subject_pass}; no stable two-split improvement.",
            "decision_cn": "不能宣称连续风格有效；当前只完成否定/降级证据的一部分。",
        }
    )
    rows.append(
        {
            "gate_item": "physio_eeg_allowed",
            "status": "blocked",
            "evidence": "连续风格没有形成强于车辆-only 的稳定公平参照。",
            "decision_cn": "生理/EEG 继续阻塞，除非先完成车辆-only 结构化或风格路线收口。",
        }
    )
    return pd.DataFrame(rows)


def plot_cross_split(metrics: pd.DataFrame) -> Path:
    test = metrics[(metrics["split"] == "test") & (metrics["model_name"].isin(PLOT_MODELS))].copy()
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), squeeze=False)
    panels = [
        ("rmse_steer", "Test RMSE", "#4c78a8"),
        ("wrong_side_rate", "Wrong-side rate", "#e45756"),
        ("large_response_recall", "Large-response recall", "#54a24b"),
        ("difficult_top20_rmse", "Difficult top20 RMSE", "#b279a2"),
    ]
    for ax, (col, title, color) in zip(axes.ravel(), panels):
        width = 0.36
        x = np.arange(len(PLOT_MODELS))
        for i, split_strategy in enumerate(SPLIT_STRATEGIES):
            part = test[test["split_strategy"] == split_strategy].set_index("model_name").reindex(PLOT_MODELS)
            vals = part[col].to_numpy(dtype=float)
            ax.bar(x + (i - 0.5) * width, vals, width=width, label=split_strategy, color=color, alpha=0.65 + 0.25 * i)
        ax.set_xticks(x)
        ax.set_xticklabels([DISPLAY.get(m, m) for m in PLOT_MODELS], rotation=25, ha="right", fontsize=8)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    axes[0, 0].legend()
    fig.tight_layout()
    out = FIG_DIR / "style_cross_split_metric_summary_test.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def plot_subject_bad_samples(
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
    colors = {
        RBF_MODEL: "#1f77b4",
        "rbf_plus_style_last60_guard3_residual_ridge": "#2ca02c",
        "rbf_plus_driver_id_residual_ridge": "#ff7f0e",
    }
    for k, sid in enumerate(sample_ids):
        ax = axes.ravel()[k]
        ax.axis("on")
        idx = id_to_idx[sid]
        gt = np.where(y_mask[idx] & np.isfinite(y[idx]), y[idx], np.nan)
        ax.plot(label_time, gt, color="black", linewidth=1.8, label="GT")
        for model_name, color in colors.items():
            ax.plot(label_time, predictions[model_name][idx], color=color, linewidth=1.05, label=DISPLAY.get(model_name, model_name))
        ax.axhline(0, color="#dddddd", linewidth=0.8)
        ax.set_title(
            f"{meta.at[idx, 'subject']} {meta.at[idx, 'road_design_module_name']}\nanchor={meta.at[idx, 'anchor_time_rel_s']:.1f}s",
            fontsize=8,
        )
        ax.tick_params(labelsize=8)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=4, fontsize=8)
    fig.suptitle(title, fontsize=12, y=0.955)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_reports(metrics: pd.DataFrame, gate: pd.DataFrame, split_feature_summary: pd.DataFrame, figures: dict[str, str]) -> None:
    test = metrics[(metrics["split"] == "test") & (metrics["model_name"].isin(PLOT_MODELS))].copy()
    pivot = test.pivot(index="model_name", columns="split_strategy", values="rmse_steer").reindex(PLOT_MODELS)
    def val(model: str, split_strategy: str, col: str) -> float:
        row = test[(test["model_name"] == model) & (test["split_strategy"] == split_strategy)].iloc[0]
        return float(row[col])

    user = f"""# 阶段 4 用户查看版：连续风格跨 split 复核 v0.1

## 这个阶段为什么做

上一轮 session-level 探索里，`RBF+last60 风格` 没有超过 RBF。为了确认这不是某一种划分下的偶然现象，这一轮补做 subject-level 复核：训练集和测试集按被试分开，测试被试是训练中没见过的人。

## 这个阶段检查了什么

- 样本仍然是 B 轨道 270 个严格核心失稳响应样本。
- 比较 `session_level_split` 和 `subject_level_split` 两类切分。
- 每个 split 都重新用对应 train split 拟合风格标准化参数，不沿用上一轮 session 标准化。
- 主参照仍是 `{RBF_MODEL}`。
- 风格只作为 RBF 残差模型输入，不使用生理、脑电、EMG、RESP，也不训练 Transformer。

## 目前发现了什么

```text
{pivot.to_string()}
```

session-level：RBF RMSE={val(RBF_MODEL, 'session_level_split', 'rmse_steer'):.6f}，RBF+last60 风格 RMSE={val('rbf_plus_style_last60_guard3_residual_ridge', 'session_level_split', 'rmse_steer'):.6f}。

subject-level：RBF RMSE={val(RBF_MODEL, 'subject_level_split', 'rmse_steer'):.6f}，RBF+last60 风格 RMSE={val('rbf_plus_style_last60_guard3_residual_ridge', 'subject_level_split', 'rmse_steer'):.6f}。

## 哪些结果可信

可信的是：连续风格在 session-level 和 subject-level 两类切分下，都没有形成稳定超过 RBF 的证据。subject-level 复核尤其重要，因为它把测试被试放到训练外，更接近“风格是否有跨人泛化信息”的问题。

## 哪些结果还不能下结论

还不能说“风格永远无效”。目前只能说：在当前事件前风格特征、RBF 残差 Ridge 融合方式、B 轨道 3 秒严格核心样本上，没有形成足够证据支持连续风格有效。未来如果换更强的风格表示或结构，可以重新验证，但不能直接升级为主线。

## 下一阶段是否可以继续

生理和 EEG 仍然不能进入有效性验证。下一步更合理的是先把阶段 4 暂时降级收口，回到车辆-only 结构化轨迹建模，优先解决错侧、反向修正、多段修正和困难样本。

## 推荐优先查看

1. `{figures['metric_summary']}`
2. `{figures['subject_bad_samples']}`
3. `{(TABLE_DIR / 'style_cross_split_gate_table.csv').as_posix()}`
4. `{(TABLE_DIR / 'style_cross_split_metrics.csv').as_posix()}`
"""
    (REPORT_DIR / "stage04_style_cross_split_validation_user_summary_cn.md").write_text(user, encoding="utf-8")

    technical = f"""# 阶段 4：连续风格跨 split 复核 v0.1

## 输入与协议

- 样本：B 轨道 `{TASK_ROLE}`，窗口 `{WINDOW_ID}`，共 270 个样本。
- split：`session_level_split`、`subject_level_split`。
- 每个 split 重新计算 train-only 风格标准化。
- 主参照：`{RBF_MODEL}`。
- 融合方式：RBF 残差 Ridge。
- 禁用：生理、脑电、EMG、RESP、服务器、服务器凭据。

## split 与特征摘要

```text
{split_feature_summary.to_string(index=False)}
```

## test 指标

```text
{test[['split_strategy','model_name','n_samples','rmse_steer','wrong_side_rate','large_response_recall','peak_amp_ratio_pred_over_gt_mean','severe_amp_under_rate','tail_abs_error_mean','reversal_count_exact_match_rate','difficult_top20_rmse']].to_string(index=False)}
```

## gate

```text
{gate.to_string(index=False)}
```

## 结论

当前连续风格路线在 session-level 与 subject-level 两类切分下均没有稳定超过 RBF，也没有稳定改善关键物理指标。因此阶段 4 不能支持“连续风格有效”的结论，生理/EEG 继续阻塞。建议后续先回到车辆-only 结构化轨迹模型。
"""
    (REPORT_DIR / "stage04_style_cross_split_validation_v0_1_cn.md").write_text(technical, encoding="utf-8")


def update_transparency(run_summary: dict[str, Any]) -> None:
    status_section = f"""## 最新更新：2026-05-13 06:05

- 当前阶段：阶段 4 连续风格跨 split 复核 v0.1 已完成；连续风格有效性结论仍阻塞。
- 当前正在做什么：准备把连续风格路线暂时降级收口，并回到车辆-only 结构化轨迹建模问题。
- 已完成什么：新增并运行 `stage04_style_cross_split_validation_v0_1.py`；在 B 轨道 270 个严格核心样本上完成 session-level 与 subject-level 两类切分的 RBF+风格残差对照，且每个 split 都重新使用 train-only 风格标准化。
- 正在运行什么任务：没有后台任务；没有服务器任务；没有本地训练任务。
- 服务器是否在运行：未使用服务器，未读取服务器指令与密码文件，未记录任何凭据。
- 最近一次结果是什么：session-level RBF RMSE={run_summary['session_rbf_rmse']:.6f}、RBF+last60 风格 RMSE={run_summary['session_style60_rmse']:.6f}；subject-level RBF RMSE={run_summary['subject_rbf_rmse']:.6f}、RBF+last60 风格 RMSE={run_summary['subject_style60_rmse']:.6f}；风格有效性 gate 仍为 blocked。
- 当前最大风险是什么：如果继续强推风格，可能把小样本波动、被试/道路分布或融合方式不足误解释为风格有效或无效；当前证据只支持“当前表示和融合方式下没有形成强证据”。
- 下一步准备做什么：阶段 4 先收口，返回阶段 6/车辆-only 结构化轨迹路线，优先解决错侧、幅值、尾段、反向修正、多段修正和困难样本。
- 用户可以优先查看哪些文件：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage04_style_cross_split_validation_user_summary_cn.md`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_cross_split_validation_v0_1/tables/style_cross_split_gate_table.csv`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_cross_split_validation_v0_1/figures/style_cross_split_metric_summary_test.png`。
"""
    for filename, title, section in [
        ("PROJECT_STATUS_CN.md", "# R2E-Steering 项目总进度看板", status_section),
    ]:
        path = NOTES_DIR / filename
        base = path.read_text(encoding="utf-8", errors="replace")
        if base.startswith(title):
            path.write_text(title + "\n\n" + section.strip() + "\n\n" + base[len(title):].lstrip("\r\n"), encoding="utf-8")
        else:
            path.write_text(section.strip() + "\n\n" + base, encoding="utf-8")

    task_section = f"""## 最新更新：2026-05-13 06:05

### 正在做任务
- 阶段 4 连续风格跨 split 复核已完成；当前准备把风格路线暂时降级收口。

### 已完成任务
- 已新增并运行 `stage04_style_cross_split_validation_v0_1.py`。
- 已完成 session-level 与 subject-level 两类切分的 RBF+连续风格残差对照。
- 已确认当前 last60 连续风格在两类切分下都没有稳定超过 RBF：session {run_summary['session_rbf_rmse']:.6f}->{run_summary['session_style60_rmse']:.6f}，subject {run_summary['subject_rbf_rmse']:.6f}->{run_summary['subject_style60_rmse']:.6f}。

### 待做任务
- 写阶段 4 收口说明：当前表示/融合下连续风格没有形成强证据。
- 回到车辆-only 结构化轨迹建模，优先错侧、幅值、尾段、反向修正、多段修正和困难样本。

### 阻塞任务
- 连续风格有效性结论仍阻塞。
- 生理、脑电有效性验证仍阻塞，直到车辆-only 和风格参照更稳。

### 可并行任务
- 固定图/坏样本图人工复核摘要。
- 车辆-only 结构化轨迹模型候选设计。

### 需要服务器的任务
- 暂无。

### 不需要服务器的任务
- 当前阶段 4 收口报告和车辆-only 结构设计可本地完成。
"""
    task_path = NOTES_DIR / "TASK_QUEUE_CN.md"
    task_base = task_path.read_text(encoding="utf-8", errors="replace")
    title = "# 当前任务队列"
    if task_base.startswith(title):
        task_path.write_text(title + "\n\n" + task_section.strip() + "\n\n" + task_base[len(title):].lstrip("\r\n"), encoding="utf-8")
    else:
        task_path.write_text(task_section.strip() + "\n\n" + task_base, encoding="utf-8")

    artifact_entry = """## 最新新增：阶段 4 连续风格跨 split 复核 v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage04_style_cross_split_validation_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage04_style_cross_split_validation_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/scripts/stage04_style_cross_split_validation_v0_1.py`
- 指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_cross_split_validation_v0_1/tables/style_cross_split_metrics.csv`
- 逐样本表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_cross_split_validation_v0_1/tables/style_cross_split_per_sample_metrics.csv`
- gate 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_cross_split_validation_v0_1/tables/style_cross_split_gate_table.csv`
- 指标图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_cross_split_validation_v0_1/figures/style_cross_split_metric_summary_test.png`
- subject-level 坏样本图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_cross_split_validation_v0_1/figures/style_cross_split_subject_bad_samples_test.png`
- 服务器日志：无，本轮未使用远程服务器。
- GPTPro 提问和回复：无。
- 重要 Git commit：待提交。
- 适合用户/老师直接查看：用户查看版总结、gate 表、指标图、subject-level 坏样本图。
"""
    artifact_path = NOTES_DIR / "ARTIFACT_INDEX_CN.md"
    artifact_path.write_text(artifact_path.read_text(encoding="utf-8", errors="replace").rstrip() + "\n\n" + artifact_entry, encoding="utf-8")

    daily_entry = f"""
## 06:05 阶段 4：连续风格跨 split 复核 v0.1

- 为什么做：上一轮 session-level 风格没有超过 RBF，需要检查在 subject-level 跨被试切分下是否仍然没有稳定增量。
- 做了什么：新增并运行 `stage04_style_cross_split_validation_v0_1.py`，分别在 session-level 和 subject-level 下重新拟合 RBF 主参照与风格残差 Ridge，并对每个 split 使用 train-only 风格标准化。
- 用了哪些输入：B 轨道严格核心样本、原始连续风格候选宽表、阶段 3 clean task 车辆-only 代码。
- 生成了哪些输出：`04_style/stage04_style_cross_split_validation_v0_1/` 下的 tables、figures、logs，以及 `09_reports/stage04_style_cross_split_validation_user_summary_cn.md`。
- 当前结果如何：session-level RBF={run_summary['session_rbf_rmse']:.6f}、style60={run_summary['session_style60_rmse']:.6f}；subject-level RBF={run_summary['subject_rbf_rmse']:.6f}、style60={run_summary['subject_style60_rmse']:.6f}；风格有效性仍 blocked。
- 是否遇到问题：没有运行错误；解释风险是当前结果只能否定“当前表示/融合方式下的强证据”，不能否定所有未来风格表示。
- 是否需要用户决策：暂不需要；建议下一步回到车辆-only 结构化轨迹模型。
"""
    with DAILY_LOG.open("a", encoding="utf-8") as f:
        f.write(daily_entry)


def main() -> None:
    ensure_dirs()
    y, y_mask, input_values, input_time, label_time, meta = load_b_track()
    style = pd.read_csv(STYLE_RAW_TABLE)
    style = meta[["sample_id"]].merge(style, on="sample_id", how="left", validate="one_to_one")
    if style["event_uid"].isna().any():
        raise RuntimeError("style table alignment failed")

    all_metrics: list[pd.DataFrame] = []
    all_per_sample: list[pd.DataFrame] = []
    all_selection: list[pd.DataFrame] = []
    all_feature_summary: list[pd.DataFrame] = []
    prediction_cache: dict[str, dict[str, Any]] = {}

    for split_strategy in SPLIT_STRATEGIES:
        predictions, metrics, per_sample, selection, feature_summary = run_split(
            split_strategy,
            y,
            y_mask,
            input_values,
            input_time,
            label_time,
            meta,
            style,
        )
        all_metrics.append(metrics)
        all_per_sample.append(per_sample)
        all_selection.append(selection)
        all_feature_summary.append(feature_summary)
        prediction_cache[split_strategy] = {"predictions": predictions}

    metrics_all = pd.concat(all_metrics, ignore_index=True)
    per_sample_all = pd.concat(all_per_sample, ignore_index=True)
    selection_all = pd.concat(all_selection, ignore_index=True)
    feature_summary_all = pd.concat(all_feature_summary, ignore_index=True, sort=False)
    gate = build_gate_table(metrics_all)

    metrics_all.to_csv(TABLE_DIR / "style_cross_split_metrics.csv", index=False, encoding="utf-8-sig")
    per_sample_all.to_csv(TABLE_DIR / "style_cross_split_per_sample_metrics.csv", index=False, encoding="utf-8-sig")
    selection_all.to_csv(TABLE_DIR / "style_cross_split_validation_selection.csv", index=False, encoding="utf-8-sig")
    feature_summary_all.to_csv(TABLE_DIR / "style_cross_split_feature_and_split_summary.csv", index=False, encoding="utf-8-sig")
    gate.to_csv(TABLE_DIR / "style_cross_split_gate_table.csv", index=False, encoding="utf-8-sig")

    metric_fig = plot_cross_split(metrics_all)
    subject_bad = per_sample_all[
        (per_sample_all["split_strategy"] == "subject_level_split")
        & (per_sample_all["split"] == "test")
        & (per_sample_all["model_name"] == RBF_MODEL)
    ].sort_values("sample_rmse", ascending=False)
    bad_ids = subject_bad.head(min(8, len(subject_bad)))["sample_id"].astype(str).tolist()
    subject_bad_fig = FIG_DIR / "style_cross_split_subject_bad_samples_test.png"
    plot_subject_bad_samples(
        bad_ids,
        y,
        y_mask,
        label_time,
        meta,
        prediction_cache["subject_level_split"]["predictions"],
        subject_bad_fig,
        "Subject-level test bad samples: RBF vs style",
    )
    figures = {"metric_summary": metric_fig.as_posix(), "subject_bad_samples": subject_bad_fig.as_posix()}
    write_reports(metrics_all, gate, feature_summary_all, figures)

    test = metrics_all[metrics_all["split"] == "test"].set_index(["split_strategy", "model_name"])
    run_summary = {
        "run_time_local": "2026-05-13 06:05",
        "track_id": TRACK_ID,
        "window_config_id": WINDOW_ID,
        "task_sample_role": TASK_ROLE,
        "n_samples": int(len(meta)),
        "session_train_val_test": [int((meta["session_level_split"].astype(str) == s).sum()) for s in ["train", "val", "test"]],
        "subject_train_val_test": [int((meta["subject_level_split"].astype(str) == s).sum()) for s in ["train", "val", "test"]],
        "session_rbf_rmse": float(test.loc[("session_level_split", RBF_MODEL), "rmse_steer"]),
        "session_style60_rmse": float(test.loc[("session_level_split", "rbf_plus_style_last60_guard3_residual_ridge"), "rmse_steer"]),
        "subject_rbf_rmse": float(test.loc[("subject_level_split", RBF_MODEL), "rmse_steer"]),
        "subject_style60_rmse": float(test.loc[("subject_level_split", "rbf_plus_style_last60_guard3_residual_ridge"), "rmse_steer"]),
        "metrics_path": (TABLE_DIR / "style_cross_split_metrics.csv").as_posix(),
        "gate_path": (TABLE_DIR / "style_cross_split_gate_table.csv").as_posix(),
        "figures": figures,
        "server_used": False,
        "server_access_file_read": False,
        "uses_physio": False,
        "uses_eeg": False,
        "uses_emg": False,
        "uses_resp": False,
        "style_effectiveness_claim_allowed": False,
        "raw_files_modified": False,
    }
    (LOG_DIR / "style_cross_split_validation_summary.json").write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    update_transparency(run_summary)
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
