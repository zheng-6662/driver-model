# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import os
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


ROOT = Path(r"F:/data_set_process/data_process/05_rebuild_from_raw_20260511")
OUT_ROOT = ROOT / "03_baselines" / "stage03_vehicle_instability_rbf_keypoint_multihypothesis_review_v0_1"
TABLE_DIR = OUT_ROOT / "tables"
FIG_DIR = OUT_ROOT / "figures"
LOG_DIR = OUT_ROOT / "logs"
REPORT_ROOT = ROOT / "09_reports"

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_stage3_vehicle_baselines as eval_utils  # noqa: E402
import stage03_vehicle_instability_clean_task_vehicle_baselines_v0_1 as clean_v01  # noqa: E402
import stage03_vehicle_instability_keypoint_residual_vehicle_transformer_v0_1 as keypoint_v01  # noqa: E402


TRACK_ID = "B_response3s_strict_core"
SPLIT_STRATEGY = "session_level_split"
RBF_MODEL = "rbf_kernel_ridge_context_no_subject"
KEYPOINT_MODEL = "keypoint_residual_vehicle_transformer_no_subject"
SELECTOR_MODEL = "selector_logreg_rbf_keypoint_no_subject"
ORACLE_MODEL = "oracle_best_of_rbf_keypoint_upper_bound"
OUTPUT_VERSION = "stage03_vehicle_instability_rbf_keypoint_multihypothesis_review_v0_1"

KEYPOINT_DIR = ROOT / "03_baselines" / "stage03_vehicle_instability_keypoint_residual_vehicle_transformer_v0_1"
KEYPOINT_CKPT = KEYPOINT_DIR / "checkpoints" / f"{TRACK_ID}_{KEYPOINT_MODEL}_best.pt"
SELECTOR_DIR = ROOT / "03_baselines" / "stage03_vehicle_instability_rbf_keypoint_selector_v0_1"
SELECTOR_DECISIONS = SELECTOR_DIR / "tables" / "rbf_keypoint_selector_decisions.csv"

PLOT_MODELS = [
    (RBF_MODEL, "#1f77b4", "rbf"),
    (KEYPOINT_MODEL, "#2ca02c", "keypoint"),
    (SELECTOR_MODEL, "#d62728", "selector"),
    (ORACLE_MODEL, "#9467bd", "oracle"),
]


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIG_DIR, LOG_DIR, REPORT_ROOT]:
        path.mkdir(parents=True, exist_ok=True)


def row_rmse(y: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = mask & np.isfinite(y) & np.isfinite(pred)
    diff = np.where(valid, pred - y, np.nan)
    denom = np.maximum(valid.sum(axis=1), 1)
    mse = np.nansum(diff * diff, axis=1) / denom
    return np.sqrt(mse).astype(np.float32)


def load_base_track() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    manifest = pd.read_csv(keypoint_v01.TASK_MANIFEST_PATH)
    cfg = keypoint_v01.TRACKS[TRACK_ID]
    y, y_mask, input_values, input_mask, input_time, label_time, meta = keypoint_v01.load_track(TRACK_ID, cfg, manifest)
    train_idx, val_idx, test_idx = keypoint_v01.split_indices(meta)
    return y, y_mask, input_values, input_mask, input_time, label_time, meta, train_idx, val_idx, test_idx


def load_keypoint_prediction(
    input_values: np.ndarray,
    input_mask: np.ndarray,
    label_time: np.ndarray,
    meta: pd.DataFrame,
    train_idx: np.ndarray,
) -> np.ndarray:
    ckpt = torch.load(KEYPOINT_CKPT, map_location="cpu")
    x_scaled, _ = keypoint_v01.standardize_vehicle_inputs(input_values, input_mask, train_idx)
    context, _ = keypoint_v01.build_context_features(meta, train_idx)
    step = int(ckpt.get("input_downsample_step", max(1, int(round(x_scaled.shape[1] / keypoint_v01.TARGET_INPUT_TOKENS)))))
    x_model = x_scaled[:, ::step, :].copy()
    model = keypoint_v01.KeypointResidualVehicleTransformer(
        vehicle_dim=x_model.shape[2],
        context_dim=context.shape[1],
        label_time=label_time,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return keypoint_v01.predict_all(model, x_model, context, float(ckpt["label_scale"]))


def build_candidate_predictions() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    np.ndarray,
    dict[str, np.ndarray],
    pd.DataFrame,
]:
    y, y_mask, input_values, input_mask, input_time, label_time, meta, train_idx, val_idx, _ = load_base_track()
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
        raise RuntimeError(f"{RBF_MODEL} is missing from rebuilt baseline predictions")
    pred_rbf = baseline_predictions[RBF_MODEL].astype(np.float32)
    pred_keypoint = load_keypoint_prediction(input_values, input_mask, label_time, meta, train_idx).astype(np.float32)

    decisions = pd.read_csv(SELECTOR_DECISIONS)
    decisions = decisions.set_index("sample_id")
    meta_ids = meta["sample_id"].astype(str).tolist()
    missing = [sid for sid in meta_ids if sid not in decisions.index]
    if missing:
        raise RuntimeError(f"selector decisions missing {len(missing)} samples")
    selected_model = decisions.loc[meta_ids, "selected_model"].astype(str).to_numpy()
    use_keypoint_selector = selected_model == KEYPOINT_MODEL
    pred_selector = np.where(use_keypoint_selector[:, None], pred_keypoint, pred_rbf).astype(np.float32)

    rmse_rbf = row_rmse(y, pred_rbf, y_mask)
    rmse_keypoint = row_rmse(y, pred_keypoint, y_mask)
    use_keypoint_oracle = rmse_keypoint < rmse_rbf
    pred_oracle = np.where(use_keypoint_oracle[:, None], pred_keypoint, pred_rbf).astype(np.float32)
    predictions = {
        RBF_MODEL: pred_rbf,
        KEYPOINT_MODEL: pred_keypoint,
        SELECTOR_MODEL: pred_selector,
        ORACLE_MODEL: pred_oracle,
    }
    oracle_info = pd.DataFrame(
        {
            "sample_id": meta_ids,
            "split": meta[SPLIT_STRATEGY].astype(str).to_numpy(),
            "oracle_best_model": np.where(use_keypoint_oracle, KEYPOINT_MODEL, RBF_MODEL),
            "rmse_rbf": rmse_rbf,
            "rmse_keypoint": rmse_keypoint,
            "rmse_best_of_two": np.minimum(rmse_rbf, rmse_keypoint),
            "rmse_oracle_gain_over_rbf": rmse_rbf - np.minimum(rmse_rbf, rmse_keypoint),
            "selector_used_keypoint": use_keypoint_selector.astype(int),
            "oracle_used_keypoint": use_keypoint_oracle.astype(int),
        }
    )
    return y, y_mask, label_time, meta, train_idx, predictions, decisions.reset_index(), oracle_info


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
                    keypoint_v01.TRACKS[TRACK_ID]["window_config_id"],
                    large_thr,
                    difficult_thr,
                )
            )
    per_sample = pd.DataFrame(rows)
    per_sample["track_id"] = TRACK_ID
    metrics = eval_utils.aggregate_metrics(per_sample)
    metrics["track_id"] = TRACK_ID
    return metrics, per_sample


def build_choice_tables(per_sample: pd.DataFrame, decisions: pd.DataFrame, oracle_info: pd.DataFrame) -> dict[str, pd.DataFrame]:
    metrics_wide = (
        per_sample[per_sample["model_name"].isin([RBF_MODEL, KEYPOINT_MODEL, SELECTOR_MODEL, ORACLE_MODEL])]
        .pivot(index="sample_id", columns="model_name", values="sample_rmse")
        .reset_index()
    )
    for col in [RBF_MODEL, KEYPOINT_MODEL, SELECTOR_MODEL, ORACLE_MODEL]:
        metrics_wide = metrics_wide.rename(columns={col: f"rmse__{col}"})
    detail = decisions.merge(oracle_info, on=["sample_id", "split"], how="left", validate="one_to_one")
    detail = detail.merge(metrics_wide, on="sample_id", how="left", validate="one_to_one")
    detail["selector_correct_choice"] = (detail["selected_model"] == detail["oracle_best_model"]).astype(int)
    detail["selector_regret"] = detail[f"rmse__{SELECTOR_MODEL}"] - detail[f"rmse__{ORACLE_MODEL}"]
    detail["oracle_gain_over_selector"] = detail[f"rmse__{SELECTOR_MODEL}"] - detail[f"rmse__{ORACLE_MODEL}"]
    detail["oracle_gain_over_rbf"] = detail[f"rmse__{RBF_MODEL}"] - detail[f"rmse__{ORACLE_MODEL}"]

    summary_rows: list[dict[str, Any]] = []
    for split, grp in detail.groupby("split"):
        summary_rows.append(
            {
                "split": split,
                "n_samples": int(len(grp)),
                "selector_keypoint_rate": float(grp["selector_used_keypoint"].mean()),
                "oracle_keypoint_rate": float(grp["oracle_used_keypoint"].mean()),
                "selector_choice_accuracy": float(grp["selector_correct_choice"].mean()),
                "mean_selector_regret": float(grp["selector_regret"].mean()),
                "median_selector_regret": float(grp["selector_regret"].median()),
                "mean_oracle_gain_over_rbf": float(grp["oracle_gain_over_rbf"].mean()),
                "n_selector_misselected": int((grp["selector_correct_choice"] == 0).sum()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    confusion = (
        detail[detail["split"] == "test"]
        .groupby(["selected_model", "oracle_best_model"])
        .size()
        .reset_index(name="n_samples")
        .sort_values(["selected_model", "oracle_best_model"])
    )
    test_detail = detail[detail["split"] == "test"].copy()
    misselected = test_detail[test_detail["selector_correct_choice"] == 0].sort_values("selector_regret", ascending=False)
    oracle_gap = test_detail.sort_values("oracle_gain_over_rbf", ascending=False)
    return {
        "choice_detail": detail,
        "choice_summary": summary,
        "choice_confusion_test": confusion,
        "test_misselected_samples": misselected,
        "test_oracle_gap_samples": oracle_gap,
    }


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
    choice_detail: pd.DataFrame,
    title: str,
) -> None:
    idxs = sample_indices(meta, sample_ids)[:12]
    if not idxs:
        return
    choice_map = choice_detail.set_index("sample_id").to_dict("index")
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
            style = "--" if model_name in [SELECTOR_MODEL, ORACLE_MODEL] else "-"
            alpha = 0.85 if model_name in [SELECTOR_MODEL, ORACLE_MODEL] else 0.70
            width = 1.3 if model_name in [SELECTOR_MODEL, ORACLE_MODEL] else 1.0
            ax.plot(label_time[valid], pred[valid], color=color, linestyle=style, linewidth=width, alpha=alpha, label=label)
        choice = choice_map.get(sid, {})
        short_id = sid.split("__")[-2] if "__" in sid else sid[-10:]
        selected = str(choice.get("selected_model", "")).replace("_context_no_subject", "").replace("_vehicle_transformer_no_subject", "")
        oracle = str(choice.get("oracle_best_model", "")).replace("_context_no_subject", "").replace("_vehicle_transformer_no_subject", "")
        regret = choice.get("selector_regret", np.nan)
        ax.set_title(f"{short_id} sel={selected[:10]} oracle={oracle[:10]} regret={regret:.3f}", fontsize=8)
        ax.grid(True, alpha=0.22)
    handles, labels = axes_arr[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=9)
    fig.suptitle(title, fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_choice_confusion(path: Path, confusion: pd.DataFrame) -> None:
    labels = [RBF_MODEL, KEYPOINT_MODEL]
    mat = np.zeros((2, 2), dtype=int)
    for _, row in confusion.iterrows():
        if row["selected_model"] in labels and row["oracle_best_model"] in labels:
            mat[labels.index(row["selected_model"]), labels.index(row["oracle_best_model"])] = int(row["n_samples"])
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(2), ["oracle rbf", "oracle keypoint"], rotation=20, ha="right")
    ax.set_yticks(range(2), ["select rbf", "select keypoint"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center", fontsize=12)
    ax.set_title("Test selector choice confusion")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_oracle_gap(path: Path, oracle_gap: pd.DataFrame) -> None:
    top = oracle_gap.head(15).copy()
    labels = [sid.split("__")[-2] if "__" in sid else sid[-8:] for sid in top["sample_id"].astype(str)]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.bar(np.arange(len(top)), top["oracle_gain_over_rbf"].to_numpy(dtype=float), color="#9467bd", alpha=0.82)
    ax.set_xticks(np.arange(len(top)), labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("RMSE gain over RBF")
    ax.set_title("Top oracle best-of-two gains on test")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_reports(metrics: pd.DataFrame, choice_summary: pd.DataFrame, figures: dict[str, str]) -> None:
    test = metrics[metrics["split"] == "test"].set_index("model_name")
    choice_test = choice_summary[choice_summary["split"] == "test"].iloc[0]

    def val(model: str, col: str) -> float:
        return float(test.loc[model, col])

    user = f"""# 阶段 3 用户查看版：RBF/keypoint 多候选车辆-only 复盘 v0.1

## 这个阶段为什么做

上一轮发现 RBF 整体 RMSE 稳，但 keypoint+residual 能修复一部分错侧和大幅响应。这个阶段不训练新模型，只把 RBF、keypoint、train/val selector 和 oracle best-of-two 放到同一套表和图里，判断“多候选车辆-only”是否值得继续。

## 这个阶段检查了什么

- RBF、keypoint、selector、oracle 在同一 test 集上的整体误差和物理指标。
- selector 什么时候选对，什么时候选错。
- oracle 上限和可部署 selector 之间还有多大差距。
- 固定样本图、selector 坏样本图和 oracle 增益图。

## 目前发现了什么

- RBF：RMSE={val(RBF_MODEL, 'rmse_steer'):.6f}，错侧率={val(RBF_MODEL, 'wrong_side_rate'):.3f}，大幅响应召回={val(RBF_MODEL, 'large_response_recall'):.3f}。
- keypoint：RMSE={val(KEYPOINT_MODEL, 'rmse_steer'):.6f}，错侧率={val(KEYPOINT_MODEL, 'wrong_side_rate'):.3f}，大幅响应召回={val(KEYPOINT_MODEL, 'large_response_recall'):.3f}。
- selector：RMSE={val(SELECTOR_MODEL, 'rmse_steer'):.6f}，错侧率={val(SELECTOR_MODEL, 'wrong_side_rate'):.3f}，大幅响应召回={val(SELECTOR_MODEL, 'large_response_recall'):.3f}。
- oracle best-of-two：RMSE={val(ORACLE_MODEL, 'rmse_steer'):.6f}，这是事后上限，不能部署，但说明两个候选确实互补。
- test 上 selector 选择准确率={float(choice_test['selector_choice_accuracy']):.3f}，平均选择后悔={float(choice_test['mean_selector_regret']):.6f}。

## 哪些结果可信

可信的是：RBF 和 keypoint 在同一数据、同一 split、同一评价指标下确实有互补；selector 目前能改善错侧、大幅响应召回和困难 top20 RMSE，但整体 RMSE 还没有稳定超过 RBF。

## 哪些结果还不能下结论

不能把 oracle 当成真实模型效果；也不能说车辆-only 已经解决，更不能据此进入连续风格、生理或 EEG 有效性结论。当前只说明多候选方向值得继续，但 selector 还需要更强的可靠性特征或结构。

## 下一阶段是否可以继续

可以继续阶段 3，但方向应是正式多假设/可靠性车辆-only，而不是直接加生理。下一步需要让模型自己输出多个候选和可靠性，而不是只在两个已训练候选之间做简单二选一。

## 推荐优先查看哪些图和表

1. `{figures['fixed']}`
2. `{figures['selector_bad']}`
3. `{figures['oracle_gap']}`
4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_multihypothesis_review_v0_1/tables/multihypothesis_metrics.csv`
5. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_multihypothesis_review_v0_1/tables/test_selector_misselected_samples.csv`
"""
    tech = f"""# 阶段 3 技术报告：RBF/keypoint 多候选车辆-only 复盘 v0.1

## 范围

- 轨道：`{TRACK_ID}`。
- 候选：`{RBF_MODEL}` 与 `{KEYPOINT_MODEL}`。
- 可部署策略：复用上一轮 train/val logistic selector，test 只最终评价。
- 上限策略：oracle best-of-two，仅用于诊断，不作为可部署结果。
- 未使用：subject ID、生理、脑电、连续风格、服务器、服务器密码文件。

## 主要 test 指标

| 模型 | RMSE | 错侧率 | 大幅召回 | 困难 top20 RMSE |
|---|---:|---:|---:|---:|
| RBF | {val(RBF_MODEL, 'rmse_steer'):.6f} | {val(RBF_MODEL, 'wrong_side_rate'):.3f} | {val(RBF_MODEL, 'large_response_recall'):.3f} | {val(RBF_MODEL, 'difficult_top20_rmse'):.6f} |
| keypoint | {val(KEYPOINT_MODEL, 'rmse_steer'):.6f} | {val(KEYPOINT_MODEL, 'wrong_side_rate'):.3f} | {val(KEYPOINT_MODEL, 'large_response_recall'):.3f} | {val(KEYPOINT_MODEL, 'difficult_top20_rmse'):.6f} |
| selector | {val(SELECTOR_MODEL, 'rmse_steer'):.6f} | {val(SELECTOR_MODEL, 'wrong_side_rate'):.3f} | {val(SELECTOR_MODEL, 'large_response_recall'):.3f} | {val(SELECTOR_MODEL, 'difficult_top20_rmse'):.6f} |
| oracle | {val(ORACLE_MODEL, 'rmse_steer'):.6f} | {val(ORACLE_MODEL, 'wrong_side_rate'):.3f} | {val(ORACLE_MODEL, 'large_response_recall'):.3f} | {val(ORACLE_MODEL, 'difficult_top20_rmse'):.6f} |

## 选择器诊断

- test selector choice accuracy={float(choice_test['selector_choice_accuracy']):.6f}
- test selector keypoint rate={float(choice_test['selector_keypoint_rate']):.6f}
- test oracle keypoint rate={float(choice_test['oracle_keypoint_rate']):.6f}
- test mean selector regret={float(choice_test['mean_selector_regret']):.6f}
- test mean oracle gain over RBF={float(choice_test['mean_oracle_gain_over_rbf']):.6f}

## 结论

RBF/keypoint 的 oracle 上限较明显，证明两者在样本层有互补；但 train/val selector 还不能把这个上限稳定转化为整体 RMSE 收益。当前证据支持继续做车辆-only 多假设/可靠性路线，不支持进入连续风格、生理或 EEG 有效性结论。
"""
    (REPORT_ROOT / "stage03_vehicle_instability_rbf_keypoint_multihypothesis_review_user_summary_cn.md").write_text(user, encoding="utf-8")
    (REPORT_ROOT / "stage03_vehicle_instability_rbf_keypoint_multihypothesis_review_v0_1_cn.md").write_text(tech, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    y, y_mask, label_time, meta, train_idx, predictions, decisions, oracle_info = build_candidate_predictions()
    metrics, per_sample = evaluate_predictions(y, y_mask, label_time, meta, train_idx, predictions)
    tables = build_choice_tables(per_sample, decisions, oracle_info)

    metrics.to_csv(TABLE_DIR / "multihypothesis_metrics.csv", index=False, encoding="utf-8-sig")
    per_sample.to_csv(TABLE_DIR / "multihypothesis_per_sample_metrics.csv", index=False, encoding="utf-8-sig")
    for name, table in tables.items():
        table.to_csv(TABLE_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")

    test_detail = tables["choice_detail"][tables["choice_detail"]["split"] == "test"].copy()
    fixed_ids = test_detail["sample_id"].astype(str).head(12).tolist()
    selector_bad_ids = (
        test_detail.sort_values(f"rmse__{SELECTOR_MODEL}", ascending=False)["sample_id"].astype(str).head(12).tolist()
    )
    oracle_gap_ids = test_detail.sort_values("oracle_gain_over_rbf", ascending=False)["sample_id"].astype(str).head(12).tolist()
    pd.DataFrame({"track_id": TRACK_ID, "sample_id": fixed_ids}).to_csv(TABLE_DIR / "fixed_plot_samples.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"track_id": TRACK_ID, "sample_id": selector_bad_ids}).to_csv(TABLE_DIR / "selector_bad_plot_samples.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"track_id": TRACK_ID, "sample_id": oracle_gap_ids}).to_csv(TABLE_DIR / "oracle_gap_plot_samples.csv", index=False, encoding="utf-8-sig")

    fig_fixed = FIG_DIR / "multihypothesis_fixed_predictions_test.png"
    fig_selector_bad = FIG_DIR / "multihypothesis_selector_bad_samples_test.png"
    fig_oracle_gap = FIG_DIR / "multihypothesis_oracle_gap_samples_test.png"
    fig_confusion = FIG_DIR / "selector_choice_confusion_test.png"
    fig_gap_bar = FIG_DIR / "oracle_gap_top_samples.png"
    plot_prediction_grid(fig_fixed, fixed_ids, y, y_mask, label_time, meta, predictions, tables["choice_detail"], "Fixed test samples: RBF/keypoint/selector/oracle")
    plot_prediction_grid(fig_selector_bad, selector_bad_ids, y, y_mask, label_time, meta, predictions, tables["choice_detail"], "Worst selector test samples")
    plot_prediction_grid(fig_oracle_gap, oracle_gap_ids, y, y_mask, label_time, meta, predictions, tables["choice_detail"], "Top oracle gains over RBF")
    plot_choice_confusion(fig_confusion, tables["choice_confusion_test"])
    plot_oracle_gap(fig_gap_bar, tables["test_oracle_gap_samples"])

    figures = {
        "fixed": str(fig_fixed).replace("\\", "/"),
        "selector_bad": str(fig_selector_bad).replace("\\", "/"),
        "oracle_gap": str(fig_oracle_gap).replace("\\", "/"),
        "confusion": str(fig_confusion).replace("\\", "/"),
        "gap_bar": str(fig_gap_bar).replace("\\", "/"),
    }
    write_reports(metrics, tables["choice_summary"], figures)
    summary = {
        "output_version": OUTPUT_VERSION,
        "track_id": TRACK_ID,
        "candidate_models": [RBF_MODEL, KEYPOINT_MODEL],
        "selector_model": SELECTOR_MODEL,
        "oracle_model": ORACLE_MODEL,
        "server_used": False,
        "credential_file_read": False,
        "raw_files_modified": False,
        "metrics_path": str(TABLE_DIR / "multihypothesis_metrics.csv").replace("\\", "/"),
        "choice_summary_path": str(TABLE_DIR / "choice_summary.csv").replace("\\", "/"),
        "figures": figures,
    }
    (LOG_DIR / "multihypothesis_review_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
