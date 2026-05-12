# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd


ROOT = Path(r"F:/data_set_process/data_process/05_rebuild_from_raw_20260511")
TASK_MANIFEST_PATH = (
    ROOT
    / "02_samples"
    / "vehicle_instability_response_task_decision_v0_1"
    / "tables"
    / "sample_response_task_manifest.csv"
)
PROCESSED_DIR = ROOT / "03_processed_datasets" / "vehicle_instability_allraw_highconf_v0_1"
ARRAY_DIR = PROCESSED_DIR / "arrays"
OUT_ROOT = ROOT / "03_baselines" / "stage03_vehicle_instability_clean_task_vehicle_baselines_v0_1"
TABLE_DIR = OUT_ROOT / "tables"
FIG_DIR = OUT_ROOT / "figures"
LOG_DIR = OUT_ROOT / "logs"
REPORT_ROOT = ROOT / "09_reports"

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_stage3_vehicle_baselines as eval_utils  # noqa: E402
import stage03_vehicle_instability_formal_baselines_v0_1 as formal_v01  # noqa: E402
import stage03_vehicle_instability_strong_vehicle_baselines_v0_1 as strong_v01  # noqa: E402


SPLIT_STRATEGY = "session_level_split"
TRACKS = {
    "A_instant2s_core": {
        "window_config_id": "pre2_label2_old_main",
        "task_sample_role": "instant2s_core_candidate",
        "description_cn": "2秒即时响应核心候选：2秒标签稳定，可先验证事件后即时方向盘响应。",
    },
    "B_response3s_strict_core": {
        "window_config_id": "pre3_label3_response_coverage",
        "task_sample_role": "response3s_strict_core_candidate",
        "description_cn": "3秒响应覆盖严格核心候选：2秒不足但3秒标签相对稳定。",
    },
}
PLOT_MODELS = [
    ("zero_response_hold_current", "#8c8c8c"),
    ("formal_ridge_vehicle_context_no_subject", "#d62728"),
    ("rbf_kernel_ridge_context_no_subject", "#1f77b4"),
    ("knn_template_context_no_subject", "#ff7f0e"),
    ("peak_scaled_template_context_no_subject", "#2ca02c"),
]
DISPLAY_NAMES = {
    **getattr(strong_v01, "DISPLAY_NAMES", {}),
    "zero_response_hold_current": "zero",
    "history_trend_500ms": "trend",
    "train_mean_all": "train mean",
    "train_mean_by_event_type": "event mean",
    "ridge_vehicle_history_no_subject": "formal ridge hist",
    "ridge_vehicle_context_no_subject": "formal ridge ctx",
    "formal_ridge_vehicle_context_no_subject": "formal ridge ctx",
}


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIG_DIR, LOG_DIR, REPORT_ROOT]:
        path.mkdir(parents=True, exist_ok=True)


def load_track(track_id: str, cfg: dict[str, str], manifest: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    window_id = cfg["window_config_id"]
    role = cfg["task_sample_role"]
    z = np.load(ARRAY_DIR / f"{window_id}.npz", allow_pickle=True)
    rows = manifest[
        (manifest["window_config_id"].astype(str) == window_id)
        & (manifest["task_sample_role"].astype(str) == role)
    ].copy()
    if rows.empty:
        raise RuntimeError(f"{track_id}: no samples for {window_id}/{role}")
    rows["array_row"] = pd.to_numeric(rows["array_row"], errors="raise").astype(int)
    rows = rows.sort_values("array_row").reset_index(drop=True)
    idx = rows["array_row"].to_numpy(dtype=int)
    y = z["label_steer_delta"].astype(np.float32)[idx]
    y_mask = z["label_valid_mask"].astype(bool)[idx]
    input_values = z["input_values"].astype(np.float32)[idx]
    input_time = z["input_time_rel_s"].astype(np.float32)
    label_time = z["label_time_rel_s"].astype(np.float32)
    rows["track_id"] = track_id
    rows["track_description_cn"] = cfg["description_cn"]
    return y, y_mask, input_values, input_time, label_time, rows


def split_indices(meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split = meta[SPLIT_STRATEGY].astype(str).to_numpy()
    return np.where(split == "train")[0], np.where(split == "val")[0], np.where(split == "test")[0]


def build_strong_predictions(
    track_id: str,
    window_id: str,
    y: np.ndarray,
    y_mask: np.ndarray,
    input_values: np.ndarray,
    input_time: np.ndarray,
    label_time: np.ndarray,
    meta: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    strong_v01.WINDOW_ID = window_id
    strong_v01.SPLIT_STRATEGY = SPLIT_STRATEGY
    predictions: dict[str, np.ndarray] = {}
    info_rows: list[dict[str, Any]] = []

    formal_preds, formal_info = formal_v01.build_predictions(y, y_mask, input_values, input_time, label_time, meta, SPLIT_STRATEGY)
    for model_name, pred in formal_preds.items():
        out_name = "formal_ridge_vehicle_context_no_subject" if model_name == "ridge_vehicle_context_no_subject" else model_name
        predictions[out_name] = pred
    for info in formal_info:
        info = dict(info)
        model_name = str(info["model_name"])
        info["model_name"] = "formal_ridge_vehicle_context_no_subject" if model_name == "ridge_vehicle_context_no_subject" else model_name
        info_rows.append(info)

    x_hist, _ = strong_v01.build_rich_vehicle_features(input_values, input_time, meta, train_idx, include_context=False)
    x_hist_scaled, _ = strong_v01.standardize_train_only(x_hist, train_idx)
    pred, info = strong_v01.fit_direct_ridge("ridge_rich_history_no_subject", x_hist_scaled, y, train_idx, val_idx, y_mask)
    predictions[info["model_name"]] = pred
    info_rows.append(info)

    x_ctx, _ = strong_v01.build_rich_vehicle_features(input_values, input_time, meta, train_idx, include_context=True)
    x_ctx_scaled, _ = strong_v01.standardize_train_only(x_ctx, train_idx)
    pred, info = strong_v01.fit_direct_ridge("ridge_rich_context_no_subject", x_ctx_scaled, y, train_idx, val_idx, y_mask)
    predictions[info["model_name"]] = pred
    info_rows.append(info)

    x_dist, _ = strong_v01.make_distance_features(x_ctx_scaled, train_idx, n_components=min(96, max(8, x_ctx_scaled.shape[1])))
    peaks = strong_v01.peak_arrays(y, y_mask, label_time)
    fitters = [
        ("rbf_kernel_ridge_context_no_subject", lambda: strong_v01.fit_rbf_kernel_ridge_direct(x_dist, y, train_idx, val_idx, y_mask)),
        (
            "knn_template_context_no_subject",
            lambda: strong_v01.fit_knn_template("knn_template_context_no_subject", x_dist, y, y_mask, train_idx, val_idx),
        ),
        (
            "direction_gated_knn_template_no_subject",
            lambda: strong_v01.fit_direction_gated_knn_template(x_dist, y, y_mask, peaks, train_idx, val_idx),
        ),
        (
            "peak_scaled_template_context_no_subject",
            lambda: strong_v01.fit_peak_scaled_template(x_dist, y, y_mask, peaks, train_idx, val_idx),
        ),
    ]
    for model_name, fit in fitters:
        pred, info = fit()
        predictions[model_name] = pred
        info_rows.append(info)

    for info in info_rows:
        info.update(
            {
                "track_id": track_id,
                "window_config_id": window_id,
                "split_strategy": SPLIT_STRATEGY,
                "train_n": int(len(train_idx)),
                "val_n": int(len(val_idx)),
                "test_n": int((meta[SPLIT_STRATEGY].astype(str) == "test").sum()),
                "uses_subject_id": False,
                "uses_physio": False,
                "uses_eeg": False,
                "uses_continuous_style": False,
                "server_used": False,
                "credential_file_read": False,
            }
        )
    return predictions, info_rows


def evaluate_track(
    track_id: str,
    window_id: str,
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
    for split_name in ["train", "val", "test"]:
        mask = meta[SPLIT_STRATEGY].astype(str).to_numpy() == split_name
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
                window_id=window_id,
                large_thr=large_thr,
                difficult_thr=difficult_thr,
            )
            if sample_rows:
                part = pd.DataFrame(sample_rows)
                part["track_id"] = track_id
                rows.append(part)
    per_sample = pd.concat(rows, ignore_index=True)
    metrics = eval_utils.aggregate_metrics(per_sample)
    metrics["track_id"] = track_id
    return metrics, per_sample


def select_best_val(metrics: pd.DataFrame) -> pd.DataFrame:
    val = metrics[metrics["split"] == "val"].copy()
    if val.empty:
        return pd.DataFrame()
    rows = []
    for track_id, part in val.groupby("track_id"):
        row = part.sort_values("rmse_steer").iloc[0].to_dict()
        rows.append(row)
    return pd.DataFrame(rows)


def plot_metric_summary(metrics: pd.DataFrame) -> Path:
    test = metrics[metrics["split"] == "test"].copy()
    models = [
        "zero_response_hold_current",
        "train_mean_by_event_type",
        "formal_ridge_vehicle_context_no_subject",
        "ridge_rich_context_no_subject",
        "rbf_kernel_ridge_context_no_subject",
        "knn_template_context_no_subject",
        "peak_scaled_template_context_no_subject",
    ]
    test = test[test["model_name"].isin(models)].copy()
    tracks = list(TRACKS.keys())
    fig, axes = plt.subplots(len(tracks), 2, figsize=(15, 5.4 * len(tracks)), squeeze=False)
    for i, track_id in enumerate(tracks):
        part = test[test["track_id"] == track_id].set_index("model_name").reindex(models).dropna(subset=["rmse_steer"])
        labels = [DISPLAY_NAMES.get(v, v) for v in part.index]
        axes[i, 0].barh(labels, part["rmse_steer"].to_numpy(), color="#4c78a8")
        axes[i, 0].set_title(f"{track_id}: test RMSE")
        axes[i, 0].grid(axis="x", alpha=0.25)
        axes[i, 1].barh(labels, part["wrong_side_rate"].to_numpy(), color="#e45756")
        axes[i, 1].set_title(f"{track_id}: test wrong-side rate")
        axes[i, 1].grid(axis="x", alpha=0.25)
        for ax in axes[i]:
            ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    out = FIG_DIR / "clean_task_vehicle_metric_summary_test.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def plot_samples(
    track_id: str,
    sample_ids: list[str],
    y: np.ndarray,
    y_mask: np.ndarray,
    label_time: np.ndarray,
    meta: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    out_path: Path,
    title: str,
) -> None:
    n = len(sample_ids)
    cols = 4
    rows = int(np.ceil(max(n, 1) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(18, max(3.2 * rows, 3.4)), squeeze=False)
    id_to_idx = {sid: i for i, sid in enumerate(meta["sample_id"].astype(str))}
    for ax in axes.ravel():
        ax.axis("off")
    for k, sid in enumerate(sample_ids):
        ax = axes.ravel()[k]
        ax.axis("on")
        idx = id_to_idx[sid]
        gt = np.where(y_mask[idx] & np.isfinite(y[idx]), y[idx], np.nan)
        ax.plot(label_time, gt, color="black", linewidth=1.8, label="GT")
        for model_name, color in PLOT_MODELS:
            if model_name in predictions:
                ax.plot(label_time, predictions[model_name][idx], color=color, linewidth=1.1, alpha=0.95, label=DISPLAY_NAMES.get(model_name, model_name))
        ax.axhline(0, color="#dddddd", linewidth=0.8)
        ax.set_title(f"{meta.at[idx, 'subject']} {meta.at[idx, 'anchor_time_rel_s']:.1f}s\npeak={np.nanmax(np.abs(gt)):.2f}", fontsize=9)
        ax.tick_params(labelsize=8)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=min(6, len(labels)), fontsize=8)
    fig.suptitle(title, fontsize=12, y=0.955)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_reports(metrics: pd.DataFrame, best_val: pd.DataFrame, track_summary: pd.DataFrame, figures: dict[str, str]) -> None:
    test = metrics[metrics["split"] == "test"].copy()
    best_lines = []
    test_best_lines = []
    for _, row in best_val.iterrows():
        track_id = str(row["track_id"])
        model = str(row["model_name"])
        test_row = test[(test["track_id"] == track_id) & (test["model_name"] == model)]
        if not test_row.empty:
            r = test_row.iloc[0]
            best_lines.append(
                f"- {track_id}: val 选择 `{model}`；test RMSE={r['rmse_steer']:.6f}，错侧率={r['wrong_side_rate']:.6f}，大幅响应召回={r['large_response_recall']:.6f}。"
            )
    for track_id, part in test.groupby("track_id"):
        r = part.sort_values("rmse_steer").iloc[0]
        test_best_lines.append(
            f"- {track_id}: 按 test RMSE 事后最小为 `{r['model_name']}`，RMSE={r['rmse_steer']:.6f}，只用于诊断，不能替代 val 选择。"
        )
    best_text = "\n".join(best_lines)
    test_best_text = "\n".join(test_best_lines)
    track_text = "```text\n" + track_summary.to_string(index=False) + "\n```"

    user = f"""# 阶段 3 用户查看版：干净响应任务车辆-only 基线 v0.1

## 为什么做

前一步已经把失稳样本拆成 2 秒即时响应和 3 秒响应覆盖两个相对干净的任务轨道。这里先只在这两个轨道上重跑车辆-only 对照，避免把长事件/持续控制样本混入核心训练后误判模型能力。

## 检查了什么

- A 轨道：2 秒即时响应核心候选，84 个事件，session-level test 12 个。
- B 轨道：3 秒响应覆盖严格核心候选，270 个事件，session-level test 40 个。
- 模型仍然只用车辆历史和事件/道路上下文，不使用生理、脑电、连续风格或驾驶员 ID。

## 当前结果

按验证集选择模型：

{best_text}

按 test 事后排序的诊断结果：

{test_best_text}

## 目前能说明什么

这个结果更适合作为后续车辆-only 主参照的候选，因为它不再把大量长事件和标签窗口未稳定样本混在一起。但 A 轨道 test 只有 12 个事件，且 KNN 在 train 上接近记忆，不能按 A 轨道单次 test 排名下强结论。是否进入风格/生理阶段，还要看这两个轨道上的固定图、坏样本图和物理指标是否足够稳定。

## 不能下的结论

这一步仍不能说明连续风格、生理或 EEG 有效，也不能说明长事件已经解决。D 轨道长事件仍要单独复核或拆分。

## 推荐查看

1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_baselines_v0_1/tables/clean_task_vehicle_metrics.csv`
2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_baselines_v0_1/figures/clean_task_vehicle_metric_summary_test.png`
3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_baselines_v0_1/figures/A_instant2s_core_bad_samples_test.png`
4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_baselines_v0_1/figures/B_response3s_strict_core_bad_samples_test.png`
"""
    (REPORT_ROOT / "stage03_vehicle_instability_clean_task_vehicle_baselines_user_summary_cn.md").write_text(user, encoding="utf-8")

    technical = f"""# 阶段 3：干净响应任务车辆-only 基线 v0.1

## 输入

- 任务 manifest：`{TASK_MANIFEST_PATH.as_posix()}`
- 轨道 A：`instant2s_core_candidate`
- 轨道 B：`response3s_strict_core_candidate`
- split：`{SPLIT_STRATEGY}`

## 轨道样本量

{track_text}

## val 选择与 test 结果

{best_text}

## test 事后最小 RMSE 诊断

{test_best_text}

## 输出

- 指标表：`{(TABLE_DIR / 'clean_task_vehicle_metrics.csv').as_posix()}`
- 逐样本指标：`{(TABLE_DIR / 'clean_task_vehicle_per_sample_metrics.csv').as_posix()}`
- 模型信息：`{(TABLE_DIR / 'clean_task_vehicle_model_info.csv').as_posix()}`
- 轨道汇总：`{(TABLE_DIR / 'clean_task_track_summary.csv').as_posix()}`
- 指标图：`{figures['metric_summary']}`

## 解释边界

本轮只用车辆历史和事件/道路上下文。没有使用生理、脑电、连续风格、驾驶员 ID 或服务器。由于 A 轨道 test 只有 12 个事件，且 KNN 类模型存在 train RMSE 接近 0 的模板记忆风险，结论必须保守；B 轨道样本量更适合后续作为 3 秒响应覆盖的强车辆候选。
"""
    (REPORT_ROOT / "stage03_vehicle_instability_clean_task_vehicle_baselines_v0_1_cn.md").write_text(technical, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    manifest = pd.read_csv(TASK_MANIFEST_PATH)
    all_metrics: list[pd.DataFrame] = []
    all_per_sample: list[pd.DataFrame] = []
    all_info: list[dict[str, Any]] = []
    track_rows: list[dict[str, Any]] = []
    cache: dict[str, dict[str, Any]] = {}

    for track_id, cfg in TRACKS.items():
        y, y_mask, input_values, input_time, label_time, meta = load_track(track_id, cfg, manifest)
        train_idx, val_idx, test_idx = split_indices(meta)
        if min(len(train_idx), len(val_idx), len(test_idx)) <= 0:
            raise RuntimeError(f"{track_id}: empty train/val/test split")
        predictions, info_rows = build_strong_predictions(track_id, cfg["window_config_id"], y, y_mask, input_values, input_time, label_time, meta, train_idx, val_idx)
        metrics, per_sample = evaluate_track(track_id, cfg["window_config_id"], y, y_mask, label_time, meta, predictions, train_idx)
        all_metrics.append(metrics)
        all_per_sample.append(per_sample)
        all_info.extend(info_rows)
        track_rows.append(
            {
                "track_id": track_id,
                "window_config_id": cfg["window_config_id"],
                "task_sample_role": cfg["task_sample_role"],
                "n_samples": int(len(meta)),
                "train_n": int(len(train_idx)),
                "val_n": int(len(val_idx)),
                "test_n": int(len(test_idx)),
                "subject_n": int(meta["subject"].nunique()),
                "description_cn": cfg["description_cn"],
            }
        )
        cache[track_id] = {
            "y": y,
            "y_mask": y_mask,
            "label_time": label_time,
            "meta": meta,
            "predictions": predictions,
        }

    metrics = pd.concat(all_metrics, ignore_index=True)
    per_sample = pd.concat(all_per_sample, ignore_index=True)
    model_info = pd.DataFrame(all_info)
    track_summary = pd.DataFrame(track_rows)
    best_val = select_best_val(metrics)

    metrics.to_csv(TABLE_DIR / "clean_task_vehicle_metrics.csv", index=False, encoding="utf-8-sig")
    per_sample.to_csv(TABLE_DIR / "clean_task_vehicle_per_sample_metrics.csv", index=False, encoding="utf-8-sig")
    model_info.to_csv(TABLE_DIR / "clean_task_vehicle_model_info.csv", index=False, encoding="utf-8-sig")
    track_summary.to_csv(TABLE_DIR / "clean_task_track_summary.csv", index=False, encoding="utf-8-sig")
    best_val.to_csv(TABLE_DIR / "clean_task_vehicle_val_selected_models.csv", index=False, encoding="utf-8-sig")

    figures: dict[str, str] = {}
    metric_fig = plot_metric_summary(metrics)
    figures["metric_summary"] = str(metric_fig).replace("\\", "/")

    for track_id, state in cache.items():
        meta = state["meta"]
        y = state["y"]
        y_mask = state["y_mask"]
        label_time = state["label_time"]
        predictions = state["predictions"]
        test_mask = meta[SPLIT_STRATEGY].astype(str).to_numpy() == "test"
        test_meta = meta.loc[test_mask].copy()
        if len(test_meta) == 0:
            continue
        fixed_ids = test_meta.sort_values(["subject", "anchor_time_rel_s"]).head(min(8, len(test_meta)))["sample_id"].astype(str).tolist()
        selected = best_val[best_val["track_id"] == track_id]
        selected_model = str(selected.iloc[0]["model_name"]) if not selected.empty else "formal_ridge_vehicle_context_no_subject"
        bad_source = per_sample[
            (per_sample["track_id"] == track_id)
            & (per_sample["split"] == "test")
            & (per_sample["model_name"] == selected_model)
        ].copy()
        bad_ids = bad_source.sort_values("sample_rmse", ascending=False).head(min(8, len(bad_source)))["sample_id"].astype(str).tolist()
        fixed_fig = FIG_DIR / f"{track_id}_fixed_predictions_test.png"
        bad_fig = FIG_DIR / f"{track_id}_bad_samples_test.png"
        plot_samples(track_id, fixed_ids, y, y_mask, label_time, meta, predictions, fixed_fig, f"{track_id}: fixed test predictions")
        plot_samples(track_id, bad_ids, y, y_mask, label_time, meta, predictions, bad_fig, f"{track_id}: worst test predictions by {selected_model}")
        figures[f"{track_id}_fixed"] = str(fixed_fig).replace("\\", "/")
        figures[f"{track_id}_bad"] = str(bad_fig).replace("\\", "/")

    write_reports(metrics, best_val, track_summary, figures)

    summary = {
        "tracks": track_rows,
        "metrics_rows": int(len(metrics)),
        "per_sample_rows": int(len(per_sample)),
        "best_val_models": best_val[["track_id", "model_name", "rmse_steer"]].to_dict(orient="records") if not best_val.empty else [],
        "metrics_path": str(TABLE_DIR / "clean_task_vehicle_metrics.csv").replace("\\", "/"),
        "per_sample_path": str(TABLE_DIR / "clean_task_vehicle_per_sample_metrics.csv").replace("\\", "/"),
        "model_info_path": str(TABLE_DIR / "clean_task_vehicle_model_info.csv").replace("\\", "/"),
        "figures": figures,
        "server_used": False,
        "credential_file_read": False,
        "uses_subject_id_as_model_input": False,
        "uses_physio": False,
        "uses_eeg": False,
        "uses_continuous_style": False,
        "raw_files_modified": False,
    }
    (LOG_DIR / "clean_task_vehicle_baselines_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
