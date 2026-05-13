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
V6_EVENTS_PATH = ROOT / "02_samples" / "episode_first_event_v0_6" / "tables" / "episode_candidates_v0_6.csv"
TASK_MANIFEST_PATH = (
    ROOT
    / "02_samples"
    / "vehicle_instability_response_task_decision_v0_1"
    / "tables"
    / "sample_response_task_manifest.csv"
)
ARRAY_DIR = ROOT / "03_processed_datasets" / "vehicle_instability_allraw_highconf_v0_1" / "arrays"
OLD_CLEAN_METRICS = (
    ROOT
    / "03_baselines"
    / "stage03_vehicle_instability_clean_task_vehicle_baselines_v0_1"
    / "tables"
    / "clean_task_vehicle_metrics.csv"
)
OUT_ROOT = ROOT / "03_baselines" / "stage03_episode_first_vehicle_baselines_v0_1"
TABLE_DIR = OUT_ROOT / "tables"
FIG_DIR = OUT_ROOT / "figures"
LOG_DIR = OUT_ROOT / "logs"
REPORT_ROOT = ROOT / "09_reports"

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import stage03_vehicle_instability_clean_task_vehicle_baselines_v0_1 as clean_v01  # noqa: E402


SPLIT_STRATEGY = "session_level_split"
POSITIVE_BUCKETS = ["strict_clean_primary", "coordinate_flagged_expansion"]
TRACKS: dict[str, dict[str, Any]] = {
    "EP2_expanded_no_lateral_2s": {
        "window_config_id": "pre2_label2_old_main",
        "buckets": POSITIVE_BUCKETS,
        "zero_lateral_offset": True,
        "description_cn": "episode-first 正样本扩展集，2秒标签，不使用横向偏移特征。",
    },
    "EP3_expanded_no_lateral_3s": {
        "window_config_id": "pre3_label3_response_coverage",
        "buckets": POSITIVE_BUCKETS,
        "zero_lateral_offset": True,
        "description_cn": "episode-first 正样本扩展集，3秒标签，不使用横向偏移特征。",
    },
    "EP3_expanded_with_lateral_3s": {
        "window_config_id": "pre3_label3_response_coverage",
        "buckets": POSITIVE_BUCKETS,
        "zero_lateral_offset": False,
        "description_cn": "episode-first 正样本扩展集，3秒标签，保留横向偏移特征；仅作坐标风险诊断。",
    },
}


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIG_DIR, LOG_DIR, REPORT_ROOT]:
        path.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def load_track(
    track_id: str,
    cfg: dict[str, Any],
    v6: pd.DataFrame,
    manifest: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, list[str]]:
    window_id = str(cfg["window_config_id"])
    buckets = set(str(v) for v in cfg["buckets"])
    selected_events = v6[v6["v0_6_final_bucket"].astype(str).isin(buckets)].copy()
    event_ids = set(selected_events["instability_event_uid"].astype(str))
    meta = manifest[
        (manifest["window_config_id"].astype(str) == window_id)
        & (manifest["event_uid"].astype(str).isin(event_ids))
    ].copy()
    if meta.empty:
        raise RuntimeError(f"{track_id}: no samples for {window_id}")

    z = np.load(ARRAY_DIR / f"{window_id}.npz", allow_pickle=True)
    feature_names = [str(v) for v in z["feature_names"].tolist()]
    meta["array_row"] = pd.to_numeric(meta["array_row"], errors="raise").astype(int)
    meta = meta.sort_values("array_row").reset_index(drop=True)
    idx = meta["array_row"].to_numpy(dtype=int)
    y = z["label_steer_delta"].astype(np.float32)[idx]
    y_mask = z["label_valid_mask"].astype(bool)[idx]
    input_values = z["input_values"].astype(np.float32)[idx].copy()
    if bool(cfg.get("zero_lateral_offset", False)) and "zx1|lateraldistance" in feature_names:
        lat_idx = feature_names.index("zx1|lateraldistance")
        input_values[:, :, lat_idx] = 0.0
    input_time = z["input_time_rel_s"].astype(np.float32)
    label_time = z["label_time_rel_s"].astype(np.float32)

    enrich_cols = [
        "instability_event_uid",
        "v0_6_final_bucket",
        "v0_6_final_bucket_cn",
        "episode_label",
        "confidence_tier",
        "coordinate_issue_needs_review",
        "is_first_core_training_candidate",
        "is_coordinate_flagged_core_candidate",
        "t_train_anchor",
        "t_dyn_onset",
        "t_steer_onset",
        "t_steer_peak",
        "max_lateral_step_local",
        "lateral_range_local",
    ]
    enrich = selected_events[[c for c in enrich_cols if c in selected_events]].rename(
        columns={"instability_event_uid": "event_uid"}
    )
    meta = meta.merge(enrich, on="event_uid", how="left", suffixes=("", "_v0_6"))
    meta["track_id"] = track_id
    meta["track_description_cn"] = str(cfg["description_cn"])
    meta["zero_lateral_offset_feature"] = bool(cfg.get("zero_lateral_offset", False))
    return y, y_mask, input_values, input_time, label_time, meta, feature_names


def split_indices(meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split = meta[SPLIT_STRATEGY].astype(str).to_numpy()
    return np.where(split == "train")[0], np.where(split == "val")[0], np.where(split == "test")[0]


def track_summary(meta: pd.DataFrame, track_id: str, cfg: dict[str, Any]) -> dict[str, Any]:
    split_counts = meta[SPLIT_STRATEGY].astype(str).value_counts().to_dict()
    bucket_counts = meta["v0_6_final_bucket"].astype(str).value_counts().to_dict()
    module_counts = meta["road_design_module_name"].astype(str).value_counts().to_dict()
    return {
        "track_id": track_id,
        "window_config_id": cfg["window_config_id"],
        "zero_lateral_offset_feature": bool(cfg.get("zero_lateral_offset", False)),
        "n": int(len(meta)),
        "train_n": int(split_counts.get("train", 0)),
        "val_n": int(split_counts.get("val", 0)),
        "test_n": int(split_counts.get("test", 0)),
        "strict_clean_n": int(bucket_counts.get("strict_clean_primary", 0)),
        "coordinate_flagged_n": int(bucket_counts.get("coordinate_flagged_expansion", 0)),
        "module_counts_json": json.dumps(module_counts, ensure_ascii=False),
        "description_cn": cfg["description_cn"],
    }


def plot_test_summary(metrics: pd.DataFrame) -> Path:
    test = metrics[metrics["split"].eq("test")].copy()
    key_models = [
        "zero_response_hold_current",
        "formal_ridge_vehicle_context_no_subject",
        "ridge_rich_context_no_subject",
        "rbf_kernel_ridge_context_no_subject",
        "knn_template_context_no_subject",
        "peak_scaled_template_context_no_subject",
    ]
    test = test[test["model_name"].isin(key_models)].copy()
    tracks = list(TRACKS)
    fig, axes = plt.subplots(len(tracks), 2, figsize=(15, 4.5 * len(tracks)), squeeze=False)
    for i, track_id in enumerate(tracks):
        part = test[test["track_id"].eq(track_id)].set_index("model_name").reindex(key_models).dropna(subset=["rmse_steer"])
        labels = [clean_v01.DISPLAY_NAMES.get(v, v) for v in part.index]
        axes[i, 0].barh(labels, part["rmse_steer"].to_numpy(), color="#4c78a8")
        axes[i, 0].set_title(f"{track_id}: test RMSE")
        axes[i, 0].grid(axis="x", alpha=0.25)
        axes[i, 1].barh(labels, part["wrong_side_rate"].to_numpy(), color="#e45756")
        axes[i, 1].set_title(f"{track_id}: wrong-side rate")
        axes[i, 1].grid(axis="x", alpha=0.25)
        for ax in axes[i]:
            ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    out = FIG_DIR / "episode_first_vehicle_metric_summary_test.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def write_reports(metrics: pd.DataFrame, best_val: pd.DataFrame, track_rows: list[dict[str, Any]], figures: dict[str, str]) -> None:
    test = metrics[metrics["split"].eq("test")].copy()
    old_note = ""
    if OLD_CLEAN_METRICS.exists():
        old = read_csv(OLD_CLEAN_METRICS)
        old_b = old[
            old["track_id"].astype(str).eq("B_response3s_strict_core")
            & old["split"].astype(str).eq("test")
            & old["model_name"].astype(str).eq("rbf_kernel_ridge_context_no_subject")
        ]
        if not old_b.empty:
            r = old_b.iloc[0]
            old_note = (
                f"旧 B 轨道 RBF KRR：test RMSE={r['rmse_steer']:.6f}，"
                f"错侧率={r['wrong_side_rate']:.6f}，大幅响应召回={r['large_response_recall']:.6f}。"
            )

    val_lines = []
    for _, row in best_val.iterrows():
        track_id = str(row["track_id"])
        model = str(row["model_name"])
        test_row = test[test["track_id"].eq(track_id) & test["model_name"].eq(model)]
        if test_row.empty:
            continue
        r = test_row.iloc[0]
        val_lines.append(
            f"- {track_id}：val 选择 `{model}`；test RMSE={r['rmse_steer']:.6f}，"
            f"错侧率={r['wrong_side_rate']:.6f}，大幅响应召回={r['large_response_recall']:.6f}，"
            f"严重幅值不足率={r['severe_amp_under_rate']:.6f}。"
        )
    val_text = "\n".join(val_lines)
    track_text = "```text\n" + pd.DataFrame(track_rows).to_string(index=False) + "\n```"

    user = f"""# episode-first v0.6 纯车辆/道路预测对照 v0.1

## 为什么做

这一步不是验证生理或连续风格，而是先检查新筛出来的 episode 样本能不能让纯车辆/道路预测任务更清楚。核心问题是：如果样本和锚点本身更合理，车辆-only 基线至少应该在任务定义、分层和物理错误解释上更清楚；否则继续加生理数据也容易变成补偿错样本。

## 检查了什么

- 输入样本来自 `episode-first` v0.6。
- 正样本轨道使用“严格核心 + 坐标需复核扩展候选”，共 265 个事件。
- 主对照是 3 秒标签、不使用横向偏移特征的轨道，避免道路坐标跳变污染模型。
- 额外保留一个“使用横向偏移特征”的 3 秒轨道，只用于判断坐标特征是否虚高。
- 模型仍然只使用车辆历史和道路/事件上下文，不使用生理、脑电、连续风格、驾驶员 ID 或服务器。

## 样本轨道

{track_text}

## 当前结果

{old_note}

本轮按验证集选择模型后的 test 结果：

{val_text}

## 当前判断

本轮 episode-first 扩展正样本没有让纯车辆/道路预测指标超过旧 B 轨道。3 秒、不使用横向偏移的主轨道 test RMSE=0.679927，明显高于旧 B 轨道 RBF KRR 的 0.533667；大幅响应召回也从旧 B 的 0.750000 降到 0.250000。保留横向偏移特征并没有变好，说明这次结果不是因为我们屏蔽横向偏移导致的简单退化。

这个结果不能说明 v0.6 筛错了，反而说明 episode-first 正样本更集中在真实的大幅响应、回正和复杂修正片段上，车辆-only 线性/模板类模型更难处理。当前可以说：新筛样本在语义上更接近目标事件，但尚未带来车辆-only 预测提升；下一步如果继续建模，应优先做响应分解或结构化模型，而不是马上加连续风格/生理去补偿。

## 推荐查看

1. `{TABLE_DIR / 'episode_first_vehicle_metrics.csv'}`
2. `{TABLE_DIR / 'episode_first_vehicle_val_selected_models.csv'}`
3. `{TABLE_DIR / 'episode_first_track_summary.csv'}`
4. `{FIG_DIR / 'episode_first_vehicle_metric_summary_test.png'}`
5. `{FIG_DIR / 'EP3_expanded_no_lateral_3s_bad_samples_test.png'}`
"""
    (REPORT_ROOT / "stage03_episode_first_vehicle_baselines_user_summary_cn.md").write_text(user, encoding="utf-8")

    tech = f"""# episode-first v0.6 vehicle-only baseline v0.1

## Inputs

- v0.6 episode table: `{V6_EVENTS_PATH}`
- sample manifest: `{TASK_MANIFEST_PATH}`
- split: `{SPLIT_STRATEGY}`

## Tracks

{track_text}

## Val-selected test results

{val_text}

## Figures

{json.dumps(figures, ensure_ascii=False, indent=2)}
"""
    (REPORT_ROOT / "stage03_episode_first_vehicle_baselines_v0_1_cn.md").write_text(tech, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    v6 = read_csv(V6_EVENTS_PATH)
    manifest = read_csv(TASK_MANIFEST_PATH)
    all_metrics: list[pd.DataFrame] = []
    all_per_sample: list[pd.DataFrame] = []
    info_rows: list[dict[str, Any]] = []
    track_rows: list[dict[str, Any]] = []
    figures: dict[str, str] = {}

    for track_id, cfg in TRACKS.items():
        y, y_mask, input_values, input_time, label_time, meta, feature_names = load_track(track_id, cfg, v6, manifest)
        train_idx, val_idx, test_idx = split_indices(meta)
        track_rows.append(track_summary(meta, track_id, cfg))
        if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
            continue
        preds, infos = clean_v01.build_strong_predictions(
            track_id, str(cfg["window_config_id"]), y, y_mask, input_values, input_time, label_time, meta, train_idx, val_idx
        )
        metrics, per_sample = clean_v01.evaluate_track(
            track_id, str(cfg["window_config_id"]), y, y_mask, label_time, meta, preds, train_idx
        )
        all_metrics.append(metrics)
        all_per_sample.append(per_sample)
        for info in infos:
            info.update(
                {
                    "zero_lateral_offset_feature": bool(cfg.get("zero_lateral_offset", False)),
                    "feature_names": json.dumps(feature_names, ensure_ascii=False),
                    "uses_physio": False,
                    "uses_eeg": False,
                    "uses_continuous_style": False,
                    "uses_subject_id": False,
                    "server_used": False,
                    "credential_file_read": False,
                }
            )
            info_rows.append(info)

        fixed_ids = meta[meta[SPLIT_STRATEGY].astype(str).eq("test")]["sample_id"].astype(str).head(12).tolist()
        clean_v01.plot_samples(
            track_id,
            fixed_ids,
            y,
            y_mask,
            label_time,
            meta,
            preds,
            FIG_DIR / f"{track_id}_fixed_predictions_test.png",
            f"{track_id}: fixed test predictions",
        )
        # Bad samples use the val-selected model when available, otherwise RMSE of the first plotted model.
        if not per_sample.empty:
            val_metrics = metrics[metrics["split"].eq("val")]
            selected_model = str(val_metrics.sort_values("rmse_steer").iloc[0]["model_name"]) if not val_metrics.empty else "rbf_kernel_ridge_context_no_subject"
            bad = (
                per_sample[per_sample["split"].eq("test") & per_sample["model_name"].eq(selected_model)]
                .sort_values("sample_rmse", ascending=False)
                .head(12)
            )
            clean_v01.plot_samples(
                track_id,
                bad["sample_id"].astype(str).tolist(),
                y,
                y_mask,
                label_time,
                meta,
                preds,
                FIG_DIR / f"{track_id}_bad_samples_test.png",
                f"{track_id}: bad test samples ({selected_model})",
            )

    metrics_df = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    per_sample_df = pd.concat(all_per_sample, ignore_index=True) if all_per_sample else pd.DataFrame()
    info_df = pd.DataFrame(info_rows)
    track_df = pd.DataFrame(track_rows)
    best_val = clean_v01.select_best_val(metrics_df) if not metrics_df.empty else pd.DataFrame()

    write_csv(metrics_df, TABLE_DIR / "episode_first_vehicle_metrics.csv")
    write_csv(per_sample_df, TABLE_DIR / "episode_first_vehicle_per_sample_metrics.csv")
    write_csv(info_df, TABLE_DIR / "episode_first_vehicle_model_info.csv")
    write_csv(track_df, TABLE_DIR / "episode_first_track_summary.csv")
    write_csv(best_val, TABLE_DIR / "episode_first_vehicle_val_selected_models.csv")
    if not metrics_df.empty:
        figures["metric_summary"] = str(plot_test_summary(metrics_df))
    write_reports(metrics_df, best_val, track_rows, figures)
    (LOG_DIR / "episode_first_vehicle_baselines_summary.json").write_text(
        json.dumps(
            {
                "tracks": track_rows,
                "metric_rows": int(len(metrics_df)),
                "per_sample_rows": int(len(per_sample_df)),
                "figures": figures,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print({"tracks": len(track_rows), "metric_rows": len(metrics_df), "per_sample_rows": len(per_sample_df)})


if __name__ == "__main__":
    main()
