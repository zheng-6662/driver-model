# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(r"F:/data_set_process/data_process/05_rebuild_from_raw_20260511")
SCRIPT_DIR = ROOT / "03_baselines" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_stage3_vehicle_baselines as base_eval  # noqa: E402
import stage03_vehicle_diagnostics_v0_3 as diag  # noqa: E402


OUT_DIR = ROOT / "03_baselines" / "stage03_rbf_krr_model_card_v0_4"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
LOG_DIR = OUT_DIR / "logs"
REPORT_DIR = ROOT / "09_reports"

MODEL_NAME = "rbf_krr_vehicle_no_subject"
SPLIT_STRATEGY = "session_level_split"
WINDOWS = ["pre2_label2_old_main", "pre3_label3_response_coverage"]


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIG_DIR, LOG_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def fonts() -> tuple[Any, Any, Any]:
    try:
        return (
            ImageFont.truetype("arial.ttf", 26),
            ImageFont.truetype("arial.ttf", 20),
            ImageFont.truetype("arial.ttf", 15),
        )
    except OSError:
        font = ImageFont.load_default()
        return font, font, font


def panel_line_points(
    time_axis: np.ndarray,
    values: np.ndarray,
    mask: np.ndarray,
    box: tuple[int, int, int, int],
    y_lim: float,
) -> list[tuple[int, int]]:
    x0, y0, x1, y1 = box
    valid = mask & np.isfinite(values)
    if not np.any(valid):
        return []
    t = time_axis[valid]
    v = values[valid]
    t_min = float(np.nanmin(time_axis))
    t_max = float(np.nanmax(time_axis))
    if abs(t_max - t_min) < 1e-9:
        t_max = t_min + 1.0
    xs = x0 + (t - t_min) / (t_max - t_min) * (x1 - x0)
    ys = y1 - (v + y_lim) / (2.0 * y_lim) * (y1 - y0)
    return [(int(x), int(y)) for x, y in zip(xs, ys)]


def draw_prediction_grid(
    out_path: Path,
    title: str,
    sample_indices: list[int],
    y: np.ndarray,
    y_mask: np.ndarray,
    pred: np.ndarray,
    label_time: np.ndarray,
    meta: pd.DataFrame,
    metric_rows: pd.DataFrame,
) -> None:
    title_font, font, small = fonts()
    width, height = 1800, 1240
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((35, 25), title, fill=(0, 0, 0), font=title_font)
    draw.text((35, 62), "black = ground truth; blue = RBF KRR no-subject prediction; red line = zero", fill=(80, 80, 80), font=font)
    cols, rows = 3, 4
    panel_w, panel_h = 560, 255
    left, top = 35, 115
    gap_x, gap_y = 35, 35
    metrics_by_sample = metric_rows.set_index("sample_id")
    for n, idx in enumerate(sample_indices[: cols * rows]):
        r, c = divmod(n, cols)
        x0 = left + c * (panel_w + gap_x)
        y0 = top + r * (panel_h + gap_y)
        box = (x0 + 52, y0 + 42, x0 + panel_w - 18, y0 + panel_h - 36)
        draw.rectangle((x0, y0, x0 + panel_w, y0 + panel_h), outline=(190, 190, 190), width=1)
        gt = y[idx]
        pr = pred[idx]
        mask = y_mask[idx]
        y_lim = float(np.nanmax(np.abs(np.r_[gt[mask], pr[np.isfinite(pr)]]))) if np.any(mask) else 1.0
        y_lim = max(y_lim * 1.15, 0.2)
        zero_y = int(box[3] - (0.0 + y_lim) / (2.0 * y_lim) * (box[3] - box[1]))
        draw.line((box[0], zero_y, box[2], zero_y), fill=(220, 60, 60), width=1)
        draw.rectangle(box, outline=(150, 150, 150), width=1)
        gt_pts = panel_line_points(label_time, gt, mask, box, y_lim)
        pred_pts = panel_line_points(label_time, pr, np.isfinite(pr), box, y_lim)
        if len(gt_pts) > 1:
            draw.line(gt_pts, fill=(0, 0, 0), width=3)
        if len(pred_pts) > 1:
            draw.line(pred_pts, fill=(40, 110, 210), width=3)
        row = meta.iloc[idx]
        metric = metrics_by_sample.loc[row["sample_id"]] if row["sample_id"] in metrics_by_sample.index else None
        if metric is not None and isinstance(metric, pd.DataFrame):
            metric = metric.iloc[0]
        headline = f"{n+1}. {row['subject']}  peak={float(np.nanmax(np.abs(gt[mask]))):.2f}"
        draw.text((x0 + 12, y0 + 10), headline, fill=(0, 0, 0), font=font)
        if metric is not None:
            text = (
                f"rmse={float(metric['sample_rmse']):.3f} "
                f"ratio={float(metric['peak_amp_ratio_pred_over_gt']):.2f} "
                f"wrong={int(metric['wrong_side'])} under={int(metric['severe_amp_under'])}"
            )
            draw.text((x0 + 12, y0 + panel_h - 28), text, fill=(80, 0, 0), font=small)
    img.save(out_path)


def get_prediction_for_window(window_id: str) -> dict[str, Any]:
    y, y_mask, input_values, label_time, meta = diag.load_window(window_id)
    per_sample, info, preds = diag.evaluate_model_set(window_id, SPLIT_STRATEGY, y, y_mask, input_values, label_time, meta)
    pred = preds[MODEL_NAME]
    target = per_sample[
        (per_sample["model_name"] == MODEL_NAME)
        & (per_sample["split_strategy"] == SPLIT_STRATEGY)
        & (per_sample["split"] == "test")
    ].copy()
    return {
        "window_id": window_id,
        "y": y,
        "y_mask": y_mask,
        "pred": pred,
        "label_time": label_time,
        "meta": meta,
        "per_sample": per_sample,
        "test_metrics": target,
        "info": info[info["model_name"] == MODEL_NAME].copy(),
    }


def select_fixed_samples(test_metrics: pd.DataFrame, meta: pd.DataFrame) -> list[int]:
    test_meta = meta[meta[SPLIT_STRATEGY] == "test"].copy()
    merged = test_meta[["array_row", "sample_id"]].merge(test_metrics[["sample_id", "gt_peak_abs"]], on="sample_id", how="left")
    merged = merged.dropna(subset=["gt_peak_abs"])
    top = merged.sort_values("gt_peak_abs", ascending=False).head(6)
    median = float(merged["gt_peak_abs"].median()) if not merged.empty else 0.0
    mid = merged.assign(dist=(merged["gt_peak_abs"] - median).abs()).sort_values("dist").head(8)
    rows = pd.concat([top, mid], ignore_index=True).drop_duplicates("array_row").head(12)
    return [int(x) for x in rows["array_row"].tolist()]


def select_bad_samples(test_metrics: pd.DataFrame, meta: pd.DataFrame) -> list[int]:
    test_meta = meta[["array_row", "sample_id"]]
    rows = test_metrics.sort_values("sample_rmse", ascending=False).head(12)
    rows = rows.merge(test_meta, on="sample_id", how="left")
    return [int(x) for x in rows["array_row"].dropna().tolist()]


def group_summary(rows: pd.DataFrame, meta: pd.DataFrame, window_id: str) -> pd.DataFrame:
    enriched = rows.merge(
        meta[["sample_id", "event_type", "event_level", "curvature_anchor"]],
        on="sample_id",
        how="left",
    )
    enriched["response_direction"] = np.where(enriched["gt_peak_abs"] > 0, np.where(enriched["peak_direction_match"] == 1, "matched_or_unknown", "mismatched"), "unknown")
    enriched["gt_peak_abs_bin"] = pd.cut(
        enriched["gt_peak_abs"],
        bins=[0.0, 0.25, 0.5, 1.0, 2.0, np.inf],
        labels=["0-0.25", "0.25-0.5", "0.5-1.0", "1.0-2.0", ">=2.0"],
        include_lowest=True,
    )
    enriched["reversal_bin"] = pd.cut(
        enriched["gt_reversal_count"],
        bins=[-1, 0, 1, 3, np.inf],
        labels=["0", "1", "2-3", ">=4"],
    )
    summary_frames: list[pd.DataFrame] = []
    for group_name, col in [
        ("event_type", "event_type"),
        ("event_level", "event_level"),
        ("gt_peak_abs_bin", "gt_peak_abs_bin"),
        ("gt_reversal_count_bin", "reversal_bin"),
        ("is_large_response", "is_large_response"),
        ("is_difficult_peak_top20", "is_difficult_peak_top20"),
    ]:
        grouped = (
            enriched.groupby(col, dropna=False, observed=False)
            .agg(
                n_samples=("sample_id", "count"),
                sample_rmse_mean=("sample_rmse", "mean"),
                sample_rmse_median=("sample_rmse", "median"),
                wrong_side_rate=("wrong_side", "mean"),
                severe_under_rate=("severe_amp_under", "mean"),
                large_response_recall=("large_response_recalled", "mean"),
                peak_amp_ratio_mean=("peak_amp_ratio_pred_over_gt", "mean"),
                tail_abs_error_mean=("tail_abs_error", "mean"),
                reversal_exact_rate=("reversal_count_exact", "mean"),
            )
            .reset_index()
            .rename(columns={col: "group_value"})
        )
        grouped.insert(0, "group_name", group_name)
        grouped.insert(0, "window_config_id", window_id)
        summary_frames.append(grouped)
    return pd.concat(summary_frames, ignore_index=True)


def subject_summary(rows: pd.DataFrame, window_id: str) -> pd.DataFrame:
    out = (
        rows.groupby("subject", dropna=False)
        .agg(
            n_samples=("sample_id", "count"),
            sample_rmse_mean=("sample_rmse", "mean"),
            sample_rmse_median=("sample_rmse", "median"),
            gt_peak_abs_mean=("gt_peak_abs", "mean"),
            wrong_side_rate=("wrong_side", "mean"),
            severe_under_rate=("severe_amp_under", "mean"),
            difficult_rate=("is_difficult_peak_top20", "mean"),
            tail_abs_error_mean=("tail_abs_error", "mean"),
            reversal_exact_rate=("reversal_count_exact", "mean"),
        )
        .reset_index()
        .sort_values(["sample_rmse_mean", "n_samples"], ascending=[False, False])
    )
    out.insert(0, "window_config_id", window_id)
    return out


def metric_summary(rows: pd.DataFrame, window_id: str) -> pd.DataFrame:
    return base_eval.aggregate_metrics(rows).assign(candidate_role="stage3_current_clean_vehicle_candidate")


def write_report(
    metrics: pd.DataFrame,
    subj_pre2: pd.DataFrame,
    groups_pre2: pd.DataFrame,
    groups_pre3: pd.DataFrame,
) -> None:
    pre2 = metrics[(metrics["window_config_id"] == "pre2_label2_old_main") & (metrics["split"] == "test")]
    pre3 = metrics[(metrics["window_config_id"] == "pre3_label3_response_coverage") & (metrics["split"] == "test")]
    large_pre2 = groups_pre2[(groups_pre2["group_name"] == "gt_peak_abs_bin")]
    large_pre3 = groups_pre3[(groups_pre3["group_name"] == "gt_peak_abs_bin")]
    report = f"""# 阶段 3 v0.4 候选强车辆基线模型卡：RBF KRR 无被试 ID

更新时间：2026-05-12

## 模型定位

`rbf_krr_vehicle_no_subject` 是当前阶段 3 最干净的强车辆候选：只使用车辆历史统计、道路曲率/事件元信息，不使用 `subject`，不使用连续风格、生理或脑电，也不使用 old v400/raw dynamic 作为主锚点。

## 关键测试结果

pre2 + session-level test：

{pre2[['window_config_id','split_strategy','split','model_name','n_samples','rmse_steer','peak_direction_accuracy','wrong_side_rate','large_response_recall','peak_amp_ratio_pred_over_gt_mean','severe_amp_under_rate','peak_time_mae_s','tail_abs_error_mean','reversal_count_exact_match_rate','difficult_top20_rmse']].to_string(index=False)}

pre3 + session-level test：

{pre3[['window_config_id','split_strategy','split','model_name','n_samples','rmse_steer','peak_direction_accuracy','wrong_side_rate','large_response_recall','peak_amp_ratio_pred_over_gt_mean','severe_amp_under_rate','peak_time_mae_s','tail_abs_error_mean','reversal_count_exact_match_rate','difficult_top20_rmse']].to_string(index=False)}

## 分被试风险

{subj_pre2.head(12).to_string(index=False)}

## pre2 幅值桶

{large_pre2.to_string(index=False)}

## pre3 幅值桶

{large_pre3.to_string(index=False)}

## 图表

- 固定样本轨迹图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_rbf_krr_model_card_v0_4/figures/stage03_rbf_krr_fixed_predictions_pre2_session_v0_4.png`
- 坏样本轨迹图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_rbf_krr_model_card_v0_4/figures/stage03_rbf_krr_bad_samples_pre2_session_v0_4.png`
- 长窗口固定样本图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_rbf_krr_model_card_v0_4/figures/stage03_rbf_krr_fixed_predictions_pre3_session_v0_4.png`
- 长窗口坏样本图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_rbf_krr_model_card_v0_4/figures/stage03_rbf_krr_bad_samples_pre3_session_v0_4.png`

## 当前判断

这个模型可以暂时作为阶段 3 的强车辆参照，但仍不能开启风格/生理有效性结论。原因是：当前样本只覆盖低泄漏道路曲率事件；pre3 长窗口仍需要确认尾段和大幅响应；反向修正 exact rate 很低，说明结构化响应问题还没有解决。下一步应继续在阶段 3 内完成长窗口和物理错误复核，或者构建更明确的响应关键点/分解车辆模型。
"""
    (REPORT_DIR / "stage03_rbf_krr_candidate_model_card_v0_4_cn.md").write_text(report, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    all_metrics: list[pd.DataFrame] = []
    all_subject: list[pd.DataFrame] = []
    all_groups: list[pd.DataFrame] = []
    fixed_rows: list[dict[str, Any]] = []
    for window_id in WINDOWS:
        pack = get_prediction_for_window(window_id)
        metric = metric_summary(pack["test_metrics"], window_id)
        all_metrics.append(metric)
        subj = subject_summary(pack["test_metrics"], window_id)
        groups = group_summary(pack["test_metrics"], pack["meta"], window_id)
        all_subject.append(subj)
        all_groups.append(groups)
        fixed = select_fixed_samples(pack["test_metrics"], pack["meta"])
        bad = select_bad_samples(pack["test_metrics"], pack["meta"])
        for rank, idx in enumerate(fixed, start=1):
            row = pack["meta"].iloc[idx]
            fixed_rows.append({"window_config_id": window_id, "plot_type": "fixed", "rank": rank, "array_row": idx, "sample_id": row["sample_id"], "subject": row["subject"]})
        for rank, idx in enumerate(bad, start=1):
            row = pack["meta"].iloc[idx]
            fixed_rows.append({"window_config_id": window_id, "plot_type": "bad", "rank": rank, "array_row": idx, "sample_id": row["sample_id"], "subject": row["subject"]})
        draw_prediction_grid(
            FIG_DIR / f"stage03_rbf_krr_fixed_predictions_{window_id.replace('pre2_label2_old_main','pre2_session').replace('pre3_label3_response_coverage','pre3_session')}_v0_4.png",
            f"Stage 3 v0.4 {window_id} fixed test predictions",
            fixed,
            pack["y"],
            pack["y_mask"],
            pack["pred"],
            pack["label_time"],
            pack["meta"],
            pack["test_metrics"],
        )
        draw_prediction_grid(
            FIG_DIR / f"stage03_rbf_krr_bad_samples_{window_id.replace('pre2_label2_old_main','pre2_session').replace('pre3_label3_response_coverage','pre3_session')}_v0_4.png",
            f"Stage 3 v0.4 {window_id} worst test predictions",
            bad,
            pack["y"],
            pack["y_mask"],
            pack["pred"],
            pack["label_time"],
            pack["meta"],
            pack["test_metrics"],
        )

    metrics = pd.concat(all_metrics, ignore_index=True)
    subjects = pd.concat(all_subject, ignore_index=True)
    groups = pd.concat(all_groups, ignore_index=True)
    fixed_df = pd.DataFrame(fixed_rows)
    metrics.to_csv(TABLE_DIR / "stage03_rbf_krr_candidate_metrics_v0_4.csv", index=False, encoding="utf-8-sig")
    subjects.to_csv(TABLE_DIR / "stage03_rbf_krr_per_subject_v0_4.csv", index=False, encoding="utf-8-sig")
    groups.to_csv(TABLE_DIR / "stage03_rbf_krr_response_group_summary_v0_4.csv", index=False, encoding="utf-8-sig")
    fixed_df.to_csv(TABLE_DIR / "stage03_rbf_krr_plot_sample_set_v0_4.csv", index=False, encoding="utf-8-sig")
    write_report(
        metrics,
        subjects[subjects["window_config_id"] == "pre2_label2_old_main"],
        groups[groups["window_config_id"] == "pre2_label2_old_main"],
        groups[groups["window_config_id"] == "pre3_label3_response_coverage"],
    )
    summary = {
        "model_name": MODEL_NAME,
        "split_strategy": SPLIT_STRATEGY,
        "windows": WINDOWS,
        "metric_rows": int(len(metrics)),
        "subject_rows": int(len(subjects)),
        "group_rows": int(len(groups)),
        "plot_samples": int(len(fixed_df)),
        "server_used": False,
        "raw_files_modified": False,
    }
    (LOG_DIR / "stage03_rbf_krr_model_card_summary_v0_4.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
