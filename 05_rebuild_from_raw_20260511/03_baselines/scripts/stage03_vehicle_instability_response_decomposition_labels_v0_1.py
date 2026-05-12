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
REPORT_ROOT = ROOT / "09_reports"
PROCESSED_DIR = ROOT / "03_processed_datasets" / "vehicle_instability_allraw_highconf_v0_1"
ARRAY_DIR = PROCESSED_DIR / "arrays"
TASK_MANIFEST_PATH = (
    ROOT
    / "02_samples"
    / "vehicle_instability_response_task_decision_v0_1"
    / "tables"
    / "sample_response_task_manifest.csv"
)
OUT_ROOT = ROOT / "03_baselines" / "stage03_vehicle_instability_response_decomposition_labels_v0_1"
TABLE_DIR = OUT_ROOT / "tables"
FIG_DIR = OUT_ROOT / "figures"
LOG_DIR = OUT_ROOT / "logs"

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_stage3_vehicle_baselines as eval_utils  # noqa: E402


SPLIT_STRATEGY = "session_level_split"
TRACKS = {
    "A_instant2s_core": {
        "window_config_id": "pre2_label2_old_main",
        "task_sample_role": "instant2s_core_candidate",
        "label_horizon_s": 2.0,
        "description_cn": "2秒即时响应核心候选",
    },
    "B_response3s_strict_core": {
        "window_config_id": "pre3_label3_response_coverage",
        "task_sample_role": "response3s_strict_core_candidate",
        "label_horizon_s": 3.0,
        "description_cn": "3秒响应覆盖严格核心候选",
    },
}


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIG_DIR, LOG_DIR, REPORT_ROOT]:
        path.mkdir(parents=True, exist_ok=True)


def first_crossing(arr: np.ndarray, threshold: float) -> int:
    idx = np.where(np.abs(arr) >= threshold)[0]
    return int(idx[0]) if idx.size else -1


def direction_name(value: float) -> str:
    if not np.isfinite(value) or abs(value) < 1e-9:
        return "zero"
    return "positive" if value > 0 else "negative"


def quantile_threshold(values: pd.Series, q: float, fallback: float) -> float:
    vals = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return fallback
    out = float(vals.quantile(q))
    return out if np.isfinite(out) else fallback


def load_track_rows(manifest: pd.DataFrame, track_id: str, cfg: dict[str, Any]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    rows = manifest[
        (manifest["window_config_id"].astype(str) == cfg["window_config_id"])
        & (manifest["task_sample_role"].astype(str) == cfg["task_sample_role"])
    ].copy()
    if rows.empty:
        raise RuntimeError(f"No rows for {track_id}")
    rows["array_row"] = pd.to_numeric(rows["array_row"], errors="raise").astype(int)
    rows = rows.sort_values("array_row").reset_index(drop=True)

    z = np.load(ARRAY_DIR / f"{cfg['window_config_id']}.npz", allow_pickle=True)
    idx = rows["array_row"].to_numpy(dtype=int)
    y = z["label_steer_delta"].astype(np.float64)[idx]
    y_mask = z["label_valid_mask"].astype(bool)[idx]
    label_time = z["label_time_rel_s"].astype(np.float64)
    return rows, y, y_mask, label_time


def compute_raw_labels(rows: pd.DataFrame, y: np.ndarray, y_mask: np.ndarray, label_time: np.ndarray, track_id: str) -> pd.DataFrame:
    out_rows: list[dict[str, Any]] = []
    horizon = float(label_time[-1] - label_time[0]) if len(label_time) else np.nan
    for i, row in rows.iterrows():
        valid = y_mask[i] & np.isfinite(y[i])
        gt = np.where(valid, y[i], np.nan)
        peak = eval_utils.peak_stats(gt, label_time)
        peak_abs = float(peak["peak_abs"])
        peak_signed = float(peak["peak_signed"])
        peak_time = float(peak["peak_time_s"])
        direction = direction_name(peak_signed)
        onset_threshold = max(0.015, 0.2 * max(peak_abs, 1e-6))
        onset_idx = first_crossing(np.nan_to_num(gt, nan=0.0), onset_threshold)
        onset_time = float(label_time[onset_idx]) if onset_idx >= 0 else np.nan
        valid_vals = gt[np.isfinite(gt)]
        tail_signed = float(valid_vals[-1]) if valid_vals.size else np.nan
        tail_abs = abs(tail_signed) if np.isfinite(tail_signed) else np.nan
        tail_ratio = tail_abs / max(peak_abs, 1e-6) if np.isfinite(tail_abs) else np.nan
        reversal_count = int(eval_utils.reversal_count(gt))
        zero_crossing_has = int(eval_utils.zero_crossing_has(gt))
        valid_ratio = float(valid.mean()) if len(valid) else 0.0

        if not np.isfinite(peak_abs):
            computed_morphology = "invalid"
        elif reversal_count >= 2:
            computed_morphology = "multi_correction"
        elif reversal_count == 1 or zero_crossing_has:
            computed_morphology = "reverse_correction"
        else:
            computed_morphology = "single_lobe"

        peak_time_frac = peak_time / max(horizon, 1e-6) if np.isfinite(peak_time) else np.nan
        if not np.isfinite(peak_time_frac):
            peak_time_bucket = "unknown"
        elif peak_time_frac <= 0.33:
            peak_time_bucket = "early_peak"
        elif peak_time_frac <= 0.66:
            peak_time_bucket = "middle_peak"
        else:
            peak_time_bucket = "late_peak"

        if not np.isfinite(onset_time):
            onset_bucket = "no_onset"
        elif onset_time <= 0.30:
            onset_bucket = "fast_onset"
        elif onset_time <= 0.80:
            onset_bucket = "middle_onset"
        else:
            onset_bucket = "delayed_onset"

        if not np.isfinite(tail_ratio):
            tail_state = "unknown_tail"
        elif tail_ratio <= 0.20:
            tail_state = "returned_tail"
        elif tail_ratio <= 0.50:
            tail_state = "residual_tail"
        else:
            tail_state = "unsettled_tail"

        out_rows.append(
            {
                "track_id": track_id,
                "sample_id": row["sample_id"],
                "event_uid": row["event_uid"],
                "subject": row["subject"],
                "session_stamp": row["session_stamp"],
                "window_config_id": row["window_config_id"],
                "split_strategy": SPLIT_STRATEGY,
                "split": row[SPLIT_STRATEGY],
                "task_sample_role": row["task_sample_role"],
                "road_design_module_name": row.get("road_design_module_name", ""),
                "road_design_instance_name": row.get("road_design_instance_name", ""),
                "event_level": row.get("event_level", ""),
                "anchor_time_rel_s": row.get("anchor_time_rel_s", np.nan),
                "label_valid_ratio": valid_ratio,
                "peak_abs": peak_abs,
                "peak_signed": peak_signed,
                "peak_direction": direction,
                "peak_idx": int(peak["peak_idx"]),
                "peak_time_s": peak_time,
                "peak_time_frac": peak_time_frac,
                "peak_time_bucket": peak_time_bucket,
                "onset_time_s": onset_time,
                "onset_bucket": onset_bucket,
                "tail_signed": tail_signed,
                "tail_abs": tail_abs,
                "tail_abs_over_peak": tail_ratio,
                "tail_state": tail_state,
                "zero_crossing_has": zero_crossing_has,
                "reversal_count": reversal_count,
                "computed_morphology": computed_morphology,
                "manifest_morphology": row.get("eval_label_morphology", ""),
                "manifest_peak_direction": row.get("eval_label_peak_direction", ""),
                "manifest_peak_abs": row.get("eval_label_peak_abs", np.nan),
                "manifest_peak_time_rel_s": row.get("eval_label_peak_time_rel_s", np.nan),
                "manifest_onset_time_rel_s": row.get("eval_label_onset_time_rel_s", np.nan),
                "manifest_reversal_count": row.get("eval_label_reversal_count", np.nan),
                "manifest_tail_abs": row.get("eval_label_tail_abs", np.nan),
                "label_source_note": "label-derived target/evaluation only; never use as model input or split criterion",
            }
        )
    return pd.DataFrame(out_rows)


def add_train_split_threshold_labels(labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = labels.copy()
    threshold_rows: list[dict[str, Any]] = []
    for track_id, group in out.groupby("track_id"):
        train = group[group["split"] == "train"].copy()
        if train.empty:
            train = group
        large_thr = quantile_threshold(train["peak_abs"], 0.75, 1.0)
        difficult_thr = quantile_threshold(train["peak_abs"], 0.80, large_thr)
        small_thr = max(0.15, quantile_threshold(train["peak_abs"], 0.25, 0.25))
        amp_bins = [-np.inf, small_thr, large_thr, np.inf]
        amp_names = ["small_response", "medium_response", "large_response"]

        idx = out["track_id"] == track_id
        out.loc[idx, "large_response_threshold_train_p75"] = large_thr
        out.loc[idx, "difficult_response_threshold_train_p80"] = difficult_thr
        out.loc[idx, "small_response_threshold_train_max_p25_015"] = small_thr
        out.loc[idx, "is_large_response_target"] = out.loc[idx, "peak_abs"] >= large_thr
        out.loc[idx, "is_difficult_peak_target"] = out.loc[idx, "peak_abs"] >= difficult_thr
        out.loc[idx, "amplitude_bucket"] = pd.cut(
            out.loc[idx, "peak_abs"],
            bins=amp_bins,
            labels=amp_names,
            include_lowest=True,
        ).astype(str)
        out.loc[idx, "response_family_target"] = (
            out.loc[idx, "peak_direction"].astype(str)
            + "__"
            + out.loc[idx, "amplitude_bucket"].astype(str)
            + "__"
            + out.loc[idx, "computed_morphology"].astype(str)
        )
        out.loc[idx, "needs_structure_head"] = (
            out.loc[idx, "computed_morphology"].isin(["reverse_correction", "multi_correction"])
            | out.loc[idx, "tail_state"].isin(["unsettled_tail"])
            | out.loc[idx, "peak_time_bucket"].isin(["late_peak"])
        )
        threshold_rows.append(
            {
                "track_id": track_id,
                "n_samples": int(len(group)),
                "train_n": int((group["split"] == "train").sum()),
                "val_n": int((group["split"] == "val").sum()),
                "test_n": int((group["split"] == "test").sum()),
                "large_response_threshold_train_p75": large_thr,
                "difficult_response_threshold_train_p80": difficult_thr,
                "small_response_threshold_train_max_p25_015": small_thr,
                "threshold_scope": "fit on session-level train split within each track",
            }
        )
    return out, pd.DataFrame(threshold_rows)


def build_summaries(labels: pd.DataFrame) -> dict[str, pd.DataFrame]:
    summaries: dict[str, pd.DataFrame] = {}
    rows = []
    for track_id, group in labels.groupby("track_id"):
        rows.append(
            {
                "track_id": track_id,
                "n_samples": int(len(group)),
                "subject_n": int(group["subject"].nunique()),
                "train_n": int((group["split"] == "train").sum()),
                "val_n": int((group["split"] == "val").sum()),
                "test_n": int((group["split"] == "test").sum()),
                "mean_peak_abs": float(group["peak_abs"].mean()),
                "median_peak_abs": float(group["peak_abs"].median()),
                "large_response_rate": float(group["is_large_response_target"].mean()),
                "difficult_response_rate": float(group["is_difficult_peak_target"].mean()),
                "needs_structure_head_rate": float(group["needs_structure_head"].mean()),
                "late_peak_rate": float((group["peak_time_bucket"] == "late_peak").mean()),
                "unsettled_tail_rate": float((group["tail_state"] == "unsettled_tail").mean()),
                "multi_correction_rate": float((group["computed_morphology"] == "multi_correction").mean()),
                "reverse_or_multi_rate": float(group["computed_morphology"].isin(["reverse_correction", "multi_correction"]).mean()),
                "positive_direction_rate": float((group["peak_direction"] == "positive").mean()),
                "negative_direction_rate": float((group["peak_direction"] == "negative").mean()),
            }
        )
    summaries["track_summary"] = pd.DataFrame(rows)

    for name, cols in {
        "split_summary": ["track_id", "split"],
        "morphology_summary": ["track_id", "computed_morphology"],
        "manifest_morphology_summary": ["track_id", "manifest_morphology"],
        "response_family_summary": ["track_id", "response_family_target"],
        "road_module_summary": ["track_id", "road_design_module_name"],
        "subject_summary": ["track_id", "subject"],
    }.items():
        parts = []
        for key, group in labels.groupby(cols, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            row = dict(zip(cols, key))
            row.update(
                {
                    "n_samples": int(len(group)),
                    "mean_peak_abs": float(group["peak_abs"].mean()),
                    "large_response_rate": float(group["is_large_response_target"].mean()),
                    "late_peak_rate": float((group["peak_time_bucket"] == "late_peak").mean()),
                    "unsettled_tail_rate": float((group["tail_state"] == "unsettled_tail").mean()),
                    "needs_structure_head_rate": float(group["needs_structure_head"].mean()),
                }
            )
            parts.append(row)
        summaries[name] = pd.DataFrame(parts).sort_values(cols + ["n_samples"], ascending=[True] * len(cols) + [False])
    return summaries


def plot_count_bars(labels: pd.DataFrame) -> None:
    counts = labels.groupby(["track_id", "computed_morphology"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    counts.plot(kind="bar", stacked=True, ax=ax, color=["#4c78a8", "#f58518", "#54a24b", "#e45756"])
    ax.set_title("Response morphology counts by clean track")
    ax.set_xlabel("track")
    ax.set_ylabel("samples")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(FIG_DIR / "response_decomposition_morphology_counts.png", dpi=180)
    plt.close(fig)


def plot_peak_scatter(labels: pd.DataFrame) -> None:
    color_map = {
        "single_lobe": "#4c78a8",
        "reverse_correction": "#f58518",
        "multi_correction": "#e45756",
        "invalid": "#999999",
    }
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for morph, group in labels.groupby("computed_morphology"):
        ax.scatter(
            group["peak_time_s"],
            group["peak_abs"],
            s=32,
            alpha=0.70,
            label=morph,
            color=color_map.get(morph, "#72b7b2"),
            edgecolor="white",
            linewidth=0.4,
        )
    ax.set_xlabel("peak time (s)")
    ax.set_ylabel("peak abs")
    ax.set_title("Peak timing vs amplitude")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    fig.savefig(FIG_DIR / "response_decomposition_peak_time_amp_scatter.png", dpi=180)
    plt.close(fig)


def plot_b_mean_trajectories(labels: pd.DataFrame) -> None:
    b_labels = labels[labels["track_id"] == "B_response3s_strict_core"].copy()
    if b_labels.empty:
        return
    rows, y, y_mask, label_time = load_track_rows(
        pd.read_csv(TASK_MANIFEST_PATH),
        "B_response3s_strict_core",
        TRACKS["B_response3s_strict_core"],
    )
    row_by_sample = {sid: i for i, sid in enumerate(rows["sample_id"].astype(str))}
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for morph, group in b_labels.groupby("computed_morphology"):
        idx = [row_by_sample[sid] for sid in group["sample_id"].astype(str) if sid in row_by_sample]
        if not idx:
            continue
        arr = np.where(y_mask[idx], y[idx], np.nan)
        mean = np.nanmean(arr, axis=0)
        ax.plot(label_time, mean, linewidth=2.0, label=f"{morph} n={len(idx)}")
    ax.axhline(0, color="#666666", linewidth=0.8)
    ax.set_xlabel("label time (s)")
    ax.set_ylabel("steering delta")
    ax.set_title("B-track mean GT trajectories by computed morphology")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(FIG_DIR / "b_track_mean_gt_trajectories_by_morphology.png", dpi=180)
    plt.close(fig)


def simple_markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    show = df.head(max_rows).copy()
    cols = list(show.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in show.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_reports(labels: pd.DataFrame, thresholds: pd.DataFrame, summaries: dict[str, pd.DataFrame]) -> None:
    b = labels[labels["track_id"] == "B_response3s_strict_core"]
    b_summary = summaries["track_summary"].set_index("track_id").loc["B_response3s_strict_core"]
    a_summary = summaries["track_summary"].set_index("track_id").loc["A_instant2s_core"]
    lines = [
        "# 阶段 3：车辆-only 响应分解标签 v0.1",
        "",
        "## 为什么做",
        "",
        "B 轨道 RBF KRR 的坏样本复查显示，单纯轨迹回归仍无法处理反向修正和多段修正。下一步结构化车辆-only 模型需要先有稳定的方向、幅值、峰值时间、启动延迟、尾段状态和响应形态目标，所以本轮只从标签轨迹生成响应分解标签，不训练新模型。",
        "",
        "## 输入与无泄漏边界",
        "",
        "- 输入：`sample_response_task_manifest.csv`、`pre2_label2_old_main.npz`、`pre3_label3_response_coverage.npz`。",
        "- 标签来自事件后方向盘轨迹，只能作为训练目标、辅助任务目标和评估分层，不能作为模型输入、split 条件、标准化条件或风格/生理特征。",
        "- 大幅响应、困难响应和小响应阈值只在每个轨道的 session-level train split 上拟合，然后应用到 val/test。",
        "- 本轮未使用生理、脑电、连续风格、驾驶员 ID、服务器或服务器密码文件。",
        "",
        "## 轨道汇总",
        "",
        simple_markdown_table(summaries["track_summary"]),
        "",
        "## 阈值表",
        "",
        simple_markdown_table(thresholds),
        "",
        "## B 轨道重点结论",
        "",
        f"- B 轨道共有 {int(b_summary['n_samples'])} 个样本，train/val/test={int(b_summary['train_n'])}/{int(b_summary['val_n'])}/{int(b_summary['test_n'])}。",
        f"- B 轨道平均主峰幅值={float(b_summary['mean_peak_abs']):.4f}，大幅响应比例={float(b_summary['large_response_rate']):.4f}，需要结构化 head 的比例={float(b_summary['needs_structure_head_rate']):.4f}。",
        f"- B 轨道 computed multi-correction 比例={float(b_summary['multi_correction_rate']):.4f}，reverse/multi 合计比例={float(b_summary['reverse_or_multi_rate']):.4f}。",
        f"- B 轨道正向比例={float(b_summary['positive_direction_rate']):.4f}，负向比例={float(b_summary['negative_direction_rate']):.4f}。",
        "",
        "## A 轨道处理方式",
        "",
        f"- A 轨道只有 {int(a_summary['n_samples'])} 个样本，test 只有 {int(a_summary['test_n'])} 个；保留响应分解标签，但只作为即时响应诊断，不作为主线泛化结论。",
        "",
        "## 下一步",
        "",
        "用这些标签做车辆-only 响应分解模型：先预测方向、幅值桶、峰值时间桶、启动延迟桶、响应形态和尾段状态，再比较关键点+残差轨迹是否能改善 B 轨道坏样本。仍然不能进入连续风格或生理有效性结论。",
        "",
        "## 产物",
        "",
        f"- 样本标签表：`{TABLE_DIR / 'response_decomposition_sample_labels.csv'}`",
        f"- 轨道汇总：`{TABLE_DIR / 'response_decomposition_track_summary.csv'}`",
        f"- 响应形态汇总：`{TABLE_DIR / 'response_decomposition_morphology_summary.csv'}`",
        f"- 图：`{FIG_DIR / 'response_decomposition_morphology_counts.png'}`",
        f"- 图：`{FIG_DIR / 'response_decomposition_peak_time_amp_scatter.png'}`",
        f"- 图：`{FIG_DIR / 'b_track_mean_gt_trajectories_by_morphology.png'}`",
    ]
    (REPORT_ROOT / "stage03_vehicle_instability_response_decomposition_labels_v0_1_cn.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    user_lines = [
        "# 阶段 3 用户查看版：响应分解标签 v0.1",
        "",
        "## 这一步为什么做",
        "",
        "上一轮已经看出，B 轨道车辆-only RBF KRR 的主要问题不是单纯 RMSE，而是反向修正、多段修正、峰值时间、幅值和尾段状态预测不好。因此下一步模型不能只输出整条轨迹，应该先把响应拆成几个能解释的物理目标。",
        "",
        "## 这一步检查了什么",
        "",
        "- 从已有 2 秒和 3 秒干净轨道标签轨迹里，提取主峰方向、主峰幅值、峰值时间、启动时间、尾段状态、零线穿越、反向修正次数和响应形态。",
        "- 这些标签只作为训练目标或评估分组，不能作为模型输入。",
        "",
        "## 目前发现了什么",
        "",
        f"- A 轨道有 {int(a_summary['n_samples'])} 个即时响应样本，但 test 只有 {int(a_summary['test_n'])} 个，只适合做诊断。",
        f"- B 轨道有 {int(b_summary['n_samples'])} 个 3 秒响应覆盖严格核心样本，train/val/test={int(b_summary['train_n'])}/{int(b_summary['val_n'])}/{int(b_summary['test_n'])}。",
        f"- B 轨道里 reverse/multi 响应比例很高，合计 {float(b_summary['reverse_or_multi_rate']):.3f}；这解释了为什么普通轨迹回归容易在反向修正和多段修正上失败。",
        "",
        "## 哪些结果可信",
        "",
        "可信的是：B 轨道下一步应该优先做结构化车辆-only 响应分解，而不是直接进入风格或生理增量验证。",
        "",
        "## 哪些结果还不能下结论",
        "",
        "这些标签来自未来方向盘轨迹，所以不能作为推理输入，也不能证明生理、脑电或连续风格有效。",
        "",
        "## 下一步是否可以继续",
        "",
        "可以继续，但只继续到车辆-only 响应分解模型。等这个强车辆参考稳定后，才适合验证风格和生理是否提供额外信息。",
        "",
        "## 推荐查看",
        "",
        f"1. `{FIG_DIR / 'response_decomposition_morphology_counts.png'}`",
        f"2. `{FIG_DIR / 'response_decomposition_peak_time_amp_scatter.png'}`",
        f"3. `{FIG_DIR / 'b_track_mean_gt_trajectories_by_morphology.png'}`",
        f"4. `{TABLE_DIR / 'response_decomposition_sample_labels.csv'}`",
    ]
    (REPORT_ROOT / "stage03_vehicle_instability_response_decomposition_labels_user_summary_cn.md").write_text(
        "\n".join(user_lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    ensure_dirs()
    manifest = pd.read_csv(TASK_MANIFEST_PATH)
    label_parts = []
    for track_id, cfg in TRACKS.items():
        rows, y, y_mask, label_time = load_track_rows(manifest, track_id, cfg)
        labels = compute_raw_labels(rows, y, y_mask, label_time, track_id)
        labels["track_description_cn"] = cfg["description_cn"]
        label_parts.append(labels)
    raw_labels = pd.concat(label_parts, ignore_index=True)
    labels, thresholds = add_train_split_threshold_labels(raw_labels)
    summaries = build_summaries(labels)

    labels.to_csv(TABLE_DIR / "response_decomposition_sample_labels.csv", index=False, encoding="utf-8-sig")
    thresholds.to_csv(TABLE_DIR / "response_decomposition_train_thresholds.csv", index=False, encoding="utf-8-sig")
    for name, df in summaries.items():
        df.to_csv(TABLE_DIR / f"response_decomposition_{name}.csv", index=False, encoding="utf-8-sig")

    plot_count_bars(labels)
    plot_peak_scatter(labels)
    plot_b_mean_trajectories(labels)

    write_reports(labels, thresholds, summaries)

    summary = {
        "output_version": "stage03_vehicle_instability_response_decomposition_labels_v0_1",
        "tracks": summaries["track_summary"].to_dict(orient="records"),
        "n_sample_labels": int(len(labels)),
        "threshold_scope": "session-level train split per track",
        "label_usage": "targets/evaluation only; not model input",
        "server_used": False,
        "credential_file_read": False,
        "uses_physio": False,
        "uses_eeg": False,
        "uses_continuous_style": False,
        "uses_subject_id_as_model_input": False,
        "new_training_run": False,
    }
    (LOG_DIR / "response_decomposition_labels_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
