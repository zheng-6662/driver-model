# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(os.environ.get("DATA_PROCESS_ROOT", r"F:/data_set_process/data_process"))
ROOT = Path(os.environ.get("REBUILD_ROOT", str(PROJECT_ROOT / "05_rebuild_from_raw_20260511")))
SCRIPT_DIR = ROOT / "03_baselines" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import stage03_v03_screening_sweep_gpu as gpu  # noqa: E402
import stage03_v03_vehicle_only_baselines as base  # noqa: E402
import stage03_v03_vehicle_only_inclusion_ablation as incl  # noqa: E402


matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False


OUT_ROOT = ROOT / "03_baselines" / "stage03_v11_vehicle_only_gpu_baseline"
DATASET_ROOT = ROOT / "03_processed_datasets" / "record_episode_v1_1_vehicle_only_gpu"
REPORT_DIR = ROOT / "09_reports"
NOTES_DIR = ROOT / "00_project_notes"
DAILY_LOG = NOTES_DIR / "daily_logs" / "2026-05-20.md"
ARTIFACT_INDEX = NOTES_DIR / "ARTIFACT_INDEX_CN.md"

V11_ROOT = ROOT / "02_samples" / "record_level_episode_reconstruction_v1_1_reviewed"
V11_TABLE_DIR = V11_ROOT / "tables"
V11_TRAIN = V11_TABLE_DIR / "train_candidate_extreme_episodes_v1_1.csv"

SUMMARY_PATH = OUT_ROOT / "tables" / "v11_vehicle_only_gpu_summary.csv"
RANKING_PATH = OUT_ROOT / "tables" / "v11_vehicle_only_gpu_ranking.csv"
REPORT_PATH = REPORT_DIR / "stage03_v11_vehicle_only_gpu_user_summary_cn.md"

DROP_LATERAL_FEATURES = ["lateral_distance_selected"]
TEST_SUBJECTS = {"cwh", "gf", "tyy"}
VAL_SUBJECTS = {"byx", "gzj", "yyl"}


def ensure_dirs() -> None:
    for path in [OUT_ROOT / "tables", OUT_ROOT / "figures", OUT_ROOT / "logs", REPORT_DIR, NOTES_DIR / "daily_logs"]:
        path.mkdir(parents=True, exist_ok=True)


def configure_modules() -> None:
    gpu.OUT_ROOT = OUT_ROOT
    gpu.DATASET_ROOT = DATASET_ROOT
    gpu.REPORT_DIR = REPORT_DIR
    gpu.NOTES_DIR = NOTES_DIR
    gpu.DAILY_LOG = DAILY_LOG
    gpu.ARTIFACT_INDEX = ARTIFACT_INDEX
    gpu.SUMMARY_PATH = SUMMARY_PATH
    gpu.RANKING_PATH = RANKING_PATH

    incl.OUT_ROOT = OUT_ROOT
    incl.DATASET_ROOT = DATASET_ROOT
    incl.REPORT_DIR = REPORT_DIR
    incl.NOTES_DIR = NOTES_DIR
    incl.DAILY_LOG = DAILY_LOG
    incl.ARTIFACT_INDEX = ARTIFACT_INDEX

    base.TABLE_DIR = OUT_ROOT / "tables"
    base.FIG_DIR = OUT_ROOT / "figures"
    base.LOG_DIR = OUT_ROOT / "logs"


def coerce_numeric(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def choose_context(row: pd.Series) -> str:
    for col in ["road_module_names", "road_design_categories", "episode_group_cn"]:
        value = str(row.get(col, "")).strip()
        if value and value.lower() != "nan":
            return value
    flags = []
    for col, name in [
        ("is_low_mu_context", "低附着"),
        ("is_curve_context", "弯道"),
        ("is_roll_context", "横滚/姿态"),
        ("is_lateral_dynamic_context", "横向动态"),
    ]:
        if bool(row.get(col, False)):
            flags.append(name)
    return "+".join(flags) if flags else "未知上下文"


def build_compat_episode_table(anchor_col: str, anchor_label: str) -> tuple[Path, pd.DataFrame]:
    src = pd.read_csv(V11_TRAIN, encoding="utf-8-sig", low_memory=False)
    src[anchor_col] = pd.to_numeric(src[anchor_col], errors="coerce")
    src = src[np.isfinite(src[anchor_col])].copy()
    src["vehicle_raw_absolute_path"] = src["vehicle_file"].astype(str)
    src["vehicle_raw_relative_path"] = src["vehicle_file"].astype(str)
    src["t_condition_anchor"] = src[anchor_col].astype(float)
    src["v0_3_category"] = src["episode_group_id"].astype(str)
    src["v0_3_category_cn"] = src["episode_group_cn"].astype(str)
    src["condition_context_cn"] = src.apply(choose_context, axis=1)
    src["condition_level"] = src.get("vehicle_risk_level", "").astype(str)
    src["steer_response_strength"] = src.get("driver_response_type", "").astype(str)
    src["response_shape"] = src.get("response_order", "").astype(str)
    src["condition_score_mean"] = coerce_numeric(src, "condition_score_peak")
    src["median_speed_kmh_window"] = coerce_numeric(src, "median_speed_kmh")
    src["peak_abs_ay_window"] = coerce_numeric(src, "peak_abs_ay")
    src["peak_abs_yaw_rate_window"] = coerce_numeric(src, "peak_abs_yaw_rate")
    src["peak_abs_roll_rate_window"] = coerce_numeric(src, "peak_abs_roll_rate")
    src["peak_abs_roll_window"] = coerce_numeric(src, "peak_abs_roll")
    src["peak_abs_curvature_window"] = 0.0
    src["min_mu_window"] = coerce_numeric(src, "min_mu")
    src["anchor_source_for_training"] = anchor_col

    out = OUT_ROOT / "tables" / f"v11_compat_{anchor_label}.csv"
    src.to_csv(out, index=False, encoding="utf-8-sig")
    return out, src


def fixed_subject_split(episodes: pd.DataFrame) -> tuple[dict[str, str], dict[str, str]]:
    split_map: dict[str, str] = {}
    for _, row in episodes.iterrows():
        subject = str(row["subject"])
        if subject in TEST_SUBJECTS:
            split = "test"
        elif subject in VAL_SUBJECTS:
            split = "val"
        else:
            split = "train"
        split_map[str(row["episode_uid"])] = split
    return split_map, {}


def make_variant(variant_id: str, name_cn: str, anchor_label: str, categories: list[str], with_lateral: bool) -> dict[str, Any]:
    item: dict[str, Any] = {
        "variant_id": variant_id,
        "name_cn": name_cn,
        "description_cn": "v1.1 完整记录级 episode 主训练候选的车辆-only GPU 基线，不加入连续风格、生理或脑电。",
        "categories": categories,
        "anchor_label": anchor_label,
    }
    if not with_lateral:
        item["drop_features"] = DROP_LATERAL_FEATURES
    return item


def save_prediction_arrays(
    table_dir: Path,
    variant_id: str,
    selected_model: str,
    y: np.ndarray,
    y_mask: np.ndarray,
    meta: pd.DataFrame,
    preds: dict[str, np.ndarray],
) -> Path:
    out = table_dir / f"{variant_id}_selected_predictions.npz"
    keep = {"selected_prediction": preds[selected_model]}
    for name in ["zero_delta", "train_category_mean", "train_global_mean", "linear_trend_from_last_rate"]:
        if name in preds:
            keep[name] = preds[name]
    np.savez_compressed(
        out,
        y=y.astype(np.float32),
        y_mask=y_mask.astype(bool),
        label_time=base.LABEL_TIME.astype(np.float32),
        sample_id=meta["sample_id"].astype(str).to_numpy(dtype=object),
        split=meta["split"].astype(str).to_numpy(dtype=object),
        selected_model=np.array([selected_model], dtype=object),
        **keep,
    )
    return out


def plot_selected_predictions(
    sample_ids: list[str],
    y: np.ndarray,
    y_mask: np.ndarray,
    meta: pd.DataFrame,
    preds: dict[str, np.ndarray],
    selected_model: str,
    out_path: Path,
    title: str,
) -> None:
    sample_ids = [sid for sid in sample_ids if sid in set(meta["sample_id"].astype(str))]
    if not sample_ids:
        return
    rows = min(4, len(sample_ids))
    cols = int(math.ceil(len(sample_ids) / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.6, rows * 3.0), sharex=True)
    axes_arr = np.atleast_1d(axes).ravel()
    meta_index = {sid: i for i, sid in enumerate(meta["sample_id"].astype(str))}
    model_order = [selected_model, "zero_delta", "train_category_mean", "train_global_mean"]
    colors = {
        selected_model: "#2563EB",
        "zero_delta": "#9CA3AF",
        "train_category_mean": "#F59E0B",
        "train_global_mean": "#10B981",
    }
    for ax, sid in zip(axes_arr, sample_ids):
        i = meta_index[sid]
        valid = y_mask[i]
        ax.plot(base.LABEL_TIME[valid], y[i, valid], color="#111827", lw=2.0, label="真实")
        for model_name in model_order:
            if model_name in preds:
                label = "车辆模型" if model_name == selected_model else model_name
                ax.plot(base.LABEL_TIME[valid], preds[model_name][i, valid], lw=1.2, color=colors.get(model_name), label=label)
        ax.axhline(0.0, color="#111827", lw=0.6, alpha=0.4)
        subject = str(meta.loc[i, "subject"])
        category = str(meta.loc[i, "v0_3_category_cn"])
        peak = float(meta.loc[i, "target_peak_abs"]) if "target_peak_abs" in meta.columns else float("nan")
        ax.set_title(f"{subject} | {category} | peak={peak:.2f}", fontsize=8)
        ax.grid(True, alpha=0.25)
    for ax in axes_arr[len(sample_ids) :]:
        ax.axis("off")
    axes_arr[0].legend(fontsize=7, loc="best")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run_variant_with_plots(variant: dict[str, Any], sample_split: dict[str, str], session_split: dict[str, str], device: torch.device) -> dict[str, Any]:
    variant_id = str(variant["variant_id"])
    out_dir = OUT_ROOT / variant_id
    table_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    log_dir = out_dir / "logs"
    for path in [table_dir, fig_dir, log_dir]:
        path.mkdir(parents=True, exist_ok=True)

    summary_json = log_dir / f"{variant_id}_gpu_summary.json"
    if summary_json.exists():
        try:
            payload = json.loads(summary_json.read_text(encoding="utf-8"))
            if isinstance(payload.get("result"), dict):
                print(f"reuse {variant_id}", flush=True)
                return payload["result"]
        except Exception:
            pass

    feature_backup = list(base.VEHICLE_FEATURES)
    drop_features = set(str(x) for x in variant.get("drop_features") or [])
    if drop_features:
        base.VEHICLE_FEATURES = [x for x in feature_backup if x not in drop_features]
    try:
        x, x_mask, y, y_mask, meta, dataset_summary = incl.build_variant_dataset(variant, sample_split, session_split)
        train_idx = np.where(meta["split"].astype(str).to_numpy() == "train")[0]
        val_idx = np.where(meta["split"].astype(str).to_numpy() == "val")[0]
        test_idx = np.where(meta["split"].astype(str).to_numpy() == "test")[0]
        if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
            raise RuntimeError(f"{variant_id} split invalid: {dataset_summary.get('split_counts')}")
        X, _ = base.flatten_history_features(x, x_mask, meta)
        preds = base.build_no_learning_predictions(y, y_mask, x, x_mask, meta, train_idx)
        torch_preds, train_info = gpu.train_torch_models(X, y, y_mask, train_idx, val_idx, device)
        preds.update(torch_preds)
        metrics, per_sample = base.evaluate_all(y, y_mask, base.LABEL_TIME, meta, preds, train_idx)
    finally:
        base.VEHICLE_FEATURES = feature_backup

    metrics.to_csv(table_dir / f"{variant_id}_gpu_metrics.csv", index=False, encoding="utf-8-sig")
    per_sample.to_csv(table_dir / f"{variant_id}_gpu_per_sample_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(train_info).to_csv(table_dir / f"{variant_id}_gpu_train_info.csv", index=False, encoding="utf-8-sig")

    selected = gpu.select_by_val(metrics)
    if not selected:
        selected = metrics[metrics["split"].astype(str).eq("test")].sort_values("rmse_steer").iloc[0].to_dict()
        selected["val_selected_model"] = str(selected["model_name"])
        selected["val_rmse_for_selected"] = float("nan")
    selected_model = str(selected["val_selected_model"])
    prediction_path = save_prediction_arrays(table_dir, variant_id, selected_model, y, y_mask, meta, preds)

    selected_samples = per_sample[
        (per_sample["split"].astype(str) == "test") & (per_sample["model_name"].astype(str) == selected_model)
    ].copy()
    selected_samples["sample_rmse"] = pd.to_numeric(selected_samples["sample_rmse"], errors="coerce")
    selected_samples["gt_peak_abs"] = pd.to_numeric(selected_samples["gt_peak_abs"], errors="coerce")
    fixed_ids = (
        selected_samples[selected_samples["large_response"].astype(bool)]
        .sort_values("gt_peak_abs", ascending=False)["sample_id"]
        .astype(str)
        .head(12)
        .tolist()
    )
    bad_ids = selected_samples.sort_values("sample_rmse", ascending=False)["sample_id"].astype(str).head(12).tolist()
    plot_selected_predictions(
        fixed_ids,
        y,
        y_mask,
        meta,
        preds,
        selected_model,
        fig_dir / f"{variant_id}_large_response_overview.png",
        f"{variant['name_cn']}：大响应样本",
    )
    plot_selected_predictions(
        bad_ids,
        y,
        y_mask,
        meta,
        preds,
        selected_model,
        fig_dir / f"{variant_id}_badcase_overview.png",
        f"{variant['name_cn']}：误差较大样本",
    )

    result = {
        "variant_id": variant_id,
        "name_cn": variant["name_cn"],
        "anchor_label": variant.get("anchor_label", ""),
        "sample_count": int(dataset_summary["sample_count"]),
        "extra_episode_count": int(dataset_summary.get("extra_episode_count", 0)),
        "dropped_count": int(dataset_summary.get("dropped_count", 0)),
        "split_counts_json": json.dumps(dataset_summary.get("split_counts", {}), ensure_ascii=False),
        "category_counts_json": json.dumps(dataset_summary.get("category_counts", {}), ensure_ascii=False),
        "val_selected_model": selected_model,
        "val_rmse_for_selected": float(selected["val_rmse_for_selected"]),
        "test_rmse_steer": float(selected["rmse_steer"]),
        "test_primary_rmse_0_2s": float(selected["primary_rmse_0_2s"]),
        "test_tail_rmse_2_5s": float(selected["tail_rmse_2_5s"]),
        "test_wrong_side_rate_large": float(selected["wrong_side_rate_large"]),
        "test_severe_amp_under_rate_large": float(selected["severe_amp_under_rate_large"]),
        "test_large_response_recall": float(selected["large_response_recall"]),
        "prediction_npz": str(prediction_path),
        "large_response_plot": str(fig_dir / f"{variant_id}_large_response_overview.png"),
        "badcase_plot": str(fig_dir / f"{variant_id}_badcase_overview.png"),
    }
    summary_json.write_text(
        json.dumps({"dataset_summary": dataset_summary, "result": result}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def score_rows(summary: pd.DataFrame) -> pd.DataFrame:
    df = summary.copy()
    numeric_cols = [
        "test_rmse_steer",
        "test_primary_rmse_0_2s",
        "test_tail_rmse_2_5s",
        "test_wrong_side_rate_large",
        "test_severe_amp_under_rate_large",
        "test_large_response_recall",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    base_row = df[df["variant_id"].eq("v11_episode_start_nolat")]
    if len(base_row):
        b = base_row.iloc[0]
        df["delta_rmse_vs_start_nolat"] = df["test_rmse_steer"] - float(b["test_rmse_steer"])
        df["delta_wrong_side_vs_start_nolat"] = df["test_wrong_side_rate_large"] - float(b["test_wrong_side_rate_large"])
        df["delta_severe_under_vs_start_nolat"] = df["test_severe_amp_under_rate_large"] - float(
            b["test_severe_amp_under_rate_large"]
        )
        df["delta_large_recall_vs_start_nolat"] = df["test_large_response_recall"] - float(b["test_large_response_recall"])
    else:
        df["delta_rmse_vs_start_nolat"] = np.nan
        df["delta_wrong_side_vs_start_nolat"] = np.nan
        df["delta_severe_under_vs_start_nolat"] = np.nan
        df["delta_large_recall_vs_start_nolat"] = np.nan

    df["screening_score"] = (
        -df["delta_rmse_vs_start_nolat"].fillna(0.0)
        - 0.35 * df["delta_wrong_side_vs_start_nolat"].fillna(0.0)
        - 0.25 * df["delta_severe_under_vs_start_nolat"].fillna(0.0)
        + 0.15 * df["delta_large_recall_vs_start_nolat"].fillna(0.0)
    )
    return df.sort_values(["screening_score", "test_rmse_steer"], ascending=[False, True])


def fmt(value: Any) -> str:
    try:
        v = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(v):
        return "NA"
    return f"{v:.4f}"


def markdown_table(df: pd.DataFrame) -> str:
    cols = [
        "variant_id",
        "name_cn",
        "sample_count",
        "val_selected_model",
        "test_rmse_steer",
        "test_primary_rmse_0_2s",
        "test_tail_rmse_2_5s",
        "test_wrong_side_rate_large",
        "test_severe_amp_under_rate_large",
        "test_large_response_recall",
        "screening_score",
    ]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df[cols].iterrows():
        vals = []
        for col in cols:
            val = row[col]
            vals.append(str(val) if col in {"variant_id", "name_cn", "val_selected_model"} else fmt(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(ranking: pd.DataFrame, device: torch.device) -> None:
    best_score = ranking.iloc[0].to_dict()
    best_rmse = ranking.sort_values("test_rmse_steer").iloc[0].to_dict()
    lines = [
        "# v1.1 完整记录级样本车辆-only GPU 基线",
        "",
        "## 这次为什么做",
        "",
        "v1.1 是从完整原始车辆记录中重建 episode 后，经人工复核整理出的主训练候选。它不再直接继承旧 `.aed`、道路入口或旧锚点。本轮只训练车辆-only，目的是先看新样本定义本身是否能让车辆模型学到更稳定的方向盘后续变化。",
        "",
        "## 运行设置",
        "",
        f"- 设备：`{device}`，CUDA。不要加入连续风格、生理或脑电。",
        "- 样本入口：`train_candidate_extreme_episodes_v1_1.csv`。",
        "- 切分：test=`cwh/gf/tyy`，val=`byx/gzj/yyl`，其余被试为 train，用于和 v0.5 新样本阶段保持同类被试划分逻辑。",
        "- 输入：锚点前 2 秒车辆历史，20 Hz。",
        "- 标签：锚点后 5 秒方向盘相对变化，20 Hz。",
        "- 模型：无学习基线 + PyTorch 线性头/小型神经网络；按验证集 RMSE 选模型，再报告测试集。",
        "",
        "## 结果表",
        "",
        markdown_table(ranking),
        "",
        "## 当前读法",
        "",
        f"- 综合排序第一：`{best_score['variant_id']}`，test RMSE={fmt(best_score['test_rmse_steer'])}，大响应错侧率={fmt(best_score['test_wrong_side_rate_large'])}，严重幅值不足率={fmt(best_score['test_severe_amp_under_rate_large'])}。",
        f"- 单看整体 RMSE 最低：`{best_rmse['variant_id']}`，test RMSE={fmt(best_rmse['test_rmse_steer'])}。",
        "- 这次不能只看整体 RMSE。`episode 开始锚点，去横向偏移` 的 RMSE 最低，但验证集选中的是上下文均值模型，说明它更像把曲线平均化；它的大响应召回为 0，严重幅值不足率为 1.0，不符合极限工况建模目标。",
        "- `车辆响应开始锚点，去横向偏移` 的 RMSE 更高，但大响应错侧率最低，说明它更接近“方向不乱判”的物理目标；不过幅值仍明显不足，不能直接作为最终方案。",
        "- 因此本轮结论是：v1.1 样本能训练，但车辆-only 仍没有真正学好极限样本的幅值和形态。下一步应先继续改任务定义或输出形式，不急着加入风格/生理。",
        "- 如果车辆响应开始锚点明显好于 episode 开始锚点，说明当前 episode 开始点可能仍偏早或任务中包含较多前奏；如果 episode 开始点更好，说明它更适合预测完整后续变化。",
        "- 如果保留横向偏移改善错侧但恶化 RMSE，说明横向偏移可能是局部强提示，后续应按场景或质量分层使用。",
        "",
        "## 图和表",
        "",
        f"- 汇总表：`{SUMMARY_PATH}`",
        f"- 排名表：`{RANKING_PATH}`",
        f"- 输出目录：`{OUT_ROOT}`",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def append_notes(ranking: pd.DataFrame, device: torch.device) -> None:
    best = ranking.iloc[0].to_dict()
    block = (
        "## 2026-05-20 v1.1 完整记录级样本车辆-only GPU 基线\n\n"
        "- 为什么做：基于 v1.1 主训练候选样本，先训练车辆-only，检查新 episode 样本定义是否适合建模。\n"
        f"- 运行设备：`{device}`，本地 CUDA。\n"
        "- 切分：test=cwh/gf/tyy，val=byx/gzj/yyl，其余 train。\n"
        f"- 当前综合排序第一：`{best['variant_id']}`，test RMSE={fmt(best['test_rmse_steer'])}，大响应错侧率={fmt(best['test_wrong_side_rate_large'])}，严重幅值不足率={fmt(best['test_severe_amp_under_rate_large'])}，大响应召回={fmt(best['test_large_response_recall'])}。\n"
        f"- 用户查看版报告：`{REPORT_PATH}`。\n"
        f"- 输出目录：`{OUT_ROOT}`。\n"
    )
    for path in [NOTES_DIR / "PROJECT_STATUS_CN.md", NOTES_DIR / "TASK_QUEUE_CN.md", DAILY_LOG]:
        raw = path.read_text(encoding="utf-8") if path.exists() else ""
        if "## 2026-05-20 v1.1 完整记录级样本车辆-only GPU 基线" not in raw:
            path.write_text(raw.rstrip() + "\n\n" + block, encoding="utf-8")

    artifact = (
        "## 2026-05-20 v1.1 完整记录级样本车辆-only GPU 基线\n\n"
        f"- 用户查看版报告：`{REPORT_PATH}`\n"
        f"- 汇总表：`{SUMMARY_PATH}`\n"
        f"- 排名表：`{RANKING_PATH}`\n"
        f"- 输出目录：`{OUT_ROOT}`\n"
    )
    raw = ARTIFACT_INDEX.read_text(encoding="utf-8") if ARTIFACT_INDEX.exists() else ""
    if "## 2026-05-20 v1.1 完整记录级样本车辆-only GPU 基线" not in raw:
        ARTIFACT_INDEX.write_text(raw.rstrip() + "\n\n" + artifact, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    configure_modules()
    gpu.set_seed(20260520)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，本轮要求使用 GPU。")
    device = torch.device("cuda")
    print(f"cuda={torch.cuda.get_device_name(0)}", flush=True)

    start_table, start_df = build_compat_episode_table("episode_start_s", "episode_start")
    vehicle_table, vehicle_df = build_compat_episode_table("vehicle_response_onset_s", "vehicle_onset")
    categories = sorted(start_df["v0_3_category"].dropna().astype(str).unique().tolist())
    variants = [
        (start_table, start_df, make_variant("v11_episode_start_nolat", "v1.1 episode 开始锚点，去横向偏移", "episode_start", categories, False)),
        (start_table, start_df, make_variant("v11_episode_start_lat", "v1.1 episode 开始锚点，保留横向偏移", "episode_start", categories, True)),
        (vehicle_table, vehicle_df, make_variant("v11_vehicle_onset_nolat", "v1.1 车辆响应开始锚点，去横向偏移", "vehicle_response_onset", categories, False)),
    ]

    rows: list[dict[str, Any]] = []
    for table, table_df, variant in variants:
        base.EPISODE_TABLE = table
        sample_split, session_split = fixed_subject_split(table_df)
        print(f"run {variant['variant_id']} table_rows={len(table_df)} categories={categories}", flush=True)
        rows.append(run_variant_with_plots(variant, sample_split, session_split, device))

    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    ranking = score_rows(summary)
    ranking.to_csv(RANKING_PATH, index=False, encoding="utf-8-sig")
    write_report(ranking, device)
    append_notes(ranking, device)
    print(
        ranking[
            [
                "variant_id",
                "sample_count",
                "val_selected_model",
                "test_rmse_steer",
                "test_primary_rmse_0_2s",
                "test_tail_rmse_2_5s",
                "test_wrong_side_rate_large",
                "test_severe_amp_under_rate_large",
                "test_large_response_recall",
                "screening_score",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
