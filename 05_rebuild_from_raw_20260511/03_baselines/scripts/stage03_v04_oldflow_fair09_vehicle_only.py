# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(os.environ.get("DATA_PROCESS_ROOT", r"F:/data_set_process/data_process"))
REBUILD_ROOT = PROJECT_ROOT / "05_rebuild_from_raw_20260511"
OLD_TRAIN_DIR = PROJECT_ROOT / "02_code" / "final_code" / "model" / "training"
OLD_FAIR_DIR = OLD_TRAIN_DIR / "fair_vehicle_event_comparison_20260427"
SCRIPT_DIR = REBUILD_ROOT / "03_baselines" / "scripts"
SAMPLE_SCRIPT_DIR = REBUILD_ROOT / "02_samples" / "scripts"

for path in [SCRIPT_DIR, SAMPLE_SCRIPT_DIR, OLD_FAIR_DIR, OLD_TRAIN_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import build_oldcode_deep_clean_vehicle_manifest_v0_1 as clean_v01  # noqa: E402
import stage03_v03_vehicle_only_inclusion_ablation as split_v03  # noqa: E402
from common_compare_runner import build_args  # noqa: E402
from prediction_plotting import save_prediction_plots_for_run  # noqa: E402
from run_event_conditioned_trajectory_baseline import build_sample_bundle_from_manifest, train_one_run  # noqa: E402


RUN_ID = "stage03_v04_oldflow_fair09_vehicle_only"
DATASET_ROOT = REBUILD_ROOT / "03_processed_datasets" / RUN_ID
TABLE_DIR = DATASET_ROOT / "tables"
CLEAN_ROOT = DATASET_ROOT / "oldflow_clean_vehicle_csv_v0_4"
LOG_DIR = DATASET_ROOT / "logs"
OUT_DIR = REBUILD_ROOT / "03_baselines" / RUN_ID
REPORT_DIR = REBUILD_ROOT / "09_reports"
NOTES_DIR = REBUILD_ROOT / "00_project_notes"
DAILY_LOG = NOTES_DIR / "daily_logs" / "2026-05-20.md"
ARTIFACT_INDEX = NOTES_DIR / "ARTIFACT_INDEX_CN.md"

V04_TABLE_DIR = REBUILD_ROOT / "02_samples" / "extreme_condition_episodes_v0_4" / "tables"
V04_PRIMARY = V04_TABLE_DIR / "primary_train_episodes_v0_4.csv"
V04_SECONDARY = V04_TABLE_DIR / "secondary_train_episodes_v0_4.csv"
V04_REVIEW = V04_TABLE_DIR / "manual_review_episodes_v0_4.csv"

RAW_VEHICLE_ROOT = PROJECT_ROOT / "01_datasets" / "数据预处理" / "原始车辆数据"
MULTIMODAL_SUBJECT_ROOT = PROJECT_ROOT / "01_datasets" / "多模态数据" / "被试数据集合"

MANIFEST_PATH = TABLE_DIR / "oldflow_fair09_vehicle_only_v04_primary_secondary_review_manifest.csv"
MANIFEST_USED_CHECK_PATH = TABLE_DIR / "oldflow_fair09_vehicle_only_manifest_validity_check.json"
RUN_RECORD_PATH = OUT_DIR / "tables" / "oldflow_fair09_vehicle_only_run_record.csv"
REPORT_PATH = REPORT_DIR / "stage03_v04_oldflow_fair09_vehicle_only_user_summary_cn.md"
CASE_FILE = OUT_DIR / "tables" / "oldflow_fair09_vehicle_only_selected_cases_test.csv"

FS_OLD = 200
OLD_HISTORY_LEN = 600
OLD_FUTURE_LEN = 400


def ensure_dirs() -> None:
    for path in [TABLE_DIR, CLEAN_ROOT, LOG_DIR, OUT_DIR / "tables", OUT_DIR / "logs", REPORT_DIR, NOTES_DIR / "daily_logs"]:
        path.mkdir(parents=True, exist_ok=True)


def read_v04_inputs() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for source_name, path in [
        ("primary", V04_PRIMARY),
        ("secondary", V04_SECONDARY),
        ("manual_review", V04_REVIEW),
    ]:
        df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        df["v04_source_group"] = source_name
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates("episode_uid").sort_values(["subject", "session_stamp", "t_condition_anchor"]).reset_index(drop=True)
    return out


def configure_cleaner() -> None:
    clean_v01.DATASET_DIR = DATASET_ROOT
    clean_v01.TABLE_DIR = TABLE_DIR
    clean_v01.CLEAN_ROOT = CLEAN_ROOT
    clean_v01.LOG_DIR = LOG_DIR
    clean_v01.REPORT_DIR = REPORT_DIR


def clean_vehicle_files(episodes: pd.DataFrame) -> pd.DataFrame:
    status_path = TABLE_DIR / "oldflow_clean_vehicle_status_v0_4.csv"
    if status_path.exists():
        status = pd.read_csv(status_path, encoding="utf-8-sig", low_memory=False)
        if not status.empty and {"raw_vehicle_file", "clean_vehicle_file", "status"}.issubset(status.columns):
            return status

    configure_cleaner()
    episodes = episodes.copy()
    episodes["vehicle_source_file_for_oldflow"] = episodes.apply(resolve_vehicle_source_file, axis=1)
    unique_files = (
        episodes[["subject", "vehicle_raw_absolute_path", "vehicle_source_file_for_oldflow"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["subject", "vehicle_raw_absolute_path"])
        .reset_index(drop=True)
    )
    rows = []
    for row in unique_files.itertuples(index=False):
        cleaned = clean_v01.clean_one_vehicle(Path(str(row.vehicle_source_file_for_oldflow)), subject=str(row.subject))
        cleaned["raw_vehicle_file_requested"] = str(row.vehicle_raw_absolute_path)
        cleaned["vehicle_source_file_for_oldflow"] = str(row.vehicle_source_file_for_oldflow)
        rows.append(cleaned)
    status = pd.DataFrame(rows)
    status.to_csv(status_path, index=False, encoding="utf-8-sig")
    return status


def resolve_vehicle_source_file(ep: pd.Series) -> str:
    requested = Path(str(ep.get("vehicle_raw_absolute_path", "")))
    if requested.exists():
        return str(requested)

    rel = str(ep.get("vehicle_raw_relative_path", "")).replace("\\", "/")
    if rel and rel.lower() != "nan":
        candidate = RAW_VEHICLE_ROOT / rel
        if candidate.exists():
            return str(candidate)

    subject = str(ep.get("subject", "")).strip()
    session = str(ep.get("session_stamp", "")).strip()
    if subject and session and subject.lower() != "nan" and session.lower() != "nan":
        vehicle_dir = MULTIMODAL_SUBJECT_ROOT / subject / "vehicle"
        for name in [
            f"Entity_Recording_{session}_vehicle.csv",
            f"Entity_Recording_{session}_vehicle_aligned_cleaned.csv",
            f"Entity_Recording_{session}_vehicle_aligned_cleaned_roadtype_labeled.csv",
        ]:
            candidate = vehicle_dir / name
            if candidate.exists():
                return str(candidate)

    if rel and rel.lower() != "nan":
        rel_path = Path(rel)
        subject_from_rel = rel_path.parts[0] if rel_path.parts else subject
        stem = rel_path.stem
        vehicle_dir = MULTIMODAL_SUBJECT_ROOT / subject_from_rel / "vehicle"
        for name in [
            rel_path.name,
            f"{stem}_aligned_cleaned.csv",
            f"{stem}_aligned_cleaned_roadtype_labeled.csv",
        ]:
            candidate = vehicle_dir / name
            if candidate.exists():
                return str(candidate)

    return str(requested)


def assign_splits(episodes: pd.DataFrame) -> pd.Series:
    sample_split, session_split = split_v03.load_reference_split()
    split_meta = episodes.copy()
    split_meta["sample_id"] = split_meta["episode_uid"].astype(str)
    split_meta["vehicle_raw_relative_path"] = split_meta["vehicle_raw_relative_path"].astype(str)
    return split_v03.assign_split(split_meta, sample_split, session_split, seed=20260518)


def build_manifest(episodes: pd.DataFrame, clean_status: pd.DataFrame) -> pd.DataFrame:
    ok_map: dict[str, str] = {}
    for row in clean_status.itertuples(index=False):
        if str(row.status) != "ok":
            continue
        clean_file = str(row.clean_vehicle_file)
        for key in [
            getattr(row, "raw_vehicle_file", ""),
            getattr(row, "raw_vehicle_file_requested", ""),
            getattr(row, "vehicle_source_file_for_oldflow", ""),
        ]:
            if key:
                ok_map[str(key)] = clean_file
    split = assign_splits(episodes)
    rows: list[dict[str, Any]] = []
    dropped_rows: list[dict[str, Any]] = []
    for event_idx, (idx, ep) in enumerate(episodes.iterrows(), start=1):
        raw_file = str(ep["vehicle_raw_absolute_path"])
        clean_file = ok_map.get(raw_file)
        if not clean_file:
            dropped_rows.append({"episode_uid": ep.get("episode_uid", ""), "drop_reason": "clean_vehicle_missing"})
            continue
        anchor_s = float(ep["t_condition_anchor"])
        anchor_idx = int(round(anchor_s * FS_OLD))
        if anchor_idx < 0:
            dropped_rows.append({"episode_uid": ep.get("episode_uid", ""), "drop_reason": "negative_anchor_idx"})
            continue
        valid_future_len = OLD_FUTURE_LEN
        sample_key = (
            f"v04::{ep.get('subject', 'unknown')}::{ep.get('session_stamp', 'unknown')}::"
            f"{event_idx:05d}::{ep.get('episode_uid', event_idx)}"
        )
        rows.append(
            {
                "protocol_version": "v04_oldflow_fair09_vehicle_only",
                "sample_key": sample_key,
                "pool": "v04_primary_secondary_review",
                "subj": str(ep.get("subject", "unknown")),
                "split": str(split.loc[idx]),
                "file": Path(clean_file).name,
                "recording_id": str(ep.get("session_stamp", Path(clean_file).stem)),
                "vehicle_file": clean_file,
                "event_idx": int(event_idx),
                "episode_id": int(event_idx),
                "source_event_version": "extreme_condition_episodes_v0_4",
                "phase_type": str(ep.get("v04_source_group", "")),
                "event_level": str(ep.get("condition_level", "")),
                "trigger_type": str(ep.get("condition_context_cn", "")),
                "event_type": str(ep.get("condition_context_cn", "")),
                "road_type_anchor": str(ep.get("condition_context_cn", "")),
                "is_curve": int(bool(ep.get("is_curve_context", False))),
                "curvature_anchor": float(pd.to_numeric(ep.get("peak_abs_curvature_window", 0.0), errors="coerce") or 0.0),
                "anchor_source": "v04_condition_anchor",
                "anchor_idx": int(anchor_idx),
                "anchor_s": float(anchor_s),
                "history_len": OLD_HISTORY_LEN,
                "future_len": OLD_FUTURE_LEN,
                "valid_future_len": int(valid_future_len),
                "valid_future_s": 2.0,
                "full_future_2s": "True",
                "history_full_3s": "unknown_until_old_loader",
                "time_left_after_anchor_s": float(ep.get("time_end_s", np.nan)) - anchor_s,
                "keep_for_training": "True",
                "usable_sample": "True",
                "drop_reason": "",
                "event_start_s": float(ep.get("t_condition_anchor", anchor_s)),
                "event_end_s": float(ep.get("t_condition_end", anchor_s)),
                "event_duration_s": float(ep.get("condition_duration_s", 0.0)),
                "start_idx": int(round(float(ep.get("t_condition_anchor", anchor_s)) * FS_OLD)),
                "end_idx": int(round(float(ep.get("t_condition_end", anchor_s)) * FS_OLD)),
                "trigger_score": float(pd.to_numeric(ep.get("condition_score_peak", 0.0), errors="coerce") or 0.0),
                "primary_score": float(pd.to_numeric(ep.get("v04_post_vehicle_dyn_score", 0.0), errors="coerce") or 0.0),
                "mechanism_tag": str(ep.get("v04_label_cn", "")),
                "d3_included": "False",
                "instability_event_uid": str(ep.get("episode_uid", sample_key)),
                "leakage_note": "v0.4 event anchor; oldflow FAIR09 vehicle-only, no style/physio/eeg/teacher.",
                "raw_vehicle_file_before_cleaning": raw_file,
                "v04_source_group": str(ep.get("v04_source_group", "")),
                "v04_label": str(ep.get("v04_label", "")),
                "v04_label_cn": str(ep.get("v04_label_cn", "")),
                "v04_reason_cn": str(ep.get("v04_reason_cn", "")),
                "v04_post_vehicle_dyn_score": ep.get("v04_post_vehicle_dyn_score", np.nan),
                "v04_post_steer_delta": ep.get("v04_post_steer_delta", np.nan),
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(MANIFEST_PATH, index=False, encoding="utf-8-sig")
    pd.DataFrame(dropped_rows).to_csv(TABLE_DIR / "oldflow_manifest_build_dropped_v0_4.csv", index=False, encoding="utf-8-sig")
    return manifest


def verify_manifest(path: Path) -> dict[str, Any]:
    try:
        _, _, _, _, _, meta_df, dropped = build_sample_bundle_from_manifest(
            manifest_path=path,
            max_train_samples=None,
            max_val_samples=None,
            max_test_samples=None,
            seed=2026,
        )
        summary = {
            "status": "ok",
            "manifest_rows": int(pd.read_csv(path, encoding="utf-8-sig", low_memory=False).shape[0]),
            "old_loader_kept_rows": int(len(meta_df)),
            "old_loader_dropped_rows": int(dropped),
            "split_counts_after_old_loader": meta_df["split"].astype(str).value_counts().to_dict(),
            "source_group_counts_after_old_loader": meta_df.get("v04_source_group", pd.Series(dtype=str)).astype(str).value_counts().to_dict(),
        }
    except Exception as exc:
        summary = {"status": "error", "error": repr(exc)}
    MANIFEST_USED_CHECK_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def oldflow_fair09_args(manifest: Path, device: str, run_prefix: str) -> argparse.Namespace:
    args = build_args("vehicle_direct_coarse_fine")
    args.manifest = str(manifest)
    args.run_prefix = run_prefix
    args.seed = 2026
    args.device = device
    args.epochs = 40
    args.min_epochs = 40
    args.patience = 99
    args.batch_size = 64
    args.lr = 1e-3
    args.weight_decay = 0.0
    args.grad_clip = 1.0
    args.selection_mode = "legacy_rmse"
    return args


def write_report(
    manifest: pd.DataFrame,
    verify_summary: dict[str, Any],
    train_result: dict[str, Any] | None,
    plot_result: dict[str, Any] | None,
    run_device: str,
) -> None:
    if train_result is None:
        result_text = "本次只完成 manifest 准备，尚未训练。"
        run_root = ""
        test_rmse = np.nan
        primary = np.nan
        tail = np.nan
        selection = np.nan
    else:
        run_root = str(train_result["run_root"])
        test_rmse = float(train_result["test_metrics"]["steer_rmse"])
        selection_summary = train_result["test_metrics"].get("selection_summary", {})
        primary = float(selection_summary.get("overall_primary_steer_rmse", np.nan))
        tail = float(selection_summary.get("rmse_tail_abs_steer", np.nan))
        selection = float(selection_summary.get("selection_score", np.nan))
        result_text = (
            f"- test steer RMSE：{test_rmse:.6f}\n"
            f"- primary RMSE：{primary:.6f}\n"
            f"- tail RMSE：{tail:.6f}\n"
            f"- selection：{selection:.6f}\n"
        )
    split_counts = manifest["split"].astype(str).value_counts().to_dict() if len(manifest) else {}
    source_counts = manifest["v04_source_group"].astype(str).value_counts().to_dict() if "v04_source_group" in manifest else {}
    plot_text = ""
    if plot_result:
        plot_text = (
            f"- 预测总览图：`{plot_result.get('overview_path')}`\n"
            f"- 预测图目录：`{plot_result.get('figures_dir')}`\n"
            f"- 逐样本指标：`{Path(plot_result.get('figures_dir', '')) / 'prediction_sample_metrics.csv'}`\n"
        )
    report = f"""# v0.4 新样本集：旧流程 FAIR09 车辆-only 粗细双头

生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}

## 这次对齐的是什么

用户要求对齐截图里旧流程那套模型和参数，而不是使用 `05_rebuild` 里的新 Transformer。

本次采用旧流程车辆-only 版本：

- 版本口径：`FAIR09 / E1`，即“车辆数据 + 粗细双头 + 无显式事件注入”。
- 不使用连续驾驶风格。
- 不使用生理数据。
- 不使用脑电。
- 不使用教师蒸馏。
- 只把 v0.4 新筛出来的样本清单接入旧流程模型。

## 旧流程训练参数

- seed：2026
- device：`{run_device}`
- epochs：40
- min_epochs：40
- batch_size：64
- lr：1e-3
- weight_decay：0
- grad_clip：1.0
- selection_mode：legacy_rmse
- 模型：历史车辆 Transformer 编码器 + 粗细双头轨迹解码器
- d_model：128
- nhead：2
- encoder layers：2
- decoder layers：2
- ffn_dim：256
- dropout：0.1
- event_embed_dim：96
- event_bin_size：20
- conditioning_mode：vehicle_direct_coarse_fine
- teacher_forcing_ratio：0
- event_loss_weight：0

## 样本来源

- v0.4 主训练候选 + 次级训练候选 + 待复核样本。
- manifest 原始行数：{len(manifest)}
- manifest split：{split_counts}
- 样本来源分布：{source_counts}
- 旧流程实际可读取：{verify_summary}

## 当前结果

{result_text}

## 输出位置

- 旧流程 manifest：`{MANIFEST_PATH}`
- manifest 检查：`{MANIFEST_USED_CHECK_PATH}`
- 运行目录：`{run_root}`
- 运行记录：`{RUN_RECORD_PATH}`
{plot_text}

## 解释边界

这一步只回答：在新筛样本集上，旧流程“粗细双头车辆-only”能做到什么程度。
它不能证明连续风格、生理数据或脑电有效，也不能直接替代后续更严格的新流程车辆基线。
"""
    REPORT_PATH.write_text(report, encoding="utf-8")

    row = {
        "run_id": RUN_ID,
        "manifest": str(MANIFEST_PATH),
        "run_root": run_root,
        "device": run_device,
        "model": "FAIR09/E1 vehicle_direct_coarse_fine",
        "seed": 2026,
        "epochs": 40,
        "batch_size": 64,
        "lr": 1e-3,
        "test_steer_rmse": test_rmse,
        "primary_rmse": primary,
        "tail_rmse": tail,
        "selection": selection,
        "report": str(REPORT_PATH),
        "overview": "" if not plot_result else str(plot_result.get("overview_path", "")),
    }
    pd.DataFrame([row]).to_csv(RUN_RECORD_PATH, index=False, encoding="utf-8-sig")


def append_project_notes(verify_summary: dict[str, Any], train_result: dict[str, Any] | None) -> None:
    test_rmse = "未训练" if train_result is None else f"{float(train_result['test_metrics']['steer_rmse']):.6f}"
    block = (
        "## 2026-05-20 v0.4 新样本集旧流程 FAIR09 车辆-only\n\n"
        "- 为什么做：用户要求使用截图中旧流程的模型和参数，只看车辆-only，便于和 E1/FAIR09 口径比较。\n"
        "- 模型口径：粗细双头、不加连续风格、不加生理/脑电、不加教师，40 轮，batch=64，lr=1e-3，seed=2026。\n"
        f"- manifest 检查：`{MANIFEST_USED_CHECK_PATH}`，摘要：{verify_summary}。\n"
        f"- 当前 test RMSE：{test_rmse}。\n"
        f"- 用户查看版报告：`{REPORT_PATH}`。\n"
        f"- 输出目录：`{OUT_DIR}`。\n"
    )
    for path in [NOTES_DIR / "PROJECT_STATUS_CN.md", NOTES_DIR / "TASK_QUEUE_CN.md", DAILY_LOG]:
        raw = path.read_text(encoding="utf-8") if path.exists() else ""
        if "## 2026-05-20 v0.4 新样本集旧流程 FAIR09 车辆-only" not in raw:
            path.write_text(raw.rstrip() + "\n\n" + block, encoding="utf-8")

    artifact = (
        "## 2026-05-20 v0.4 新样本集旧流程 FAIR09 车辆-only\n\n"
        f"- 用户查看版报告：`{REPORT_PATH}`\n"
        f"- 旧流程 manifest：`{MANIFEST_PATH}`\n"
        f"- manifest 检查：`{MANIFEST_USED_CHECK_PATH}`\n"
        f"- 运行记录：`{RUN_RECORD_PATH}`\n"
        f"- 输出目录：`{OUT_DIR}`\n"
    )
    raw = ARTIFACT_INDEX.read_text(encoding="utf-8") if ARTIFACT_INDEX.exists() else ""
    if "## 2026-05-20 v0.4 新样本集旧流程 FAIR09 车辆-only" not in raw:
        ARTIFACT_INDEX.write_text(raw.rstrip() + "\n\n" + artifact, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    episodes = read_v04_inputs()
    clean_status = clean_vehicle_files(episodes)
    manifest = build_manifest(episodes, clean_status)
    verify_summary = verify_manifest(MANIFEST_PATH)
    if verify_summary.get("status") != "ok":
        write_report(manifest, verify_summary, None, None, str(args.device))
        append_project_notes(verify_summary, None)
        raise RuntimeError(f"manifest verification failed: {verify_summary}")

    if args.prepare_only:
        write_report(manifest, verify_summary, None, None, str(args.device))
        append_project_notes(verify_summary, None)
        print(json.dumps({"manifest": str(MANIFEST_PATH), "verify": verify_summary}, ensure_ascii=False, indent=2))
        return

    run_args = oldflow_fair09_args(
        MANIFEST_PATH,
        device=str(args.device),
        run_prefix="V04_OLD_FLOW_FAIR09_vehicle_only_coarse_fine_seed2026",
    )
    train_result = train_one_run(run_args)
    plot_result = None
    if not args.skip_plots:
        plot_result = save_prediction_plots_for_run(
            run_root=train_result["run_root"],
            split="test",
            case_file=CASE_FILE,
            max_cases=12,
            batch_size=int(run_args.batch_size),
            device=str(args.device),
            force_rebuild_cases=True,
        )
    write_report(manifest, verify_summary, train_result, plot_result, str(args.device))
    append_project_notes(verify_summary, train_result)
    print(
        json.dumps(
            {
                "manifest": str(MANIFEST_PATH),
                "verify": verify_summary,
                "run_root": str(train_result["run_root"]),
                "test_steer_rmse": float(train_result["test_metrics"]["steer_rmse"]),
                "report": str(REPORT_PATH),
                "plot": plot_result,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
