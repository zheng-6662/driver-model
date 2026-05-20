# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(os.environ.get("DATA_PROCESS_ROOT", r"F:/data_set_process/data_process"))
REBUILD_ROOT = Path(os.environ.get("REBUILD_ROOT", str(PROJECT_ROOT / "05_rebuild_from_raw_20260511")))
OLD_TRAIN_DIR = PROJECT_ROOT / "02_code" / "final_code" / "model" / "training"
OLD_FAIR_DIR = OLD_TRAIN_DIR / "fair_vehicle_event_comparison_20260427"
BASELINE_SCRIPT_DIR = REBUILD_ROOT / "03_baselines" / "scripts"
SAMPLE_SCRIPT_DIR = REBUILD_ROOT / "02_samples" / "scripts"

for path in [BASELINE_SCRIPT_DIR, SAMPLE_SCRIPT_DIR, OLD_FAIR_DIR, OLD_TRAIN_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import build_oldcode_deep_clean_vehicle_manifest_v0_1 as clean_v01  # noqa: E402
import refilter_extreme_condition_episodes_v0_4 as refilter_v04  # noqa: E402
from common_compare_runner import build_args  # noqa: E402
from prediction_plotting import save_prediction_plots_for_run  # noqa: E402
from run_event_conditioned_trajectory_baseline import build_sample_bundle_from_manifest, train_one_run  # noqa: E402


RUN_ID = "stage03_v05_server_aligned_subject_oldflow_fair09"
V05_SAMPLE_DIR = REBUILD_ROOT / "02_samples" / "extreme_condition_episodes_server_aligned_v0_5"
V05_TABLE_DIR = V05_SAMPLE_DIR / "tables"
DATASET_ROOT = REBUILD_ROOT / "03_processed_datasets" / RUN_ID
TABLE_DIR = DATASET_ROOT / "tables"
CLEAN_ROOT = DATASET_ROOT / "oldflow_clean_vehicle_csv_v0_5"
OUT_DIR = REBUILD_ROOT / "03_baselines" / RUN_ID
REPORT_DIR = REBUILD_ROOT / "09_reports"
NOTES_DIR = REBUILD_ROOT / "00_project_notes"

MULTIMODAL_SUBJECT_ROOT = PROJECT_ROOT / "01_datasets" / "多模态数据" / "被试数据集合"
RAW_VEHICLE_ROOT = PROJECT_ROOT / "01_datasets" / "数据预处理" / "原始车辆数据"

V05_ALL = V05_TABLE_DIR / "extreme_condition_episodes_refiltered_v0_5.csv"
V05_PRIMARY = V05_TABLE_DIR / "primary_train_episodes_v0_5.csv"
V05_SECONDARY = V05_TABLE_DIR / "secondary_train_episodes_v0_5.csv"
V05_REVIEW = V05_TABLE_DIR / "manual_review_episodes_v0_5.csv"
V05_TRAIN_CANDIDATE = V05_TABLE_DIR / "train_candidate_episodes_v0_5.csv"
V05_EXCLUDED = V05_TABLE_DIR / "excluded_episodes_v0_5.csv"

MANIFEST_PATH = TABLE_DIR / "oldflow_fair09_vehicle_only_server_aligned_v05_subject_split_manifest.csv"
VERIFY_PATH = TABLE_DIR / "oldflow_fair09_vehicle_only_server_aligned_v05_subject_split_manifest_check.json"
RUN_RECORD_PATH = OUT_DIR / "tables" / "server_aligned_v05_subject_oldflow_fair09_run_record.csv"
REPORT_PATH = REPORT_DIR / "stage03_v05_server_aligned_subject_oldflow_fair09_user_summary_cn.md"
CASE_FILE = OUT_DIR / "tables" / "server_aligned_v05_subject_oldflow_fair09_selected_cases_test.csv"

TEST_SUBJECTS = {"cwh", "gf", "tyy"}
VAL_SUBJECTS = {"byx", "gzj", "yyl"}
FS_OLD = 200
OLD_HISTORY_LEN = 600
OLD_FUTURE_LEN = 400


def ensure_dirs() -> None:
    for path in [
        V05_TABLE_DIR,
        TABLE_DIR,
        CLEAN_ROOT,
        OUT_DIR / "tables",
        OUT_DIR / "logs",
        REPORT_DIR,
        NOTES_DIR / "daily_logs",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def resolve_vehicle_source_file(path_text: str, subject: str | None = None, session_stamp: str | None = None) -> Path:
    raw_text = str(path_text)
    p = Path(raw_text)
    if p.exists():
        return p

    win = PureWindowsPath(raw_text.replace("/", "\\"))
    parts = list(win.parts)
    if "原始车辆数据" in parts:
        idx = parts.index("原始车辆数据")
        rel_parts = parts[idx + 1 :]
        if rel_parts:
            candidate = RAW_VEHICLE_ROOT.joinpath(*rel_parts)
            if candidate.exists():
                return candidate
            if subject is None and len(rel_parts) >= 2:
                subject = rel_parts[0]
            if session_stamp is None and rel_parts:
                stem = Path(rel_parts[-1]).stem
                session_stamp = stem.replace("Entity_Recording_", "").replace("_vehicle", "")

    subject = str(subject or "").strip()
    session_stamp = str(session_stamp or "").strip()
    if subject and session_stamp and subject.lower() != "nan" and session_stamp.lower() != "nan":
        vehicle_dir = MULTIMODAL_SUBJECT_ROOT / subject / "vehicle"
        for name in [
            f"Entity_Recording_{session_stamp}_vehicle.csv",
            f"Entity_Recording_{session_stamp}_vehicle_aligned_cleaned.csv",
            f"Entity_Recording_{session_stamp}_vehicle_aligned_cleaned_roadtype_labeled.csv",
        ]:
            candidate = vehicle_dir / name
            if candidate.exists():
                return candidate

    return p


def configure_refilter_module() -> None:
    refilter_v04.PROJECT_ROOT = PROJECT_ROOT
    refilter_v04.ROOT = REBUILD_ROOT
    refilter_v04.V03_DIR = REBUILD_ROOT / "02_samples" / "extreme_condition_episodes_v0_3"
    refilter_v04.V03_TABLE_DIR = refilter_v04.V03_DIR / "tables"
    refilter_v04.EPISODE_TABLE = refilter_v04.V03_TABLE_DIR / "extreme_condition_episodes_all_v0_3.csv"
    refilter_v04.FAST_TABLE = refilter_v04.V03_TABLE_DIR / "fast_steer_vehicle_response_split_v0_3.csv"
    refilter_v04.TIMING_TABLE = refilter_v04.V03_TABLE_DIR / "fast_steer_anchor_timing_audit_v0_3.csv"
    refilter_v04.OUT_DIR = V05_SAMPLE_DIR
    refilter_v04.TABLE_DIR = V05_TABLE_DIR
    refilter_v04.FIG_DIR = V05_SAMPLE_DIR / "figures"
    refilter_v04.PANEL_DIR = V05_SAMPLE_DIR / "figures" / "review_panels"
    refilter_v04.REPORT_DIR = REPORT_DIR
    refilter_v04.NOTES_DIR = NOTES_DIR
    refilter_v04.DAILY_LOG = NOTES_DIR / "daily_logs" / f"{time.strftime('%Y-%m-%d')}.md"
    refilter_v04.ARTIFACT_INDEX = NOTES_DIR / "ARTIFACT_INDEX_CN.md"

    original_load_vehicle = refilter_v04.load_vehicle

    def load_vehicle_from_server_aligned(path_text: str) -> pd.DataFrame | None:
        # Use path text only here because the v0.4 scoring code calls load_vehicle(path_text).
        # The subject/session fallback is handled before scoring by rewriting the absolute path.
        resolved = resolve_vehicle_source_file(path_text)
        return original_load_vehicle(str(resolved))

    refilter_v04.load_vehicle = load_vehicle_from_server_aligned


def build_server_aligned_refilter_tables() -> pd.DataFrame:
    configure_refilter_module()
    refilter_v04.ensure_dirs()
    episodes = pd.read_csv(refilter_v04.EPISODE_TABLE, encoding="utf-8-sig", low_memory=False)
    episodes = refilter_v04.merge_context_tables(episodes)
    episodes["vehicle_source_file_for_v05"] = episodes.apply(
        lambda row: str(
            resolve_vehicle_source_file(
                str(row.get("vehicle_raw_absolute_path", "")),
                subject=str(row.get("subject", "")),
                session_stamp=str(row.get("session_stamp", "")),
            )
        ),
        axis=1,
    )
    episodes["vehicle_raw_absolute_path_before_v05"] = episodes.get("vehicle_raw_absolute_path", "")
    episodes["vehicle_raw_absolute_path"] = episodes["vehicle_source_file_for_v05"]

    cache: dict[str, pd.DataFrame | None] = {}
    rows: list[dict[str, Any]] = []
    for idx, row in episodes.iterrows():
        if idx % 100 == 0:
            print(f"refilter server-aligned {idx}/{len(episodes)}", flush=True)
        rows.append(refilter_v04.score_row(row, cache))
    result = pd.DataFrame(rows)
    result = result.sort_values(["v04_recommended_use", "subject", "session_stamp", "t_condition_anchor"]).reset_index(drop=True)

    # Preserve the old v04 column names because downstream scripts already use them,
    # but write v05 filenames to make the data entrance explicit.
    result.to_csv(V05_ALL, index=False, encoding="utf-8-sig")
    result[result["v04_recommended_use"].eq("primary_train")].to_csv(V05_PRIMARY, index=False, encoding="utf-8-sig")
    result[result["v04_recommended_use"].eq("secondary_train")].to_csv(V05_SECONDARY, index=False, encoding="utf-8-sig")
    result[result["v04_recommended_use"].eq("review")].to_csv(V05_REVIEW, index=False, encoding="utf-8-sig")
    result[result["v04_recommended_use"].eq("exclude")].to_csv(V05_EXCLUDED, index=False, encoding="utf-8-sig")
    result[result["v04_recommended_use"].isin(["primary_train", "secondary_train"])].to_csv(
        V05_TRAIN_CANDIDATE, index=False, encoding="utf-8-sig"
    )
    result["v04_label"].value_counts().rename_axis("v04_label").reset_index(name="count").to_csv(
        V05_TABLE_DIR / "v05_label_counts.csv", index=False, encoding="utf-8-sig"
    )
    pd.crosstab(result["subject"], result["v04_recommended_use"]).reset_index().to_csv(
        V05_TABLE_DIR / "v05_subject_use_counts.csv", index=False, encoding="utf-8-sig"
    )
    pd.crosstab(result["condition_context_cn"], result["v04_recommended_use"]).reset_index().to_csv(
        V05_TABLE_DIR / "v05_context_use_counts.csv", index=False, encoding="utf-8-sig"
    )
    return result


def read_v05_inputs() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for source_name, path in [
        ("primary", V05_PRIMARY),
        ("secondary", V05_SECONDARY),
        ("manual_review", V05_REVIEW),
    ]:
        df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        df["v05_source_group"] = source_name
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates("episode_uid").sort_values(["subject", "session_stamp", "t_condition_anchor"]).reset_index(drop=True)
    return out


def configure_cleaner() -> None:
    clean_v01.DATASET_DIR = DATASET_ROOT
    clean_v01.TABLE_DIR = TABLE_DIR
    clean_v01.CLEAN_ROOT = CLEAN_ROOT
    clean_v01.LOG_DIR = DATASET_ROOT / "logs"
    clean_v01.REPORT_DIR = REPORT_DIR
    clean_v01.LOG_DIR.mkdir(parents=True, exist_ok=True)


def clean_vehicle_files(episodes: pd.DataFrame) -> pd.DataFrame:
    status_path = TABLE_DIR / "oldflow_clean_vehicle_status_v0_5.csv"
    if status_path.exists():
        status = pd.read_csv(status_path, encoding="utf-8-sig", low_memory=False)
        if not status.empty and {"raw_vehicle_file", "clean_vehicle_file", "status"}.issubset(status.columns):
            return status

    configure_cleaner()
    unique_files = (
        episodes[["subject", "vehicle_raw_absolute_path"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["subject", "vehicle_raw_absolute_path"])
        .reset_index(drop=True)
    )
    rows = []
    for row in unique_files.itertuples(index=False):
        source_file = resolve_vehicle_source_file(str(row.vehicle_raw_absolute_path), subject=str(row.subject))
        cleaned = clean_v01.clean_one_vehicle(source_file, subject=str(row.subject))
        cleaned["raw_vehicle_file_requested"] = str(row.vehicle_raw_absolute_path)
        cleaned["vehicle_source_file_for_oldflow"] = str(source_file)
        rows.append(cleaned)
    status = pd.DataFrame(rows)
    status.to_csv(status_path, index=False, encoding="utf-8-sig")
    return status


def subject_split(subject: str) -> str:
    s = str(subject).strip().lower()
    if s in TEST_SUBJECTS:
        return "test"
    if s in VAL_SUBJECTS:
        return "val"
    return "train"


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


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

    rows: list[dict[str, Any]] = []
    dropped_rows: list[dict[str, Any]] = []
    for event_idx, ep in enumerate(episodes.itertuples(index=False), start=1):
        ep_dict = ep._asdict()
        raw_file = str(ep_dict.get("vehicle_raw_absolute_path", ""))
        clean_file = ok_map.get(raw_file)
        if not clean_file:
            dropped_rows.append({"episode_uid": ep_dict.get("episode_uid", ""), "drop_reason": "clean_vehicle_missing"})
            continue
        anchor_s = safe_float(ep_dict.get("t_condition_anchor"), default=float("nan"))
        if not math.isfinite(anchor_s):
            dropped_rows.append({"episode_uid": ep_dict.get("episode_uid", ""), "drop_reason": "missing_anchor"})
            continue
        anchor_idx = int(round(anchor_s * FS_OLD))
        if anchor_idx < 0:
            dropped_rows.append({"episode_uid": ep_dict.get("episode_uid", ""), "drop_reason": "negative_anchor_idx"})
            continue
        subj = str(ep_dict.get("subject", "unknown"))
        split = subject_split(subj)
        sample_key = (
            f"v05_server_aligned::{subj}::{ep_dict.get('session_stamp', 'unknown')}::"
            f"{event_idx:05d}::{ep_dict.get('episode_uid', event_idx)}"
        )
        rows.append(
            {
                "protocol_version": "v05_server_aligned_subject_oldflow_fair09",
                "sample_key": sample_key,
                "pool": "v05_primary_secondary_review",
                "subj": subj,
                "split": split,
                "file": Path(clean_file).name,
                "recording_id": str(ep_dict.get("session_stamp", Path(clean_file).stem)),
                "vehicle_file": clean_file,
                "event_idx": int(event_idx),
                "episode_id": int(event_idx),
                "source_event_version": "server_aligned_extreme_condition_episodes_v0_5",
                "phase_type": str(ep_dict.get("v05_source_group", "")),
                "event_level": str(ep_dict.get("condition_level", "")),
                "trigger_type": str(ep_dict.get("condition_context_cn", "")),
                "event_type": str(ep_dict.get("condition_context_cn", "")),
                "road_type_anchor": str(ep_dict.get("condition_context_cn", "")),
                "is_curve": int(bool(ep_dict.get("is_curve_context", False))),
                "curvature_anchor": safe_float(ep_dict.get("peak_abs_curvature_window"), default=0.0),
                "anchor_source": "v05_server_aligned_condition_anchor",
                "anchor_idx": int(anchor_idx),
                "anchor_s": float(anchor_s),
                "history_len": OLD_HISTORY_LEN,
                "future_len": OLD_FUTURE_LEN,
                "valid_future_len": OLD_FUTURE_LEN,
                "valid_future_s": 2.0,
                "full_future_2s": "True",
                "history_full_3s": "unknown_until_old_loader",
                "time_left_after_anchor_s": safe_float(ep_dict.get("time_end_s"), default=np.nan) - anchor_s,
                "keep_for_training": "True",
                "usable_sample": "True",
                "drop_reason": "",
                "event_start_s": safe_float(ep_dict.get("t_condition_anchor"), default=anchor_s),
                "event_end_s": safe_float(ep_dict.get("t_condition_end"), default=anchor_s),
                "event_duration_s": safe_float(ep_dict.get("condition_duration_s"), default=0.0),
                "start_idx": int(round(safe_float(ep_dict.get("t_condition_anchor"), default=anchor_s) * FS_OLD)),
                "end_idx": int(round(safe_float(ep_dict.get("t_condition_end"), default=anchor_s) * FS_OLD)),
                "trigger_score": safe_float(ep_dict.get("condition_score_peak"), default=0.0),
                "primary_score": safe_float(ep_dict.get("v04_post_vehicle_dyn_score"), default=0.0),
                "mechanism_tag": str(ep_dict.get("v04_label_cn", "")),
                "d3_included": "False",
                "instability_event_uid": str(ep_dict.get("episode_uid", sample_key)),
                "leakage_note": "v0.5 server-aligned vehicle data; subject split; FAIR09 vehicle-only; no style/physio/eeg/teacher.",
                "raw_vehicle_file_before_cleaning": raw_file,
                "v05_source_group": str(ep_dict.get("v05_source_group", "")),
                "v04_label": str(ep_dict.get("v04_label", "")),
                "v04_label_cn": str(ep_dict.get("v04_label_cn", "")),
                "v04_reason_cn": str(ep_dict.get("v04_reason_cn", "")),
                "v04_post_vehicle_dyn_score": ep_dict.get("v04_post_vehicle_dyn_score", np.nan),
                "v04_post_steer_delta": ep_dict.get("v04_post_steer_delta", np.nan),
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(MANIFEST_PATH, index=False, encoding="utf-8-sig")
    pd.DataFrame(dropped_rows).to_csv(TABLE_DIR / "oldflow_manifest_build_dropped_v0_5.csv", index=False, encoding="utf-8-sig")
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
            "subject_counts_after_old_loader": meta_df["subj"].astype(str).value_counts().to_dict()
            if "subj" in meta_df.columns
            else {},
            "source_group_counts_after_old_loader": meta_df.get("v05_source_group", pd.Series(dtype=str)).astype(str).value_counts().to_dict(),
        }
    except Exception as exc:
        summary = {"status": "error", "error": repr(exc)}
    VERIFY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def oldflow_fair09_args(manifest: Path, device: str) -> argparse.Namespace:
    args = build_args("vehicle_direct_coarse_fine")
    args.manifest = str(manifest)
    args.run_prefix = "V05_SERVER_ALIGNED_SUBJECT_FAIR09_vehicle_only_seed2026"
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
    refilter_df: pd.DataFrame,
    manifest: pd.DataFrame,
    verify_summary: dict[str, Any],
    train_result: dict[str, Any] | None,
    plot_result: dict[str, Any] | None,
    device: str,
) -> None:
    use_counts = refilter_df["v04_recommended_use"].value_counts().to_dict() if not refilter_df.empty else {}
    source_counts = manifest["v05_source_group"].value_counts().to_dict() if not manifest.empty else {}
    split_counts = manifest["split"].value_counts().to_dict() if not manifest.empty else {}
    if train_result is None:
        metrics_text = "- 尚未训练，仅完成筛选与 manifest 准备。"
        run_root = ""
        plot_text = ""
    else:
        test = train_result["test_metrics"]
        selection = test.get("selection_summary", {})
        run_root = str(train_result["run_root"])
        metrics_text = (
            f"- test steer RMSE：{float(test['steer_rmse']):.6f}\n"
            f"- primary RMSE：{float(selection.get('overall_primary_steer_rmse', np.nan)):.6f}\n"
            f"- tail RMSE：{float(selection.get('rmse_tail_abs_steer', np.nan)):.6f}\n"
            f"- selection：{float(selection.get('selection_score', np.nan)):.6f}\n"
            f"- best epoch：{train_result.get('run_summary', {}).get('best_epoch', 'unknown')}"
        )
        if plot_result:
            plot_text = (
                f"- 预测总览图：`{plot_result.get('overview_path', '')}`\n"
                f"- 预测图目录：`{plot_result.get('figures_dir', '')}`\n"
                f"- 逐样本指标：`{Path(str(plot_result.get('figures_dir', ''))) / 'prediction_sample_metrics.csv'}`"
            )
        else:
            plot_text = ""

    report = f"""# v0.5 服务器处理后车辆数据重筛 + 被试划分旧流程车辆-only

生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}

## 这次做了什么

本轮不再直接使用本地原始车辆 CSV 的 v0.4 筛选结果，而是在服务器现有的已对齐/清洗车辆数据上重新按之前的规则筛选。

服务器车辆数据入口：

`/root/autodl-tmp/data_process/01_datasets/多模态数据/被试数据集合/被试/vehicle/*_vehicle_aligned_cleaned.csv`

筛选规则保持和 v0.4 一致：

- 锚点后车辆动态仍明显变化：保留；
- 锚点后车辆有变化但驾驶员操作弱：保留；
- 车和驾驶员都有弱变化：次级保留；
- 快速打方向但车辆变化弱：待复核；
- 锚点后车和人都没有明显变化：排除；
- 锚点偏晚、窗口不完整或坐标风险：复核或排除。

## 样本筛选结果

- 初始 episode：{len(refilter_df)}
- 筛选用途统计：{use_counts}
- 进入本次训练的样本范围：primary + secondary + manual_review
- 训练 manifest 行数：{len(manifest)}
- 样本来源：{source_counts}

## 被试划分

本轮采用被试分组划分：

- test 被试：{sorted(TEST_SUBJECTS)}
- val 被试：{sorted(VAL_SUBJECTS)}
- train 被试：其余被试

manifest split：{split_counts}

旧流程 loader 检查：

```json
{json.dumps(verify_summary, ensure_ascii=False, indent=2)}
```

## 模型口径

- 旧流程 `FAIR09 / E1` 车辆-only；
- 车辆数据 + 粗细双头；
- 不加连续驾驶风格；
- 不加生理；
- 不加脑电；
- 不加教师蒸馏；
- seed=2026，epochs=40，batch=64，lr=0.001；
- device=`{device}`。

## 当前结果

{metrics_text}

## 预测图

{plot_text}

## 产物位置

- v0.5 筛选总表：`{V05_ALL}`
- v0.5 主训练表：`{V05_PRIMARY}`
- v0.5 次级训练表：`{V05_SECONDARY}`
- v0.5 待复核表：`{V05_REVIEW}`
- 旧流程 manifest：`{MANIFEST_PATH}`
- manifest 检查：`{VERIFY_PATH}`
- 运行记录：`{RUN_RECORD_PATH}`
- 运行目录：`{run_root}`

## 怎么理解

这次和上一次最大的区别是：数据入口换成服务器上的处理后车辆 CSV，切分换成被试分组。这个结果更严格，但也更难；如果指标明显变差，不一定是筛选规则错，也可能是跨被试泛化难度上升。
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def write_run_record(train_result: dict[str, Any] | None, plot_result: dict[str, Any] | None, device: str) -> None:
    if train_result is None:
        return
    test = train_result["test_metrics"]
    selection = test.get("selection_summary", {})
    row = {
        "run_id": RUN_ID,
        "manifest": str(MANIFEST_PATH),
        "run_root": str(train_result["run_root"]),
        "device": device,
        "model": "FAIR09/E1 vehicle_direct_coarse_fine",
        "split_mode": "subject_fixed",
        "test_subjects": ",".join(sorted(TEST_SUBJECTS)),
        "val_subjects": ",".join(sorted(VAL_SUBJECTS)),
        "seed": 2026,
        "epochs": 40,
        "batch_size": 64,
        "lr": 0.001,
        "test_steer_rmse": float(test["steer_rmse"]),
        "primary_rmse": float(selection.get("overall_primary_steer_rmse", np.nan)),
        "tail_rmse": float(selection.get("rmse_tail_abs_steer", np.nan)),
        "selection": float(selection.get("selection_score", np.nan)),
        "report": str(REPORT_PATH),
        "overview": str(plot_result.get("overview_path", "")) if plot_result else "",
    }
    pd.DataFrame([row]).to_csv(RUN_RECORD_PATH, index=False, encoding="utf-8-sig")


def load_existing_train_result(run_root: Path) -> dict[str, Any]:
    run_root = run_root.resolve()
    metrics_path = run_root / "metrics.json"
    summary_path = run_root / "run_summary.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"missing metrics.json: {metrics_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"missing run_summary.json: {summary_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "run_root": str(run_root),
        "run_summary": summary,
        "test_metrics": metrics["test"],
        "val_metrics": metrics.get("val", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-refilter", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--postprocess-run-root", default="")
    args = parser.parse_args()

    ensure_dirs()
    if args.skip_refilter and V05_ALL.exists():
        refilter_df = pd.read_csv(V05_ALL, encoding="utf-8-sig", low_memory=False)
    else:
        refilter_df = build_server_aligned_refilter_tables()

    if args.postprocess_run_root:
        if not MANIFEST_PATH.exists():
            raise FileNotFoundError(f"missing manifest: {MANIFEST_PATH}")
        manifest = pd.read_csv(MANIFEST_PATH, encoding="utf-8-sig", low_memory=False)
        if VERIFY_PATH.exists():
            verify_summary = json.loads(VERIFY_PATH.read_text(encoding="utf-8"))
        else:
            verify_summary = verify_manifest(MANIFEST_PATH)
        train_result = load_existing_train_result(Path(args.postprocess_run_root))
        plot_result = save_prediction_plots_for_run(
            run_root=Path(str(train_result["run_root"])),
            split="test",
            case_file=CASE_FILE,
            max_cases=12,
            device=args.device,
            force_rebuild_cases=True,
        )
        write_run_record(train_result, plot_result, str(args.device))
        write_report(refilter_df, manifest, verify_summary, train_result, plot_result, str(args.device))
        print(
            json.dumps(
                {
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
        return

    episodes = read_v05_inputs()
    clean_status = clean_vehicle_files(episodes)
    manifest = build_manifest(episodes, clean_status)
    verify_summary = verify_manifest(MANIFEST_PATH)
    if verify_summary.get("status") != "ok":
        raise RuntimeError(f"manifest verify failed: {verify_summary}")

    if args.prepare_only:
        write_report(refilter_df, manifest, verify_summary, None, None, str(args.device))
        print(json.dumps({"verify": verify_summary, "report": str(REPORT_PATH)}, ensure_ascii=False, indent=2))
        return

    run_args = oldflow_fair09_args(MANIFEST_PATH, str(args.device))
    train_result = train_one_run(run_args)
    plot_result = save_prediction_plots_for_run(
        run_root=Path(str(train_result["run_root"])),
        split="test",
        case_file=CASE_FILE,
        max_cases=12,
        device=args.device,
        force_rebuild_cases=True,
    )
    write_run_record(train_result, plot_result, str(args.device))
    write_report(refilter_df, manifest, verify_summary, train_result, plot_result, str(args.device))
    print(
        json.dumps(
            {
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
