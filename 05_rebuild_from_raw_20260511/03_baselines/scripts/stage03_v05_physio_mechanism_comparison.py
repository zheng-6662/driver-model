# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


_CWD = Path.cwd()
_DEFAULT_PROJECT_ROOT = _CWD if (_CWD / "05_rebuild_from_raw_20260511").exists() else Path(r"F:/data_set_process/data_process")
PROJECT_ROOT = Path(os.environ.get("DATA_PROCESS_ROOT", str(_DEFAULT_PROJECT_ROOT)))
REBUILD_ROOT = Path(os.environ.get("REBUILD_ROOT", str(PROJECT_ROOT / "05_rebuild_from_raw_20260511")))
OLD_TRAIN_DIR = PROJECT_ROOT / "02_code" / "final_code" / "model" / "training"
OLD_FAIR_DIR = OLD_TRAIN_DIR / "fair_vehicle_event_comparison_20260427"
SCRIPT_DIR = REBUILD_ROOT / "03_baselines" / "scripts"

for path in [SCRIPT_DIR, OLD_FAIR_DIR, OLD_TRAIN_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import stage03_v05_server_aligned_subject_oldflow_fair09 as v05_base  # noqa: E402
from common_compare_runner import build_args  # noqa: E402
from prediction_plotting import save_prediction_plots_for_run  # noqa: E402
from run_event_conditioned_trajectory_baseline import (  # noqa: E402
    apply_optional_context_augmentation,
    build_sample_bundle_from_manifest,
    train_one_run,
)


RUN_ID = "stage03_v05_physio_mechanism_comparison"
OUT_DIR = REBUILD_ROOT / "03_baselines" / RUN_ID
TABLE_DIR = OUT_DIR / "tables"
LOG_DIR = OUT_DIR / "logs"
REPORT_DIR = REBUILD_ROOT / "09_reports"
NOTES_DIR = REBUILD_ROOT / "00_project_notes"
DAILY_LOG = NOTES_DIR / "daily_logs" / f"{time.strftime('%Y-%m-%d')}.md"
REPORT_PATH = REPORT_DIR / "stage03_v05_physio_mechanism_comparison_user_summary_cn.md"
REGISTRY_PATH = TABLE_DIR / "v05_physio_experiment_registry.csv"
STATUS_PATH = TABLE_DIR / "v05_physio_run_status.csv"
AVAILABILITY_PATH = TABLE_DIR / "v05_physio_availability_check.csv"
COMPARISON_PATH = TABLE_DIR / "v05_physio_comparison_table.csv"
SUBJECT_TABLE_PATH = TABLE_DIR / "v05_physio_subject_metrics.csv"
MECHANISM_TABLE_PATH = TABLE_DIR / "v05_physio_mechanism_table.csv"
SERVER_COMMANDS_PATH = OUT_DIR / "launch_commands_server_no_password.sh"
CASE_FILE = v05_base.CASE_FILE
MANIFEST_PATH = v05_base.MANIFEST_PATH

COMMON_ARGS: dict[str, Any] = {
    "seed": 2026,
    "device": "cuda",
    "epochs": 40,
    "min_epochs": 40,
    "patience": 99,
    "batch_size": 64,
    "lr": 1e-3,
    "weight_decay": 0.0,
    "grad_clip": 1.0,
    "selection_mode": "legacy_rmse",
    "teacher_forcing_ratio": 0.0,
    "event_loss_weight": 0.0,
    "trajectory_loss_weight": 1.0,
}

STYLE_VECTOR_PATH = PROJECT_ROOT / "04_project_logs" / "reports" / "style_probe_artifacts" / "driver_style_vectors.csv"


@dataclass(frozen=True)
class ExperimentSpec:
    exp_id: str
    label_cn: str
    purpose_cn: str
    group_cn: str
    style: bool = False
    teacher_state_mode: str = ""
    teacher_state_dim: int = 0
    response_aux: bool = False
    distill_from: str = ""
    student_teacher_state_mode: str = ""
    student_teacher_state_dim: int = 0
    mechanism_hint_cn: str = ""
    extra_args: dict[str, Any] = field(default_factory=dict)


def ensure_dirs() -> None:
    for path in [OUT_DIR, TABLE_DIR, LOG_DIR, REPORT_DIR, NOTES_DIR, NOTES_DIR / "daily_logs"]:
        path.mkdir(parents=True, exist_ok=True)


def experiment_registry() -> dict[str, ExperimentSpec]:
    specs = [
        ExperimentSpec("B0", "车辆-only 粗细双头", "已完成的主基准，不重复训练。", "基准", mechanism_hint_cn="车辆历史和道路/事件上下文基准"),
        ExperimentSpec("B1", "车辆 + 连续风格", "验证连续驾驶风格在 v0.5 新样本上是否仍有增量。", "基准", style=True, mechanism_hint_cn="个体长期驾驶差异"),
        ExperimentSpec("S1", "车辆 + 心率", "看心率是否单独提供增量。", "单信号", teacher_state_mode="raw_hr_only", teacher_state_dim=1, mechanism_hint_cn="压力、唤醒、负荷的慢变化线索"),
        ExperimentSpec("S2", "车辆 + 皮电", "看皮电是否单独提供增量。", "单信号", teacher_state_mode="raw_eda_only", teacher_state_dim=2, mechanism_hint_cn="紧张、唤醒、认知负荷的慢变化线索"),
        ExperimentSpec("S3", "车辆 + 肌电", "看肌电是否单独提供增量。", "单信号", teacher_state_mode="raw_emg_only", teacher_state_dim=1, mechanism_hint_cn="动作准备、操纵意图、方向盘操作相关线索"),
        ExperimentSpec("S4", "车辆 + 脑电", "看脑电直接输入是否单独有效。", "单信号", teacher_state_mode="raw_eeg_only", teacher_state_dim=8, mechanism_hint_cn="训练或推理中的神经状态线索"),
        ExperimentSpec("SF1", "车辆 + 连续风格 + 心率", "看心率是否有连续风格之外的增量。", "单信号+风格", style=True, teacher_state_mode="raw_hr_only", teacher_state_dim=1, mechanism_hint_cn="风格之外的唤醒/压力线索"),
        ExperimentSpec("SF2", "车辆 + 连续风格 + 皮电", "看皮电是否有连续风格之外的增量。", "单信号+风格", style=True, teacher_state_mode="raw_eda_only", teacher_state_dim=2, mechanism_hint_cn="风格之外的慢变化紧张线索"),
        ExperimentSpec("SF3", "车辆 + 连续风格 + 肌电", "看肌电是否有连续风格之外的增量。", "单信号+风格", style=True, teacher_state_mode="raw_emg_only", teacher_state_dim=1, mechanism_hint_cn="风格之外的动作准备或操纵意图"),
        ExperimentSpec("SF4", "车辆 + 连续风格 + 脑电", "看脑电直接输入是否有连续风格之外的增量，并作为 T1 教师。", "单信号+风格", style=True, teacher_state_mode="raw_eeg_only", teacher_state_dim=8, mechanism_hint_cn="脑电直接输入及 EEG 教师候选"),
        ExperimentSpec("C1", "车辆 + 心率 + 皮电 + 肌电", "验证非脑电生理组合是否有效。", "组合", teacher_state_mode="raw_physio_no_eeg", teacher_state_dim=4, mechanism_hint_cn="非脑电生理整体状态"),
        ExperimentSpec("C2", "车辆 + 连续风格 + 心率 + 皮电 + 肌电", "验证非脑电生理是否有连续风格之外的增量，并作为 T2 教师。", "组合", style=True, teacher_state_mode="raw_physio_no_eeg", teacher_state_dim=4, mechanism_hint_cn="风格之外的非脑电状态监督"),
        ExperimentSpec("C3", "车辆 + 心率 + 皮电 + 肌电 + 脑电", "验证全生理直接输入是否有效。", "组合", teacher_state_mode="raw_physio", teacher_state_dim=12, mechanism_hint_cn="全生理直接融合"),
        ExperimentSpec("C4", "车辆 + 连续风格 + 全生理", "验证连续风格 + 全生理直接输入是否最强，并作为 T3/T4 教师。", "组合", style=True, teacher_state_mode="raw_physio", teacher_state_dim=12, mechanism_hint_cn="全生理直接融合和教师候选"),
        ExperimentSpec("A1", "车辆 + 连续风格 + 肌电 + 响应类型辅助", "看肌电是否更适合帮助判断响应强弱、方向和形态。", "响应类型辅助", style=True, teacher_state_mode="raw_emg_only", teacher_state_dim=1, response_aux=True, mechanism_hint_cn="肌电帮助响应类型判断"),
        ExperimentSpec("A2", "车辆 + 连续风格 + 非脑电生理 + 响应类型辅助", "看非脑电生理是否更适合作为状态判断而非简单回归拼接。", "响应类型辅助", style=True, teacher_state_mode="raw_physio_no_eeg", teacher_state_dim=4, response_aux=True, mechanism_hint_cn="非脑电生理帮助状态/困难样本判断"),
        ExperimentSpec("A3", "车辆 + 连续风格 + 全生理 + 响应类型辅助", "看全生理是否改善响应类型和困难样本。", "响应类型辅助", style=True, teacher_state_mode="raw_physio", teacher_state_dim=12, response_aux=True, mechanism_hint_cn="全生理帮助响应类型判断"),
        ExperimentSpec("T1", "脑电教师 -> 车辆 + 连续风格学生", "训练时用 SF4 作为脑电教师，推理时不用脑电。", "多教师蒸馏", style=True, distill_from="SF4", mechanism_hint_cn="脑电是否更适合训练期教师"),
        ExperimentSpec("T2", "非脑电生理教师 -> 车辆 + 连续风格学生", "训练时用 C2 作为非脑电生理教师，推理时不用生理。", "多教师蒸馏", style=True, distill_from="C2", mechanism_hint_cn="非脑电生理是否更适合作为状态教师"),
        ExperimentSpec("T3", "全生理教师 -> 车辆 + 连续风格学生", "训练时用 C4 作为全生理教师，推理时不用生理。", "多教师蒸馏", style=True, distill_from="C4", mechanism_hint_cn="全生理是否提供更强训练期软监督"),
        ExperimentSpec("T4", "全生理教师 -> 车辆 + 连续风格 + 肌电学生", "训练时用 C4 作为全生理教师，推理时保留肌电。", "多教师蒸馏", style=True, student_teacher_state_mode="raw_emg_only", student_teacher_state_dim=1, distill_from="C4", mechanism_hint_cn="全生理教师和肌电推理输入是否互补"),
    ]
    return {spec.exp_id: spec for spec in specs}


def _set_arg(args: argparse.Namespace, name: str, value: Any) -> None:
    setattr(args, name, value)


def configure_run_args(spec: ExperimentSpec, seed: int, device: str, teacher_checkpoint: str = "") -> argparse.Namespace:
    args = build_args("vehicle_direct_coarse_fine")
    for key, value in COMMON_ARGS.items():
        _set_arg(args, key, value)
    args.manifest = str(MANIFEST_PATH)
    args.seed = int(seed)
    args.device = str(device)
    args.run_prefix = f"V05P_{spec.exp_id}_{seed}"
    args.enable_teacher_state_context = False
    args.teacher_state_mode = "pca_latent"
    args.teacher_state_dim = 4
    args.enable_driver_style_context = False
    args.driver_style_vector_csv = str(STYLE_VECTOR_PATH)
    args.driver_style_embed_dim = 4
    args.driver_style_include_iqr = True
    args.enable_response_type_head = False
    args.enable_response_type_condition = False
    args.response_type_use_context = False
    args.response_type_loss_weight = 0.0
    args.distill_teacher_checkpoint = ""
    args.distill_weight = 0.0
    args.distill_tail_weight = 0.0
    args.distill_reliability_weighting = False
    args.distill_hardcase_weighting = False

    mode = spec.teacher_state_mode or spec.student_teacher_state_mode
    dim = spec.teacher_state_dim or spec.student_teacher_state_dim
    if mode:
        args.enable_teacher_state_context = True
        args.teacher_state_mode = mode
        args.teacher_state_dim = int(dim)
    if spec.style:
        args.enable_driver_style_context = True
    if spec.response_aux:
        args.enable_response_type_head = True
        args.enable_response_type_condition = True
        args.response_type_use_context = True
        args.response_type_loss_weight = 0.25
    if teacher_checkpoint:
        args.distill_teacher_checkpoint = str(teacher_checkpoint)
        args.distill_weight = 0.20
        args.distill_tail_weight = 0.05
        args.distill_reliability_weighting = True
        args.distill_hardcase_weighting = True
        args.distill_hardcase_extra_weight = 0.50
    for key, value in spec.extra_args.items():
        _set_arg(args, key, value)
    return args


def ensure_v05_manifest() -> dict[str, Any]:
    v05_base.ensure_dirs()
    if MANIFEST_PATH.exists() and v05_base.VERIFY_PATH.exists():
        try:
            cached = json.loads(v05_base.VERIFY_PATH.read_text(encoding="utf-8"))
            if cached.get("status") == "ok":
                return cached
        except Exception:
            pass
    if not MANIFEST_PATH.exists():
        if not v05_base.V05_ALL.exists():
            refilter_df = v05_base.build_server_aligned_refilter_tables()
        else:
            refilter_df = pd.read_csv(v05_base.V05_ALL, encoding="utf-8-sig", low_memory=False)
        episodes = v05_base.read_v05_inputs()
        clean_status = v05_base.clean_vehicle_files(episodes)
        manifest = v05_base.build_manifest(episodes, clean_status)
        verify = v05_base.verify_manifest(MANIFEST_PATH)
        v05_base.write_report(refilter_df, manifest, verify, None, None, "cuda")
        return verify
    return v05_base.verify_manifest(MANIFEST_PATH)


def append_daily_log(text: str) -> None:
    DAILY_LOG.parent.mkdir(parents=True, exist_ok=True)
    with DAILY_LOG.open("a", encoding="utf-8") as handle:
        handle.write("\n\n" + text.strip() + "\n")


def update_project_notes(message: str) -> None:
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    status_path = NOTES_DIR / "PROJECT_STATUS_CN.md"
    queue_path = NOTES_DIR / "TASK_QUEUE_CN.md"
    artifact_path = NOTES_DIR / "ARTIFACT_INDEX_CN.md"
    server_path = NOTES_DIR / "SERVER_RUNS_CN.md"
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    block = f"\n\n## {now} v0.5 生理机制验证\n\n{message.strip()}\n"
    for path, fallback in [
        (status_path, "# 项目状态\n"),
        (queue_path, "# 当前任务队列\n"),
        (artifact_path, "# 产物索引\n"),
        (server_path, "# 服务器运行记录\n\n注意：不记录服务器密码。\n"),
    ]:
        if not path.exists():
            path.write_text(fallback, encoding="utf-8")
        with path.open("a", encoding="utf-8") as handle:
            handle.write(block)


def write_registry() -> pd.DataFrame:
    ensure_dirs()
    rows = []
    for spec in experiment_registry().values():
        rows.append(
            {
                "exp_id": spec.exp_id,
                "label_cn": spec.label_cn,
                "group_cn": spec.group_cn,
                "purpose_cn": spec.purpose_cn,
                "style": spec.style,
                "teacher_state_mode": spec.teacher_state_mode,
                "teacher_state_dim": spec.teacher_state_dim,
                "response_aux": spec.response_aux,
                "distill_from": spec.distill_from,
                "student_teacher_state_mode": spec.student_teacher_state_mode,
                "mechanism_hint_cn": spec.mechanism_hint_cn,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(REGISTRY_PATH, index=False, encoding="utf-8-sig")
    return df


def read_status() -> pd.DataFrame:
    if STATUS_PATH.exists():
        return pd.read_csv(STATUS_PATH, encoding="utf-8-sig", low_memory=False)
    return pd.DataFrame()


def upsert_status(row: dict[str, Any]) -> None:
    existing = read_status()
    rows = existing.to_dict("records") if not existing.empty else []
    key = (str(row.get("exp_id")), int(row.get("seed", 2026)))
    out: list[dict[str, Any]] = []
    replaced = False
    for item in rows:
        item_key = (str(item.get("exp_id")), int(item.get("seed", 2026)))
        if item_key == key:
            merged = {**item, **row}
            out.append(merged)
            replaced = True
        else:
            out.append(item)
    if not replaced:
        out.append(row)
    fieldnames: list[str] = []
    for item in out:
        for col in item:
            if col not in fieldnames:
                fieldnames.append(col)
    with STATUS_PATH.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out)


def split_indices(meta_df: pd.DataFrame) -> list[int]:
    return meta_df.index[meta_df["split"].astype(str).eq("train")].astype(int).tolist()


def availability_check(seed: int = 2026) -> pd.DataFrame:
    verify = ensure_v05_manifest()
    if verify.get("status") != "ok":
        raise RuntimeError(f"v0.5 manifest verify failed: {verify}")
    x_pool, y_pool, curve_pool, ctx_pool, mask_pool, meta_df, dropped = build_sample_bundle_from_manifest(
        manifest_path=MANIFEST_PATH,
        max_train_samples=None,
        max_val_samples=None,
        max_test_samples=None,
        seed=int(seed),
    )
    train_idx = split_indices(meta_df)
    rows: list[dict[str, Any]] = []
    modes = {
        "HR": ("raw_hr_only", 1),
        "EDA": ("raw_eda_only", 2),
        "EMG": ("raw_emg_only", 1),
        "EEG": ("raw_eeg_only", 8),
        "HR+EDA+EMG": ("raw_physio_no_eeg", 4),
        "HR+EDA+EMG+EEG": ("raw_physio", 12),
    }
    for signal, (mode, dim) in modes.items():
        probe = build_args("vehicle_direct_coarse_fine")
        probe.enable_teacher_state_context = True
        probe.teacher_state_mode = mode
        probe.teacher_state_dim = dim
        probe.enable_driver_style_context = False
        try:
            _, meta = apply_optional_context_augmentation(ctx_pool, meta_df, train_idx, probe, run_root=None)
            aug = meta["augmentations"][0]
            missing_stats = aug.get("base_missing_stats", [])
            component_names = aug.get("component_names", [])
            relevant_names = {
                "raw_hr_only": ["hr"],
                "raw_eda_only": ["eda_tonic", "eda_phasic"],
                "raw_emg_only": ["emg_rms"],
                "raw_eeg_only": [
                    "alpha_asym",
                    "occ_ta_beta",
                    "frontal_ta_beta",
                    "temporal_ta_beta",
                    "occ_alpha_abs",
                    "temporal_gamma_rel",
                    "occ_gamma_rel",
                    "frontal_gamma_rel",
                ],
                "raw_physio_no_eeg": ["hr", "eda_tonic", "eda_phasic", "emg_rms"],
                "raw_physio": [
                    "hr",
                    "eda_tonic",
                    "eda_phasic",
                    "emg_rms",
                    "alpha_asym",
                    "occ_ta_beta",
                    "frontal_ta_beta",
                    "temporal_ta_beta",
                    "occ_alpha_abs",
                    "temporal_gamma_rel",
                    "occ_gamma_rel",
                    "frontal_gamma_rel",
                ],
            }[mode]
            relevant_stats = [item for item in missing_stats if str(item.get("name", "")) in relevant_names]
            relevant_all_missing_names = [str(item.get("name", "")) for item in relevant_stats if bool(item.get("all_missing"))]
            valid_ratios = [float(item.get("valid_ratio", 0.0)) for item in relevant_stats]
            min_valid_ratio = min(valid_ratios) if valid_ratios else float("nan")
            status = "ok" if not relevant_all_missing_names else "has_relevant_all_missing"
            rows.append(
                {
                    "signal": signal,
                    "mode": mode,
                    "status": status,
                    "kept_samples": int(len(meta_df)),
                    "dropped_by_old_loader": int(dropped),
                    "train_samples": int((meta_df["split"].astype(str) == "train").sum()),
                    "val_samples": int((meta_df["split"].astype(str) == "val").sum()),
                    "test_samples": int((meta_df["split"].astype(str) == "test").sum()),
                    "context_dim_after_append": int(meta.get("final_context_dim", -1)),
                    "component_names": "|".join(map(str, component_names)),
                    "all_missing_names": "|".join(map(str, relevant_all_missing_names)),
                    "min_component_valid_ratio_train": min_valid_ratio,
                    "future_window_risk_note": "使用旧流程按锚点前窗口构造的生理上下文；本轮作为输入窗口生理特征验证，不使用标签窗口统计。",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "signal": signal,
                    "mode": mode,
                    "status": "error",
                    "error": repr(exc),
                    "kept_samples": int(len(meta_df)),
                    "dropped_by_old_loader": int(dropped),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(AVAILABILITY_PATH, index=False, encoding="utf-8-sig")
    return df


def find_run_root_for(exp_id: str, seed: int) -> str:
    status = read_status()
    if status.empty:
        return ""
    rows = status[
        status["exp_id"].astype(str).eq(str(exp_id))
        & status["seed"].astype(str).eq(str(seed))
        & status.get("status", pd.Series(dtype=str)).astype(str).eq("completed")
    ]
    if rows.empty:
        return ""
    root = str(rows.iloc[-1].get("run_root", ""))
    if root and (Path(root) / "best_model.pt").exists():
        return root
    return root


def existing_b0_row(seed: int) -> dict[str, Any] | None:
    candidates = sorted(
        (PROJECT_ROOT / "tmp" / "event_conditioned_runs").glob(f"V05_SERVER_ALIGNED_SUBJECT_FAIR09_vehicle_only_seed{seed}_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for root in candidates:
        if (root / "metrics.json").exists():
            return {
                "exp_id": "B0",
                "seed": int(seed),
                "status": "completed",
                "run_root": str(root),
                "completed_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(root.stat().st_mtime)),
                "label_cn": "车辆-only 粗细双头",
                "group_cn": "基准",
                "note": "既有 B0 基准，不重复训练。",
            }
    return None


def run_one(exp_id: str, seed: int, device: str, no_plots: bool = False) -> dict[str, Any]:
    ensure_dirs()
    specs = experiment_registry()
    if exp_id not in specs:
        raise KeyError(f"unknown exp_id={exp_id}")
    spec = specs[exp_id]
    verify = ensure_v05_manifest()
    if verify.get("status") != "ok":
        raise RuntimeError(f"v0.5 manifest verify failed: {verify}")
    if exp_id == "B0":
        row = existing_b0_row(seed)
        if row is None:
            raise FileNotFoundError("B0 existing baseline was not found; rerun the v0.5 vehicle-only baseline first.")
        upsert_status(row)
        return row

    teacher_ckpt = ""
    if spec.distill_from:
        teacher_root = find_run_root_for(spec.distill_from, seed)
        if not teacher_root:
            raise FileNotFoundError(f"{exp_id} needs completed teacher {spec.distill_from} seed={seed}")
        ckpt = Path(teacher_root) / "best_model.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"teacher checkpoint missing: {ckpt}")
        teacher_ckpt = str(ckpt)

    args = configure_run_args(spec, seed=seed, device=device, teacher_checkpoint=teacher_ckpt)
    start = time.strftime("%Y-%m-%d %H:%M:%S")
    upsert_status(
        {
            "exp_id": exp_id,
            "seed": int(seed),
            "label_cn": spec.label_cn,
            "group_cn": spec.group_cn,
            "status": "running",
            "started_at": start,
            "device": device,
            "manifest": str(MANIFEST_PATH),
            "run_prefix": args.run_prefix,
            "teacher_from": spec.distill_from,
            "teacher_checkpoint": teacher_ckpt,
        }
    )
    try:
        result = train_one_run(args)
        run_root = Path(str(result["run_root"]))
        plot_result: dict[str, Any] = {}
        if not no_plots:
            plot_result = save_prediction_plots_for_run(
                run_root=run_root,
                split="test",
                case_file=CASE_FILE,
                max_cases=12,
                batch_size=int(args.batch_size),
                device=device,
                force_rebuild_cases=False,
                save_sequences=True,
            )
        row = {
            "exp_id": exp_id,
            "seed": int(seed),
            "label_cn": spec.label_cn,
            "group_cn": spec.group_cn,
            "status": "completed",
            "started_at": start,
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": device,
            "run_root": str(run_root),
            "teacher_from": spec.distill_from,
            "teacher_checkpoint": teacher_ckpt,
            "prediction_overview": str(plot_result.get("overview_path", "")),
            "prediction_figures_dir": str(plot_result.get("figures_dir", "")),
        }
        upsert_status(row)
        return row
    except Exception as exc:
        row = {
            "exp_id": exp_id,
            "seed": int(seed),
            "label_cn": spec.label_cn,
            "group_cn": spec.group_cn,
            "status": "failed",
            "started_at": start,
            "failed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": device,
            "teacher_from": spec.distill_from,
            "teacher_checkpoint": teacher_ckpt,
            "error": repr(exc),
        }
        upsert_status(row)
        raise


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def physical_metrics_from_sequences(fig_dir: Path) -> dict[str, Any]:
    seq_path = fig_dir / "prediction_sequences.npz"
    if not seq_path.exists():
        return {}
    data = np.load(seq_path, allow_pickle=False)
    pred = np.asarray(data["pred"], dtype=np.float32)
    true = np.asarray(data["true"], dtype=np.float32)
    mask = np.asarray(data["mask"], dtype=np.float32)
    if pred.ndim != 3 or true.ndim != 3 or pred.shape[-1] < 1:
        return {}
    pred_s = pred[:, :, 0]
    true_s = true[:, :, 0]
    valid = mask > 0.5
    n = pred_s.shape[0]
    rows: list[dict[str, Any]] = []
    for i in range(n):
        idx = np.where(valid[i])[0]
        if idx.size == 0:
            continue
        t = true_s[i, idx]
        p = pred_s[i, idx]
        abs_t = np.abs(t)
        peak_local = int(np.argmax(abs_t))
        true_peak = float(t[peak_local])
        pred_at_true_peak = float(p[peak_local])
        pred_peak_abs = float(np.max(np.abs(p)))
        true_peak_abs = float(abs(true_peak))
        large = true_peak_abs >= 0.30
        direction_ok = np.sign(true_peak) == np.sign(pred_at_true_peak) if true_peak_abs >= 0.10 else True
        amp_ratio = pred_peak_abs / max(true_peak_abs, 1e-6)
        rows.append(
            {
                "large": bool(large),
                "direction_ok": bool(direction_ok),
                "wrong_side_large": bool(large and not direction_ok),
                "severe_under_large": bool(large and amp_ratio < 0.50),
                "large_recalled": bool(large and amp_ratio >= 0.50),
                "amp_ratio": float(amp_ratio),
            }
        )
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    large_df = df[df["large"]]
    out = {
        "physical_sample_count": int(len(df)),
        "large_response_count": int(len(large_df)),
        "direction_match_rate": float(df["direction_ok"].mean()),
        "mean_amp_ratio_pred_over_true": float(df["amp_ratio"].replace([np.inf, -np.inf], np.nan).mean()),
    }
    if not large_df.empty:
        out.update(
            {
                "large_wrong_side_rate": float(large_df["wrong_side_large"].mean()),
                "large_severe_under_rate": float(large_df["severe_under_large"].mean()),
                "large_response_recall": float(large_df["large_recalled"].mean()),
                "large_mean_amp_ratio": float(large_df["amp_ratio"].replace([np.inf, -np.inf], np.nan).mean()),
            }
        )
    return out


def summarize_run(row: dict[str, Any], spec: ExperimentSpec, fallback: dict[str, Any] | None = None) -> dict[str, Any]:
    run_root = Path(str(row.get("run_root", "")))
    metrics_path = run_root / "metrics.json"
    summary_path = run_root / "run_summary.json"
    if not metrics_path.exists():
        if fallback:
            preserved = dict(fallback)
            preserved["exp_id"] = spec.exp_id
            preserved["label_cn"] = spec.label_cn
            preserved["group_cn"] = spec.group_cn
            preserved["status"] = preserved.get("status") or row.get("status", "completed")
            preserved["run_root"] = preserved.get("run_root") or str(run_root)
            return preserved
        return {"exp_id": spec.exp_id, "seed": row.get("seed", 2026), "status": row.get("status", "missing_metrics"), "run_root": str(run_root)}
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    run_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    test = metrics.get("test", {})
    sel = test.get("selection_summary", {})
    fig_dir = run_root / "prediction_figures" / "test"
    physical = physical_metrics_from_sequences(fig_dir)
    out = {
        "exp_id": spec.exp_id,
        "label_cn": spec.label_cn,
        "group_cn": spec.group_cn,
        "seed": int(row.get("seed", 2026)),
        "status": row.get("status", "completed"),
        "run_root": str(run_root),
        "test_steer_rmse": finite_float(test.get("steer_rmse")),
        "primary_rmse": finite_float(sel.get("overall_primary_steer_rmse")),
        "tail_rmse": finite_float(sel.get("rmse_tail_abs_steer")),
        "selection": finite_float(sel.get("selection_score")),
        "tail_direction_match": finite_float(sel.get("tail_direction_match")),
        "peak_time_abs_err_s": finite_float(sel.get("peak_time_abs_err_s")),
        "best_epoch": run_summary.get("best_epoch", ""),
        "style": spec.style,
        "teacher_state_mode": spec.teacher_state_mode or spec.student_teacher_state_mode,
        "response_aux": spec.response_aux,
        "distill_from": spec.distill_from,
        "mechanism_hint_cn": spec.mechanism_hint_cn,
        **physical,
    }
    return out


def summarize_all(write_report_flag: bool = True) -> pd.DataFrame:
    ensure_dirs()
    specs = experiment_registry()
    status = read_status()
    existing_comp = pd.read_csv(COMPARISON_PATH, encoding="utf-8-sig", low_memory=False) if COMPARISON_PATH.exists() else pd.DataFrame()

    def fallback_for(exp_id: str, seed: Any) -> dict[str, Any] | None:
        if existing_comp.empty or "exp_id" not in existing_comp.columns:
            return None
        mask = existing_comp["exp_id"].astype(str).eq(str(exp_id))
        if "seed" in existing_comp.columns:
            mask = mask & existing_comp["seed"].astype(str).eq(str(seed))
        found = existing_comp[mask]
        if found.empty:
            return None
        return found.iloc[-1].to_dict()

    rows: list[dict[str, Any]] = []
    b0 = existing_b0_row(2026)
    if b0 is not None:
        rows.append(summarize_run(b0, specs["B0"], fallback_for("B0", 2026)))
    if not status.empty:
        for row in status.to_dict("records"):
            exp_id = str(row.get("exp_id", ""))
            if exp_id in specs and str(row.get("status", "")) == "completed" and exp_id != "B0":
                rows.append(summarize_run(row, specs[exp_id], fallback_for(exp_id, row.get("seed", 2026))))
    comp = pd.DataFrame(rows)
    if not comp.empty:
        comp = comp.drop_duplicates(["exp_id", "seed"], keep="last").sort_values(["seed", "group_cn", "exp_id"]).reset_index(drop=True)
        comp.to_csv(COMPARISON_PATH, index=False, encoding="utf-8-sig")
        write_subject_tables(comp)
        write_mechanism_table(comp)
    if write_report_flag:
        write_summary_report(comp)
    return comp


def write_subject_tables(comp: pd.DataFrame) -> None:
    rows: list[dict[str, Any]] = []
    specs = experiment_registry()
    for item in comp.to_dict("records"):
        fig_dir = Path(str(item.get("run_root", ""))) / "prediction_figures" / "test"
        sample_path = fig_dir / "prediction_sample_metrics.csv"
        if not sample_path.exists():
            continue
        df = pd.read_csv(sample_path, encoding="utf-8-sig", low_memory=False)
        if "subj" not in df.columns:
            continue
        for subj, g in df.groupby("subj"):
            rows.append(
                {
                    "exp_id": item["exp_id"],
                    "label_cn": specs[str(item["exp_id"])].label_cn if str(item["exp_id"]) in specs else item["exp_id"],
                    "seed": item.get("seed", 2026),
                    "subj": subj,
                    "n": int(len(g)),
                    "rmse_2s_abs_steer": float(pd.to_numeric(g.get("rmse_2s_abs_steer"), errors="coerce").mean()),
                    "primary_rmse": float(pd.to_numeric(g.get("rmse_pre_tail_abs_steer"), errors="coerce").mean()),
                    "tail_rmse": float(pd.to_numeric(g.get("rmse_tail_abs_steer"), errors="coerce").mean()),
                    "direction_match": float(pd.to_numeric(g.get("direction_match"), errors="coerce").mean()),
                    "tail_direction_match": float(pd.to_numeric(g.get("tail_direction_match"), errors="coerce").mean()),
                }
            )
    if rows:
        pd.DataFrame(rows).to_csv(SUBJECT_TABLE_PATH, index=False, encoding="utf-8-sig")


def write_mechanism_table(comp: pd.DataFrame) -> None:
    if comp.empty:
        return
    b0 = comp[comp["exp_id"].eq("B0")]
    b1 = comp[comp["exp_id"].eq("B1")]
    ref_b0 = b0.iloc[0].to_dict() if not b0.empty else {}
    ref_b1 = b1.iloc[0].to_dict() if not b1.empty else ref_b0
    rows = []
    for row in comp.to_dict("records"):
        rows.append(
            {
                "exp_id": row.get("exp_id"),
                "label_cn": row.get("label_cn"),
                "group_cn": row.get("group_cn"),
                "mechanism_hint_cn": row.get("mechanism_hint_cn"),
                "delta_rmse_vs_B0": finite_float(row.get("test_steer_rmse")) - finite_float(ref_b0.get("test_steer_rmse")) if ref_b0 else float("nan"),
                "delta_rmse_vs_B1": finite_float(row.get("test_steer_rmse")) - finite_float(ref_b1.get("test_steer_rmse")) if ref_b1 else float("nan"),
                "delta_tail_vs_B0": finite_float(row.get("tail_rmse")) - finite_float(ref_b0.get("tail_rmse")) if ref_b0 else float("nan"),
                "delta_large_wrong_side_vs_B0": finite_float(row.get("large_wrong_side_rate")) - finite_float(ref_b0.get("large_wrong_side_rate")) if ref_b0 else float("nan"),
                "delta_large_under_vs_B0": finite_float(row.get("large_severe_under_rate")) - finite_float(ref_b0.get("large_severe_under_rate")) if ref_b0 else float("nan"),
                "possible_role_cn": infer_possible_role(row, ref_b0, ref_b1),
            }
        )
    pd.DataFrame(rows).to_csv(MECHANISM_TABLE_PATH, index=False, encoding="utf-8-sig")


def dataframe_to_simple_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "无记录。"
    view = df.head(max_rows).copy()
    return view.to_string(index=False)


def infer_possible_role(row: dict[str, Any], ref_b0: dict[str, Any], ref_b1: dict[str, Any]) -> str:
    if not ref_b0:
        return "等待基准结果"
    d0 = finite_float(row.get("test_steer_rmse")) - finite_float(ref_b0.get("test_steer_rmse"))
    d1 = finite_float(row.get("test_steer_rmse")) - finite_float(ref_b1.get("test_steer_rmse")) if ref_b1 else float("nan")
    du = finite_float(row.get("large_severe_under_rate")) - finite_float(ref_b0.get("large_severe_under_rate"))
    dw = finite_float(row.get("large_wrong_side_rate")) - finite_float(ref_b0.get("large_wrong_side_rate"))
    if d0 < -0.005 and (math.isnan(d1) or d1 < -0.002):
        return "可能改善整体轨迹，并且有风格之外增量"
    if du < -0.03 or dw < -0.03:
        return "整体误差未必最优，但可能改善大响应幅值或方向问题"
    if str(row.get("distill_from", "")):
        return "作为教师路线观察：重点看困难样本、幅值和错侧是否改善"
    if d0 > 0.005:
        return "当前形式可能引入噪声或与任务不匹配"
    return "效果接近基准，需要结合分被试和预测图判断"


def write_summary_report(comp: pd.DataFrame) -> None:
    ensure_dirs()
    verify = json.loads(v05_base.VERIFY_PATH.read_text(encoding="utf-8")) if v05_base.VERIFY_PATH.exists() else ensure_v05_manifest()
    availability = pd.read_csv(AVAILABILITY_PATH, encoding="utf-8-sig") if AVAILABILITY_PATH.exists() else pd.DataFrame()
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    if comp.empty:
        result_text = "当前还没有新的完成版本，只有实验注册和可用性检查。"
    else:
        view_cols = [
            "exp_id",
            "label_cn",
            "test_steer_rmse",
            "primary_rmse",
            "tail_rmse",
            "selection",
            "large_wrong_side_rate",
            "large_severe_under_rate",
            "large_response_recall",
        ]
        result_text = dataframe_to_simple_table(comp[[c for c in view_cols if c in comp.columns]])
    availability_text = dataframe_to_simple_table(availability) if not availability.empty else "尚未生成可用性表。"
    report = f"""# v0.5 新样本集连续风格与生理机制验证

生成时间：{now}

## 本轮要回答什么

本轮不是再跑一个零散对照，而是固定 v0.5 新样本、固定被试划分、固定旧流程粗细双头结构，系统判断：

1. 连续驾驶风格是否仍然有效；
2. 心率、皮电、肌电、脑电单独输入是否有增量；
3. 非脑电生理组合、全生理组合是否优于单信号；
4. 生理信号更适合直接输入、响应类型辅助，还是作为训练期教师；
5. 改善是否体现在整体误差、方向、幅值、尾段、困难样本、分被试或分场景上。

## 固定数据和训练条件

- manifest：`{MANIFEST_PATH}`
- 旧 loader 检查：

```json
{json.dumps(verify, ensure_ascii=False, indent=2)}
```

- test 被试：cwh / gf / tyy
- val 被试：byx / gzj / yyl
- train 被试：其余被试
- seed：2026
- epochs：40
- batch：64
- lr：0.001
- device：cuda

## 生理数据可用性检查

{availability_text}

## 当前完成结果

{result_text}

## 产物位置

- 实验注册表：`{REGISTRY_PATH}`
- 运行状态表：`{STATUS_PATH}`
- 生理可用性表：`{AVAILABILITY_PATH}`
- 总指标表：`{COMPARISON_PATH}`
- 分被试表：`{SUBJECT_TABLE_PATH}`
- 机制判断表：`{MECHANISM_TABLE_PATH}`
- 服务器启动命令模板：`{SERVER_COMMANDS_PATH}`

## 当前解释边界

- B0 是已完成的车辆-only 主基准，不重复训练。
- B1 用来验证连续风格。
- S/SF/C 版本回答直接输入是否有效。
- A 版本回答生理是否更适合帮助判断响应类型。
- T 版本回答生理/脑电是否更适合作为训练期教师。
- 单个 seed 只用于筛选，不能直接形成最终论文强结论；有希望版本还要补 seed2027/2028。
"""
    REPORT_PATH.write_text(report, encoding="utf-8-sig")
    update_project_notes(
        f"已更新 v0.5 生理机制验证材料。\n\n"
        f"- 报告：`{REPORT_PATH}`\n"
        f"- 运行状态表：`{STATUS_PATH}`\n"
        f"- 对比表：`{COMPARISON_PATH}`\n"
        f"- 机制表：`{MECHANISM_TABLE_PATH}`"
    )


def write_server_commands(experiments: list[str], seed: int, gpus: list[str], no_plots: bool) -> None:
    ensure_dirs()
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "cd /root/autodl-tmp/data_process",
        "export DATA_PROCESS_ROOT=/root/autodl-tmp/data_process",
        "export PYTHONUNBUFFERED=1",
        "export PYTHON_BIN=${PYTHON_BIN:-/root/miniconda3/bin/python}",
        "mkdir -p /root/autodl-tmp/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_v05_physio_mechanism_comparison/logs",
        "",
    ]
    for i, exp_id in enumerate(experiments):
        gpu = gpus[i % max(1, len(gpus))]
        log = f"/root/autodl-tmp/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_v05_physio_mechanism_comparison/logs/{exp_id}_seed{seed}.log"
        no_plot_arg = " --no-plots" if no_plots else ""
        cmd = (
            f"screen -dmS v05p_{exp_id}_{seed} bash -lc "
            f"'cd /root/autodl-tmp/data_process && export DATA_PROCESS_ROOT=/root/autodl-tmp/data_process && "
            f"CUDA_VISIBLE_DEVICES={gpu} ${{PYTHON_BIN:-/root/miniconda3/bin/python}} 05_rebuild_from_raw_20260511/03_baselines/scripts/"
            f"stage03_v05_physio_mechanism_comparison.py --run-one {exp_id} --seed {seed} --device cuda{no_plot_arg} "
            f"2>&1 | tee {log}'"
        )
        lines.append(cmd)
    SERVER_COMMANDS_PATH.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v0.5 连续风格、生理信号和多教师机制验证。")
    parser.add_argument("--write-registry", action="store_true")
    parser.add_argument("--availability", action="store_true")
    parser.add_argument("--run-one", default="")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--write-launcher", nargs="*", default=None)
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    write_registry()
    if args.prepare_only:
        verify = ensure_v05_manifest()
        avail = availability_check(seed=args.seed)
        write_summary_report(pd.DataFrame())
        print(json.dumps({"verify": verify, "availability": str(AVAILABILITY_PATH), "registry": str(REGISTRY_PATH)}, ensure_ascii=False, indent=2))
        return
    if args.availability:
        df = availability_check(seed=args.seed)
        print(df.to_string(index=False))
    if args.write_launcher is not None:
        experiments = args.write_launcher
        if not experiments:
            experiments = ["B1", "S1", "S2", "S3"]
        write_server_commands(experiments, seed=args.seed, gpus=args.gpus, no_plots=args.no_plots)
        print(f"launcher written: {SERVER_COMMANDS_PATH}")
    if args.run_one:
        row = run_one(args.run_one, seed=args.seed, device=args.device, no_plots=args.no_plots)
        print(json.dumps(row, ensure_ascii=False, indent=2))
    if args.summarize:
        comp = summarize_all(write_report_flag=True)
        print(comp.to_string(index=False) if not comp.empty else "no completed runs")
    if not any([args.write_registry, args.availability, args.run_one, args.summarize, args.write_launcher is not None, args.prepare_only]):
        print(f"registry: {REGISTRY_PATH}")
        print(f"status: {STATUS_PATH}")
        print("use --prepare-only, --run-one EXP_ID, --summarize, or --write-launcher EXP_ID ...")


if __name__ == "__main__":
    main()
