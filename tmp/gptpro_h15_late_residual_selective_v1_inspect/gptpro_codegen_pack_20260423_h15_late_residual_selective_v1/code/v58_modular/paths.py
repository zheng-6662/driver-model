# ================================================================
# V6.0 REVERSAL-SAMPLE-WEIGHT VERSION
#
# 目标：
#   不增加新网络头，不恢复 teacher state，不引入额外复杂结构，
#   只通过“样本级加权”提高强反打/符号翻转样本在训练中的重要性。
#
# 核心思想：
#   - 普通样本：weight = 1.0
#   - 未来 steer 出现明显符号翻转（reversal）的样本：weight = REV_SAMPLE_WEIGHT
#
# 本版相对 v5.9 的改动：
#   1. teacher distill 保持关闭（LAMBDA_STATE = 0.0）
#   2. 保持 steer-only amplitude loss
#   3. 新增 reversal sample weighting
#
# 重点观察：
#   - 296 / 445 / 742 这类复杂样本是否改善
#   - 反打样本的 steer 符号、峰值、后段结构是否更合理
#   - 普通样本是否基本不退化
# ================================================================

# -*- coding: utf-8 -*-
"""
v5.4: v5.1 + Driver State Conditioning + Multi-Scale Loss (Teacher: Physio+EEG, Student: Vehicle)
==============================================================================
- Baseline: Past2FutureMultiTaskRoadPreview (Non-AR Transformer Enc-Dec)
- Add: state_head on encoder memory -> z_veh (B,2)
- Training: L = L_task + lambda_state*MSE(z_veh, z_phys) + lambda_rev*BCE(rev_logit, rev_gt)
- Inference: no physio/eeg needed; z_veh still can be computed (optional)
"""

import os
import time
from glob import glob
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================
# Unified paths for the reorganized project layout
# =========================
MODULE_DIR = Path(__file__).resolve().parent
TRAINING_DIR = MODULE_DIR.parent
PROJECT_ROOT = TRAINING_DIR.parents[3]
ENTRY_SCRIPT_PATH = TRAINING_DIR / "future_steer_event_rollpeak_transformer_v5_8_amp_tuned_fixed.py"
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "03_results" / "?????" / "??????"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "01_datasets" / "?????" / "??????"
DEFAULT_STYLE_CSV = PROJECT_ROOT / "03_results" / "?????" / "03_??????" / "driver_style_cluster_result.xlsx"
RESULT_ROOT = str(DEFAULT_RESULT_ROOT)


def discover_vehicle_data_root(project_root: Path) -> Path | None:
    dataset_root = project_root / "01_datasets"
    if not dataset_root.exists():
        return None
    for match in dataset_root.rglob("*_vehicle_aligned_cleaned.csv"):
        return match.parent.parent.parent
    return None


def discover_style_csv(project_root: Path) -> Path | None:
    matches = sorted(project_root.rglob("driver_style_cluster_result.xlsx"))
    if matches:
        return matches[0]
    return None


def resolve_existing_path(raw_value: str | os.PathLike[str] | None, fallback: Path | None) -> str:
    if raw_value:
        raw_path = Path(raw_value)
        if raw_path.exists():
            return str(raw_path)
    if fallback is not None and Path(fallback).exists():
        return str(fallback)
    return str(raw_value) if raw_value else ""


def make_run_dir(prefix="TRAIN_V5_4_STATECOND_REV"):
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(RESULT_ROOT) / f"{prefix}_{ts}"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    return run_dir




# =========================
# Runtime paths
# =========================
THIS_DIR = TRAINING_DIR
PROTOCOL_DIR = THIS_DIR / "protocol_primary_control_v2_context_full2s"
PROTOCOL_CONFIG_PATH = Path(os.environ.get("DRIVER_MODEL_PROTOCOL_CONFIG", str(PROTOCOL_DIR / "protocol_config.json")))
FROZEN_SPLIT_PATH = Path(os.environ.get("DRIVER_MODEL_FROZEN_SPLIT", str(PROTOCOL_DIR / "frozen_subject_split.json")))
PROTOCOL_SPLIT_SUMMARY_PATH = Path(os.environ.get("DRIVER_MODEL_PROTOCOL_SPLIT_SUMMARY", str(PROTOCOL_DIR / "split_summary.csv")))
ROOT = resolve_existing_path(
    os.environ.get("DRIVER_MODEL_ROOT"),
    discover_vehicle_data_root(PROJECT_ROOT) or DEFAULT_DATA_ROOT,
)
STYLE_CSV = resolve_existing_path(
    os.environ.get("DRIVER_MODEL_STYLE_CSV"),
    discover_style_csv(PROJECT_ROOT) or DEFAULT_STYLE_CSV,
)
RESULT_ROOT = os.environ.get(
    "DRIVER_MODEL_RESULT_ROOT",
    RESULT_ROOT,
)

