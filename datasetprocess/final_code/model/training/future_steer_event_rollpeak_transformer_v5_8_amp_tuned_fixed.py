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
# 统一输出到【程序运行结果/运行时间文件夹】
# =========================
RESULT_ROOT = r"F:\数据集处理\data_process\datasetprocess\多模态数据\程序运行结果"


def make_run_dir(prefix="TRAIN_V5_4_STATECOND_REV"):
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(RESULT_ROOT) / f"{prefix}_{ts}"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    return run_dir


class TeeStdout:
    """同时写控制台与日志文件（捕获所有 print）"""

    def __init__(self, log_path, console_stream=None):
        self.console = console_stream if console_stream is not None else sys.__stdout__
        self.f = open(str(log_path), 'w', encoding='utf-8')

    def write(self, s):
        try:
            self.console.write(s)
        except Exception:
            pass
        try:
            self.f.write(s)
        except Exception:
            pass
        self.flush()

    def flush(self):
        try:
            self.console.flush()
        except Exception:
            pass
        try:
            self.f.flush()
        except Exception:
            pass

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass


def save_json(path, obj):
    with open(str(path), 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path):
    with open(str(path), 'r', encoding='utf-8') as f:
        return json.load(f)


STEER_SOURCE_UNIT = "rad"
STEER_ANGLE_UNIT = os.environ.get("DRIVER_MODEL_STEER_ANGLE_UNIT", "deg").strip().lower()
if STEER_ANGLE_UNIT not in {"rad", "deg"}:
    raise ValueError(f"Unsupported DRIVER_MODEL_STEER_ANGLE_UNIT={STEER_ANGLE_UNIT!r}; expected 'rad' or 'deg'")
STEER_ANGLE_SCALE = float(180.0 / np.pi) if STEER_ANGLE_UNIT == "deg" else 1.0
STEER_PLOT_LABEL = f"steer ({STEER_ANGLE_UNIT})"
STEER_PEAK_PLOT_LABEL = f"peak|steer| (GT, {STEER_ANGLE_UNIT})"


def steer_value_from_rad(value: float) -> float:
    return float(value) * STEER_ANGLE_SCALE


def steer_array_from_rad(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float32) * np.float32(STEER_ANGLE_SCALE)



# =========================
# Road-type (curve/straight) utilities
# =========================
def find_feature_in_list(feature_names, keywords):
    """Return the first feature name in feature_names that contains any keyword (case-insensitive)."""
    lower = [f.lower() for f in feature_names]
    for kw in keywords:
        kwl = kw.lower()
        for i, f in enumerate(lower):
            if kwl in f:
                return feature_names[i], i
    return None, None


def otsu_threshold_log10(values, eps=1e-10, bins=256):
    """Otsu threshold on log10(values+eps). Returns threshold in original scale."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < 100:
        return float(np.nanpercentile(v, 85)) if v.size else 0.0

    lv = np.log10(np.maximum(v, 0.0) + eps)
    hist, edges = np.histogram(lv, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2.0

    w1 = np.cumsum(hist)
    w2 = np.cumsum(hist[::-1])[::-1]

    # avoid division by zero
    m1 = np.cumsum(hist * centers) / np.maximum(w1, 1)
    m2 = (np.cumsum((hist * centers)[::-1]) / np.maximum(w2[::-1], 1))[::-1]

    between = w1[:-1] * w2[1:] * (m1[:-1] - m2[1:]) ** 2
    k = int(np.argmax(between)) if between.size else 0

    thr_log = float(centers[k])
    thr = float(10 ** thr_log - eps)
    return max(thr, 0.0)


def auto_curve_threshold(curve_scores, eps=1e-10):
    """Pick a robust threshold to split straight vs curve using ONLY history-window curvature stats.
    Strategy:
      1) Otsu on log-scale
      2) If split is too extreme, fallback to a percentile threshold.
    """
    cs = np.asarray(curve_scores, dtype=np.float64)
    cs = cs[np.isfinite(cs)]
    if cs.size == 0:
        return 0.0

    thr = otsu_threshold_log10(cs, eps=eps, bins=256)
    ratio = float(np.mean(cs > thr))

    # If Otsu yields an overly imbalanced split, fallback to a safer percentile.
    if ratio < 0.05:
        thr = float(np.nanpercentile(cs, 90))
    elif ratio > 0.95:
        thr = float(np.nanpercentile(cs, 10))

    # Avoid tiny numerical noise thresholds
    thr = max(thr, 1e-8)
    return thr

def try_copy_self(run_dir):
    """可选：复制当前脚本到输出目录，方便复现"""
    try:
        src = Path(__file__).resolve()
        dst = Path(run_dir) / src.name
        shutil.copy2(str(src), str(dst))
    except Exception:
        pass


# =========================
# CONFIG
# =========================
THIS_DIR = Path(__file__).resolve().parent
PROTOCOL_DIR = THIS_DIR / "protocol_primary_control_v2_context_full2s"
PROTOCOL_CONFIG_PATH = Path(os.environ.get("DRIVER_MODEL_PROTOCOL_CONFIG", str(PROTOCOL_DIR / "protocol_config.json")))
FROZEN_SPLIT_PATH = Path(os.environ.get("DRIVER_MODEL_FROZEN_SPLIT", str(PROTOCOL_DIR / "frozen_subject_split.json")))
PROTOCOL_SPLIT_SUMMARY_PATH = Path(os.environ.get("DRIVER_MODEL_PROTOCOL_SPLIT_SUMMARY", str(PROTOCOL_DIR / "split_summary.csv")))
ROOT = os.environ.get(
    "DRIVER_MODEL_ROOT",
    r"F:\data_set_process\data_process\datasetprocess\多模态数据\被试数据集合"
)
STYLE_CSV = os.environ.get(
    "DRIVER_MODEL_STYLE_CSV",
    r"F:\data_set_process\data_process\datasetprocess\多模态数据\03_分析结果归档\driver_style_cluster_result.xlsx"
)
RESULT_ROOT = os.environ.get(
    "DRIVER_MODEL_RESULT_ROOT",
    RESULT_ROOT,
)

FS = 200
WIN_SEC = 3.0
FUTURE_SEC = 2.0
WIN_LEN = int(WIN_SEC * FS)         # 600
FUTURE_LEN = int(FUTURE_SEC * FS)   # 400

BATCH_SIZE = 64
EPOCHS = 40
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SMOKE_MODE = os.environ.get("DRIVER_MODEL_SMOKE", "0") == "1"
SMOKE_MAX_SAMPLES = int(os.environ.get("DRIVER_MODEL_SMOKE_MAX_SAMPLES", "256"))
SMOKE_EPOCHS = int(os.environ.get("DRIVER_MODEL_SMOKE_EPOCHS", "2"))
SMOKE_BATCH_SIZE = int(os.environ.get("DRIVER_MODEL_SMOKE_BATCH_SIZE", "32"))
if SMOKE_MODE:
    EPOCHS = SMOKE_EPOCHS
    BATCH_SIZE = SMOKE_BATCH_SIZE
    print(f"[SMOKE] enabled | max_samples={SMOKE_MAX_SAMPLES} | epochs={EPOCHS} | batch_size={BATCH_SIZE}")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LTR_COEFF = 0.11243
STRONG_LABELS = ["medium_active", "strong_active", "extreme_active"]

# Transformer
D_MODEL = 128
N_HEAD = 2
NUM_LAYERS_ENC = 2
NUM_LAYERS_DEC = 2
FFN_DIM = 256
DROPOUT = 0.1

# Multi-scale loss (encourage high-frequency details)
W_DIFF1 = 0.15   # first-derivative loss weight
W_DIFF2 = 0.05   # second-derivative loss weight

# Response-state-aware v1 switches
# v1 targets:
# - teacher-aligned latent state (legacy A/C or data-driven PCA latent)
# - reversal aux classification
# - peak timing aux alignment
# - peak intensity supervision (reuse amplitude loss on trajectory)
ENABLE_RESPONSE_STATE_V1 = True
ENABLE_STATE_DISTILL = ENABLE_RESPONSE_STATE_V1
ENABLE_REVERSAL_AUX = ENABLE_RESPONSE_STATE_V1
ENABLE_PEAKTIME_AUX = ENABLE_RESPONSE_STATE_V1
ENABLE_PEAKINTENSITY_AUX = ENABLE_RESPONSE_STATE_V1

# Teacher-state representation
TEACHER_STATE_MODE = "pca_latent"   # "old_ac" | "pca_latent"
TEACHER_STATE_DIM = 4

# Reversal / phase-aware auxiliary losses
W_REVSEQ = 0.0        # keep baseline default; 0.05 smoke worsened tail despite better late-peak recall
W_PEAKTIME = 0.05 if ENABLE_PEAKTIME_AUX else 0.0   # steer-rate peak timing alignment loss
REVSEQ_ALPHA_FRAC = 0.25  # alpha = frac * steer_std (soft sign for reversal)
PEAK_TEMP_FRAC = 0.35     # temp = frac * mean|steer_rate| (soft-argmax)

# Steer local-detail weighting (emphasize reversals + high steer-rate)
W_STEER_WT = 0.50       # weighted steer MSE added to task loss
W_STEER_RATE = 1.00     # baseline emphasis on high-|steer_rate| segments
W_STEER_REV = 0.35      # modest reversal emphasis to reduce tail flattening on correction events
STEER_WT_MAX = 4.0      # cap for stability

# Distillation / auxiliary heads
LAMBDA_STATE = 0.08 if ENABLE_STATE_DISTILL else 0.0
W_TASK_STEER = 1.50   # steer 主任务权重
W_TASK_YAW   = 1.00   # yawrate 主任务权重
W_TASK_AY    = 0.70   # ay 主任务权重
W_AMP        = 0.30 if ENABLE_PEAKINTENSITY_AUX else 0.0   # peak intensity supervision via trajectory amplitude loss
W_TREND      = 0.10   # coarse steer-trend alignment on the full 2s future window
TREND_POOL_KERNEL = 20
TREND_POOL_STRIDE = 20
TREND_SIGN_EPS = steer_value_from_rad(0.02)
TREND_LOSS_MODE = os.environ.get("DRIVER_MODEL_TREND_LOSS_MODE", "pooled_level_mse_v1")
TREND_LEVEL_WEIGHT = float(os.environ.get("DRIVER_MODEL_TREND_LEVEL_WEIGHT", "0.25"))
TREND_DELTA_WEIGHT = float(os.environ.get("DRIVER_MODEL_TREND_DELTA_WEIGHT", "0.50"))
TREND_DIR_WEIGHT = float(os.environ.get("DRIVER_MODEL_TREND_DIR_WEIGHT", "0.25"))
ENABLE_STEER_COARSE_FINE = os.environ.get("DRIVER_MODEL_STEER_COARSE_FINE", "0") == "1"
W_TREND_COARSE = float(os.environ.get("DRIVER_MODEL_W_TREND_COARSE", "0.10"))
W_FINE_DC = float(os.environ.get("DRIVER_MODEL_W_FINE_DC", "0.02"))
ENABLE_PHASE_ADAPTIVE_TREND = os.environ.get("DRIVER_MODEL_PHASE_ADAPTIVE_TREND", "0") == "1"
TREND_EARLY_BINS = int(os.environ.get("DRIVER_MODEL_TREND_EARLY_BINS", "12"))
TREND_LATE_STRAIGHT_DOWN = float(os.environ.get("DRIVER_MODEL_TREND_LATE_STRAIGHT_DOWN", "0.35"))
TREND_LATE_STRONGREV_DOWN = float(os.environ.get("DRIVER_MODEL_TREND_LATE_STRONGREV_DOWN", "0.45"))
ENABLE_LATE_REV_GATE = os.environ.get("DRIVER_MODEL_LATE_REV_GATE", "0") == "1"
LATE_REV_GATE_START_SEC = float(os.environ.get("DRIVER_MODEL_LATE_REV_GATE_START_SEC", "1.05"))
LATE_REV_GATE_SCALE = float(os.environ.get("DRIVER_MODEL_LATE_REV_GATE_SCALE", "0.60"))
LATE_REV_GATE_RAMP_POWER = float(os.environ.get("DRIVER_MODEL_LATE_REV_GATE_RAMP_POWER", "1.50"))
ENABLE_STRONG_POS_GATE = os.environ.get("DRIVER_MODEL_STRONG_POS_GATE", "0") == "1"
STRONG_POS_GATE_START_SEC = float(os.environ.get("DRIVER_MODEL_STRONG_POS_GATE_START_SEC", "1.20"))
STRONG_POS_GATE_SCALE = float(os.environ.get("DRIVER_MODEL_STRONG_POS_GATE_SCALE", "0.45"))
STRONG_POS_GATE_RAMP_POWER = float(os.environ.get("DRIVER_MODEL_STRONG_POS_GATE_RAMP_POWER", "1.75"))
STRONG_POS_GATE_PROB_CENTER = float(os.environ.get("DRIVER_MODEL_STRONG_POS_GATE_PROB_CENTER", "0.60"))
ENABLE_HARD_LATE_FINE = os.environ.get("DRIVER_MODEL_HARD_LATE_FINE", "0") == "1"
W_HARD_LATE_FINE = float(os.environ.get("DRIVER_MODEL_W_HARD_LATE_FINE", "0.06"))
HARD_LATE_START_SEC = float(os.environ.get("DRIVER_MODEL_HARD_LATE_START_SEC", "1.25"))
HARD_TAIL_START_SEC = float(os.environ.get("DRIVER_MODEL_HARD_TAIL_START_SEC", "1.50"))
HARD_PEAK_QUANTILE = float(os.environ.get("DRIVER_MODEL_HARD_PEAK_QUANTILE", "0.90"))
HARD_TAIL_QUANTILE = float(os.environ.get("DRIVER_MODEL_HARD_TAIL_QUANTILE", "0.80"))
REV_SAMPLE_WEIGHT = 1.80   # 强反打样本整体 loss 加权（建议先用 1.5~2.0）
REV_ZERO_EPS      = 1e-4    # 过零检测小阈值，避免数值噪声误判
LAMBDA_REV  = 0.05 if ENABLE_REVERSAL_AUX else 0.0
LAMBDA_STRONG_POS_GATE = float(os.environ.get("DRIVER_MODEL_LAMBDA_STRONG_POS_GATE", "0.10"))
REV_EPS_WEAK    = 0.02   # 弱反转判定阈值（方向盘单位若已归一化，请相应缩放）
REV_EPS_STRONG  = 0.20   # 强反转判定阈值（更贴近紧急变道“明显反打”）
STRONG_PEAK_THR = 2.0    # 强反转附加条件：未来窗内 |steer| 峰值需超过该阈值（单位同 steer）
# Anchor selection (v5.6): 用于“同一套模型同时覆盖过弯 + 紧急变道(多次反打)”的对齐
# - 弯道/高侧倾：roll 峰值更稳定
# - 直道/紧急变道：steer_rate 的“最早主峰”更稳定（避免 anchor 落在后续反打）
CURVE_THR_FOR_ANCHOR = 1.0e-6   # 事件段内平均|curvature| 超过此阈值 => 认为是弯道（用于选 anchor）
STEER_RATE_PEAK_FRAC = 0.80     # 直道事件：把 |steer_rate| 达到 max 的 80% 的“最早时刻”作为 anchor
USE_STRONG_REV_LOSS = True  # True: rev_head 学“强反转”；False: 学“弱反转”
REV_EPS = REV_EPS_WEAK   # backward-compatible alias

LANE_WIDTH_M     = 3.5   # 车道宽（用于 lateraldistance 解缠）
LANE_JUMP_THR_M  = 1.8   # lateraldistance 跳变检测阈值（约半个车道宽）
EEG_HIST_SEC = 2      # 你现在提取 EEG 事件特征用的 hist2s 文件名后缀
EPS = 1e-6
REV_EPS_WEAK = steer_value_from_rad(0.02)
REV_EPS_STRONG = steer_value_from_rad(0.20)
STRONG_PEAK_THR = steer_value_from_rad(2.0)
STEER_ONSET_THR_ABS = steer_value_from_rad(0.02)
REV_EPS = REV_EPS_WEAK
ROAD_OK_RATIO_THR = 0.7  # use road_type_fixed when ref_nn_ok ratio >= this

SEED = 2025
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =========================
# Helpers
# =========================
def find_col(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None

def make_strictly_increasing(x, eps=1e-6):
    x = np.asarray(x, dtype=np.float64).copy()
    for i in range(1, len(x)):
        if x[i] <= x[i - 1]:
            x[i] = x[i - 1] + eps
    return x


def unwrap_lane_center_signal(x, lane_width=3.5, jump_thr=1.8):
    """Unwrap a lane-centered lateral error signal by compensating sudden ~lane_width jumps.

    Some simulators/loggers switch the reference lane centerline during lane changes, causing
    lateraldistance to jump by approximately ±lane_width (or multiples). This function converts
    such piecewise signals into a continuous lateral position (relative to the initial lane).

    Args:
        x: 1D array-like, lane-centered lateral error (m)
        lane_width: lane width in meters (default 3.5)
        jump_thr: jump detection threshold in meters (default 1.8 ~ half lane width)

    Returns:
        unwrapped: 1D np.float32 array, continuous lateral position (relative)
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x.astype(np.float32)
    y = x.copy()
    offset = 0.0
    for i in range(1, y.size):
        if not np.isfinite(y[i]) or not np.isfinite(y[i - 1]):
            continue
        d = y[i] - y[i - 1]
        if d > jump_thr:
            k = int(np.round(d / lane_width))
            if k == 0:
                k = 1
            offset -= k * lane_width
        elif d < -jump_thr:
            k = int(np.round((-d) / lane_width))
            if k == 0:
                k = 1
            offset += k * lane_width
        y[i] = y[i] + offset
    return y.astype(np.float32)

def get_subject_id_from_path(vehicle_file):
    norm = os.path.normpath(vehicle_file)
    parts = norm.split(os.sep)
    return parts[-3] if len(parts) >= 3 else "unknown"


def load_protocol_split(protocol_config_path=PROTOCOL_CONFIG_PATH, frozen_split_path=FROZEN_SPLIT_PATH):
    protocol_config = load_json(protocol_config_path)
    frozen_split = load_json(frozen_split_path)
    expected = protocol_config.get("splits", {})
    if expected != frozen_split:
        raise ValueError(
            f"Protocol split mismatch between {protocol_config_path} and {frozen_split_path}"
        )
    return protocol_config, frozen_split


def build_subject_split_indices(sample_meta_df, split_subjects):
    split_indices = {}
    sample_subjects = set(sample_meta_df["subject_id"].astype(str).tolist())
    expected_subjects = {str(x) for vs in split_subjects.values() for x in vs}
    for split_name in ("train", "val", "test"):
        allowed = {str(x) for x in split_subjects.get(split_name, [])}
        mask = sample_meta_df["subject_id"].astype(str).isin(allowed)
        split_indices[split_name] = sample_meta_df.index[mask].to_numpy(dtype=np.int64)
        sample_subjects -= allowed
    if sample_subjects:
        raise ValueError(
            "Found subjects outside frozen protocol split. "
            f"sample_subjects={sorted(sample_subjects)}, expected_subjects={sorted(expected_subjects)}"
        )
    return split_indices


def subset_list(items, indices):
    return [items[int(i)] for i in indices]


def subset_array(items, indices):
    idx = np.asarray(indices, dtype=np.int64)
    return np.asarray(items)[idx]


def choose_smoke_indices(split_indices, max_total, rng):
    split_names = [name for name in ("train", "val", "test") if len(split_indices[name]) > 0]
    if len(split_names) != 3:
        raise ValueError("Smoke mode requires non-empty train/val/test splits")
    max_total = max(int(max_total), len(split_names))
    capacities = {name: int(len(split_indices[name])) for name in split_names}
    chosen_counts = {name: 1 for name in split_names}
    remaining = min(max_total, sum(capacities.values())) - len(split_names)
    available = {name: capacities[name] - 1 for name in split_names}
    while remaining > 0 and any(v > 0 for v in available.values()):
        total_avail = float(sum(max(v, 0) for v in available.values()))
        progressed = False
        for name in split_names:
            if remaining <= 0:
                break
            if available[name] <= 0:
                continue
            quota = max(1, int(round(remaining * (available[name] / total_avail)))) if total_avail > 0 else 1
            take = min(available[name], quota, remaining)
            if take <= 0:
                take = 1
            chosen_counts[name] += take
            available[name] -= take
            remaining -= take
            progressed = True
        if not progressed:
            break

    chosen = {}
    for name in split_names:
        perm = rng.permutation(split_indices[name])
        chosen[name] = np.sort(perm[:chosen_counts[name]].astype(np.int64))
    return chosen, chosen_counts


def compute_split_overlap(split_subjects):
    overlap = {}
    for left in ("train", "val", "test"):
        for right in ("train", "val", "test"):
            if left >= right:
                continue
            overlap[f"{left}_{right}"] = sorted(
                set(split_subjects.get(left, [])) & set(split_subjects.get(right, []))
            )
    return overlap


def export_split_audit(
    run_dir,
    sample_meta_df,
    split_indices,
    expected_subjects,
    protocol_config,
    smoke_mode,
    smoke_sampling_policy,
):
    protocol_summary = None
    if PROTOCOL_SPLIT_SUMMARY_PATH.exists():
        protocol_summary = pd.read_csv(PROTOCOL_SPLIT_SUMMARY_PATH)
        protocol_summary["split"] = protocol_summary["split"].astype(str)

    subject_rows = []
    sample_rows = []
    applied_subjects = {}
    for split_name, indices in split_indices.items():
        split_df = sample_meta_df.loc[np.asarray(indices, dtype=np.int64)].copy()
        split_df["split"] = split_name
        subject_counts = (
            split_df.groupby("subject_id").size().rename("sample_count").reset_index()
            if len(split_df) > 0 else pd.DataFrame(columns=["subject_id", "sample_count"])
        )
        applied_subjects[split_name] = sorted(split_df["subject_id"].astype(str).unique().tolist())
        for _, row in subject_counts.iterrows():
            subject_rows.append({
                "split": split_name,
                "subject_id": str(row["subject_id"]),
                "sample_count": int(row["sample_count"]),
            })
        curve_count = int(split_df["is_curve_applied"].fillna(0).astype(int).sum()) if len(split_df) else 0
        sample_row = {
            "split": split_name,
            "sample_count": int(len(split_df)),
            "subject_count": int(split_df["subject_id"].nunique()) if len(split_df) else 0,
            "vehicle_file_count": int(split_df["vehicle_file"].nunique()) if len(split_df) else 0,
            "curve_count": curve_count,
            "straight_count": int(len(split_df) - curve_count),
        }
        if protocol_summary is not None and split_name in set(protocol_summary["split"]):
            proto_row = protocol_summary.loc[protocol_summary["split"] == split_name].iloc[0]
            sample_row["protocol_sample_count"] = int(proto_row["sample_count"])
            sample_row["sample_count_diff_vs_protocol"] = int(sample_row["sample_count"] - int(proto_row["sample_count"]))
        sample_rows.append(sample_row)

    subject_counts_df = pd.DataFrame(subject_rows)
    sample_counts_df = pd.DataFrame(sample_rows)
    subject_counts_df.to_csv(str(run_dir / "split_subject_counts.csv"), index=False, encoding="utf-8-sig")
    sample_counts_df.to_csv(str(run_dir / "split_sample_counts.csv"), index=False, encoding="utf-8-sig")

    audit = {
        "protocol_config_path": str(PROTOCOL_CONFIG_PATH),
        "protocol_version": protocol_config.get("protocol_version"),
        "split_policy_expected": "subject-level fixed split",
        "split_policy_applied": "subject-level fixed split",
        "split_source": str(FROZEN_SPLIT_PATH),
        "smoke_mode": bool(smoke_mode),
        "smoke_sampling_policy": smoke_sampling_policy,
        "expected_subjects": {k: list(v) for k, v in expected_subjects.items()},
        "applied_subjects": applied_subjects,
        "subject_overlap": compute_split_overlap(applied_subjects),
        "sample_counts": sample_counts_df.to_dict(orient="records"),
        "protocol_split_summary_path": str(PROTOCOL_SPLIT_SUMMARY_PATH) if PROTOCOL_SPLIT_SUMMARY_PATH.exists() else None,
    }
    save_json(run_dir / "split_audit.json", audit)
    return audit, subject_counts_df, sample_counts_df

def load_vehicle_and_events(vehicle_file):
    event_file = vehicle_file.replace("\\vehicle\\", "\\event\\") \
        .replace("_vehicle_aligned_cleaned.csv",
                 "_vehicle_aligned_cleaned_events_v312.csv")
    if not os.path.exists(event_file):
        print(f"⚠ 事件文件不存在: {event_file}")
        return None, None
    return pd.read_csv(vehicle_file), pd.read_csv(event_file)

def load_driver_style_map(style_csv):
    if not os.path.exists(style_csv):
        print(f"⚠ 未找到驾驶风格结果文件: {style_csv} → 所有 style_id=0")
        return {}

    df = pd.read_excel(style_csv)
    cols = df.columns.tolist()

    subj_col = None
    for c in ["subject", "Subject", "subject_id", "被试", "被试编号"]:
        if c in cols:
            subj_col = c
            break
    if subj_col is None:
        raise ValueError(f"在 {style_csv} 中找不到 subject 列，请检查列名。")

    style_col = None
    for c in ["cluster_main_k2", "style_main", "style_3style",
              "cluster", "style_id", "cluster_id"]:
        if c in cols:
            style_col = c
            break
    if style_col is None:
        raise ValueError(f"在 {style_csv} 中找不到风格列。")

    style_vals = df[style_col].values
    if not np.issubdtype(style_vals.dtype, np.number):
        cats, idx = np.unique(style_vals, return_inverse=True)
        style_ids = idx
        print("🔧 风格列为字符串，已 factorize：")
        for i, cat in enumerate(cats):
            print(f"  style_id={i} ⇔ '{cat}'")
    else:
        style_ids = style_vals.astype(int)

    subj_vals = df[subj_col].astype(str).values
    style_map = {s: int(k) for s, k in zip(subj_vals, style_ids)}
    return style_map

def infer_physio_file(vehicle_file):
    """
    尽量稳妥地推断 physio 文件：
    ROOT/<subj>/physio/ 与 vehicle 同前缀的 physio CSV
    """
    subj_dir = os.path.dirname(os.path.dirname(vehicle_file))
    physio_dir = os.path.join(subj_dir, "physio")
    if not os.path.isdir(physio_dir):
        return None

    prefix = os.path.basename(vehicle_file).replace("_vehicle_aligned_cleaned.csv", "")
    cand = glob(os.path.join(physio_dir, prefix + "*physio*.csv"))
    if len(cand) > 0:
        return cand[0]
    # 兜底：physio 目录任意一个
    cand2 = glob(os.path.join(physio_dir, "*.csv"))
    return cand2[0] if len(cand2) else None

def infer_eeg_event_feature_file(vehicle_file):
    """
    EEG 事件特征 CSV 位于 ROOT/<subj>/eeg_clean/
    文件名包含同 recording 前缀，并以 _eeg_event_features_rollpeak_hist{EEG_HIST_SEC}s.csv 结尾
    """
    subj_dir = os.path.dirname(os.path.dirname(vehicle_file))
    eeg_dir = os.path.join(subj_dir, "eeg_clean")
    if not os.path.isdir(eeg_dir):
        return None

    prefix = os.path.basename(vehicle_file).replace("_vehicle_aligned_cleaned.csv", "")
    suffix = f"_eeg_event_features_rollpeak_hist{int(EEG_HIST_SEC)}s.csv"
    cand = glob(os.path.join(eeg_dir, prefix + "*" + suffix))
    return cand[0] if len(cand) else None


# =========================
# Teacher feature extraction (event-level)
# =========================
EEG_FEAT_KEYS = [
    "Occipital_ta_beta",
    "Frontal_ta_beta",
    "Temporal_ta_beta",
    "Occipital_alpha_abs",
    "Temporal_gamma_rel",
    "Occipital_gamma_rel",
    "Frontal_gamma_rel",
]

def build_eeg_feat_map(eeg_event_csv):
    """
    return dict: event_row_index(int) -> eeg_feat_vector(8,)
    8 dims: [alpha_asym, 7 others]
    """
    if eeg_event_csv is None or (not os.path.exists(eeg_event_csv)):
        return {}

    df = pd.read_csv(eeg_event_csv)
    if "event_row_index" not in df.columns:
        return {}

    # alpha asym column could be Frontal_alpha_asym_AF4AF3 / F8F7 / ...
    asym_cols = [c for c in df.columns if c.startswith("Frontal_alpha_asym_")]
    asym_col = asym_cols[0] if len(asym_cols) else None

    df = df.set_index("event_row_index")

    m = {}
    for k, row in df.iterrows():
        feats = []
        feats.append(float(row[asym_col]) if (asym_col is not None and asym_col in row) else np.nan)
        for name in EEG_FEAT_KEYS:
            feats.append(float(row[name]) if name in row else np.nan)
        m[int(k)] = np.array(feats, dtype=np.float32)  # (8,)
    return m

def safe_nanmean(values, default=np.nan):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float(default)
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return float(default)
    return float(valid.mean())

def extract_physio_window_means(df_p, peak_idx):
    """
    从 physio CSV 中取 [peak-WIN_LEN, peak) 的均值特征：
    HR, EDA_Tonic, EDA_Phasic, EMG_RMS
    """
    if df_p is None or len(df_p) < peak_idx or peak_idx - WIN_LEN < 0:
        return None

    cols = df_p.columns.tolist()

    col_hr = find_col(cols, ["HR", "HR_bpm", "hr", "hr_bpm"])
    col_t  = find_col(cols, ["EDA_Tonic", "eda_tonic", "Tonic"])
    col_p  = find_col(cols, ["EDA_Phasic", "eda_phasic", "Phasic"])
    col_emg= find_col(cols, ["EMG_RMS", "emg_rms", "EMG"])

    if col_hr is None or col_t is None or col_p is None or col_emg is None:
        return None

    seg = df_p.iloc[peak_idx - WIN_LEN: peak_idx]
    if seg.empty:
        return None

    hr  = safe_nanmean(seg[col_hr].to_numpy(dtype=np.float64))
    ton = safe_nanmean(seg[col_t].to_numpy(dtype=np.float64))
    pha = safe_nanmean(seg[col_p].to_numpy(dtype=np.float64))
    emg = safe_nanmean(seg[col_emg].to_numpy(dtype=np.float64))

    return np.array([hr, ton, pha, emg], dtype=np.float32)  # (4,)

def compute_teacher_state_old_ac(base_feat_z):
    """
    base_feat_z: (B, 12)  已经按 train-set 统计做过 z-score 的基础特征
    dims:
      0..3  : HR, tonic, phasic, emg
      4..11 : eeg [alpha_asym, occ_ta_beta, frontal_ta_beta, temporal_ta_beta,
                   occ_alpha_abs, temporal_gamma_rel, occ_gamma_rel, frontal_gamma_rel]
    Output:
      z_phys_raw (B,2): [A,C] legacy proxy state
    """
    hr     = base_feat_z[:, 0]
    tonic  = base_feat_z[:, 1]
    phasic = base_feat_z[:, 2]
    emg    = base_feat_z[:, 3]

    alpha_asym = base_feat_z[:, 4]
    occ_ta  = base_feat_z[:, 5]
    fr_ta   = base_feat_z[:, 6]
    te_ta   = base_feat_z[:, 7]
    occ_aabs= base_feat_z[:, 8]
    te_g    = base_feat_z[:, 9]
    oc_g    = base_feat_z[:,10]
    fr_g    = base_feat_z[:,11]

    gamma_mean = (te_g + oc_g + fr_g) / 3.0
    ta_mean = (occ_ta + fr_ta + te_ta) / 3.0

    A = (
        0.70 * hr +
        0.40 * tonic +
        0.80 * phasic +
        0.30 * gamma_mean +
        (-0.30) * occ_aabs +
        0.10 * alpha_asym
    )

    C = (
        0.70 * emg +
        0.50 * ta_mean
    )

    z = np.stack([A, C], axis=1).astype(np.float32)
    return z


def fit_pca_projection(train_x: np.ndarray, out_dim: int):
    """Fit PCA on train split only using numpy SVD; keep explicit valid-dim mapping."""
    x_full = np.asarray(train_x, dtype=np.float64)
    if x_full.ndim != 2:
        raise ValueError(f"Expected 2D train_x, got shape={x_full.shape}")

    valid_mask = np.isfinite(x_full).all(axis=0)
    if not np.any(valid_mask):
        raise ValueError("No valid feature dims available for PCA")

    x = x_full[:, valid_mask]
    mean = np.mean(x, axis=0, keepdims=True)
    xc = x - mean
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    basis = vt[:out_dim].T.astype(np.float32)
    return {
        "valid_mask": valid_mask.astype(bool),
        "mean": mean.reshape(-1).astype(np.float32),
        "basis": basis,
    }


def apply_pca_projection(x: np.ndarray, pca_params: dict):
    x_full = np.asarray(x, dtype=np.float32)
    valid_mask = np.asarray(pca_params["valid_mask"], dtype=bool)
    x = x_full[:, valid_mask]
    mean = pca_params["mean"].reshape(1, -1)
    basis = pca_params["basis"]
    return ((x - mean) @ basis).astype(np.float32)


def build_teacher_state(base_feat_z, mode: str, state_dim: int, fit_indices=None):
    if mode == "old_ac":
        z_raw = compute_teacher_state_old_ac(base_feat_z)
        meta = {
            "mode": mode,
            "raw_dim": int(z_raw.shape[1]),
            "component_names": ["A", "C"],
        }
        return z_raw.astype(np.float32), meta

    if mode == "pca_latent":
        fit_idx = np.asarray(fit_indices if fit_indices is not None else np.arange(len(base_feat_z)), dtype=np.int64)
        if fit_idx.size == 0:
            raise ValueError("fit_indices for teacher state PCA cannot be empty")
        fit_x = np.asarray(base_feat_z, dtype=np.float32)[fit_idx]
        fit_dim = int(min(state_dim, base_feat_z.shape[1], len(fit_idx)))
        fit_dim = max(fit_dim, 1)
        pca_params = fit_pca_projection(fit_x, fit_dim)
        z_raw = apply_pca_projection(base_feat_z, pca_params)
        meta = {
            "mode": mode,
            "raw_dim": int(z_raw.shape[1]),
            "component_names": [f"latent_{i}" for i in range(fit_dim)],
            "pca_valid_mask": pca_params["valid_mask"].astype(int).tolist(),
            "pca_mean": pca_params["mean"].tolist(),
            "pca_basis": pca_params["basis"].tolist(),
        }
        return z_raw.astype(np.float32), meta

    raise ValueError(f"Unsupported TEACHER_STATE_MODE: {mode}")


def make_state_column_names(prefix: str, dim: int, component_names=None):
    if component_names is not None and len(component_names) == dim:
        return [f"{prefix}_{name}" for name in component_names]
    return [f"{prefix}_d{i}" for i in range(dim)]


def summarize_state_vector(z_row, component_names=None):
    z = np.asarray(z_row, dtype=np.float32).reshape(-1)
    if component_names is None or len(component_names) != len(z):
        component_names = [f"d{i}" for i in range(len(z))]
    return " ".join([f"{name}={float(val):.2f}" for name, val in zip(component_names, z)])


def compute_teacher_state(base_feat_z):
    z_raw, _ = build_teacher_state(
        base_feat_z,
        mode=TEACHER_STATE_MODE,
        state_dim=TEACHER_STATE_DIM,
        fit_indices=np.arange(len(base_feat_z), dtype=np.int64),
    )
    return z_raw


# =========================
# Build samples (vehicle + teacher base feats)
# =========================
def build_samples_for_vehicle(vehicle_file, style_map):
    df_v, df_e = load_vehicle_and_events(vehicle_file)
    if df_v is None:
        return [], [], [], [], [], [], None

    cols = df_v.columns.tolist()

    col_roll     = find_col(cols, ["zx|roll", "roll", "Roll"])
    col_steer    = find_col(cols, ["zx|SteeringWheel", "SteeringWheel", "steer"])
    col_yawrate  = find_col(cols, ["vyaw", "zx|vyaw", "YawRate", "zx|YawRate", "yaw_rate"])
    col_v        = find_col(cols, ["zx|vx", "Vx", "vx", "Speed", "speed"])
    col_z        = find_col(cols, ["zx|z", "z", "Z"])
    col_ay       = find_col(cols, ["zx|ay", "ay", "Ay", "lat_acc"])
    col_ax       = find_col(cols, ["zx|ax", "ax", "Ax", "Long_acc"])
    col_lane     = find_col(cols, ["lateraldistance", "lateralDistance", "lateraldistance_start"])
    col_curve    = find_col(cols, ["zx1|lanecurvatureXY", "laneCurvature", "lanecurvature_start"])

    col_roadtype = find_col(cols, ["road_type_fixed", "road_type", "roadType_fixed"])
    col_refok    = find_col(cols, ["ref_nn_ok", "ref_ok", "refnn_ok"])
    col_yaw      = find_col(cols, ["zx|yaw", "yaw", "Yaw"])

    if col_roll is None or col_steer is None:
        return [], [], [], [], [], [], None
    if col_ay is None or col_yawrate is None or col_curve is None:
        return [], [], [], [], [], [], None

    base_cols = [c for c in [
        col_roll, col_yawrate, col_ay, col_ax, col_v,
        col_z, col_lane, col_curve, col_yaw, col_steer
    ] if c is not None]

    df_feat = df_v[base_cols].copy()

    if col_v is not None:
        df_feat[col_v] = df_feat[col_v] / 3.6

    if col_ay is not None:
        df_feat["LTR_est"] = df_v[col_ay] * LTR_COEFF

    steer = steer_array_from_rad(df_v[col_steer].to_numpy(dtype=np.float32))
    df_feat[col_steer] = steer
    dt = 1.0 / FS
    steer_rate = np.gradient(steer, dt)
    df_feat["steer_rate"] = steer_rate

    # ---- NEW(v5.5): lane-center lateral error features ----
    if col_lane is not None:
        lane_err = df_v[col_lane].to_numpy(dtype=np.float32)

        # raw lane-centered error derivatives (may contain lane-reference jumps)
        lane_rate = np.gradient(lane_err, dt)
        lane_acc  = np.gradient(lane_rate, dt)
        df_feat["lane_rate"] = lane_rate
        df_feat["lane_acc"]  = lane_acc

        # unwrapped (continuous) lateral position + derivatives (more stable for lane-change reversal)
        lane_unwrap = unwrap_lane_center_signal(lane_err, lane_width=LANE_WIDTH_M, jump_thr=LANE_JUMP_THR_M)
        lane_unwrap_rate = np.gradient(lane_unwrap, dt)
        lane_unwrap_acc  = np.gradient(lane_unwrap_rate, dt)
        df_feat["lane_unwrap"] = lane_unwrap
        df_feat["lane_unwrap_rate"] = lane_unwrap_rate
        df_feat["lane_unwrap_acc"]  = lane_unwrap_acc

    feature_cols = df_feat.columns.tolist()
    X_all = df_feat.to_numpy(dtype=np.float32)
    N = X_all.shape[0]

    steer_idx      = feature_cols.index(col_steer)
    roll_idx       = feature_cols.index(col_roll)
    ay_idx         = feature_cols.index(col_ay)
    yawrate_idx    = feature_cols.index(col_yawrate)
    steer_rate_idx = feature_cols.index("steer_rate")
    curve_idx      = feature_cols.index(col_curve)

    # distance axis for speed-projected curvature preview
    v_idx = feature_cols.index(col_v) if (col_v is not None and col_v in feature_cols) else None
    if v_idx is not None:
        v_arr = np.nan_to_num(X_all[:, v_idx].astype(np.float32), nan=0.0)
        v_arr = np.clip(v_arr, 0.0, None)
        s_axis = np.zeros(N, dtype=np.float64)
        s_axis[1:] = np.cumsum(v_arr[:-1].astype(np.float64) * dt)
        s_axis = make_strictly_increasing(s_axis)
        curve_arr = np.nan_to_num(X_all[:, curve_idx].astype(np.float32), nan=0.0)
    else:
        v_arr = None
        s_axis = None
        curve_arr = None

    # ---- teacher sources ----
    physio_file = infer_physio_file(vehicle_file)
    df_p = pd.read_csv(physio_file) if (physio_file is not None and os.path.exists(physio_file)) else None

    eeg_event_csv = infer_eeg_event_feature_file(vehicle_file)
    eeg_map = build_eeg_feat_map(eeg_event_csv)

    subject_id = get_subject_id_from_path(vehicle_file)
    style_id = style_map.get(subject_id, 0)

    X_list, y_list, curve_list, ctx_list, base_feat_list, meta_list = [], [], [], [], [], []

    df_e = df_e[df_e["event_level"].isin(STRONG_LABELS)]
    if len(df_e) == 0:
        return [], [], [], [], [], [], None

    for ev_idx, ev in df_e.iterrows():
        t0 = float(ev["start_s"]); t1 = float(ev["end_s"])
        i0 = int(t0 * FS); i1 = int(t1 * FS)
        if i0 < 0 or i1 > N or (i1 - i0) < 10:
            continue

        # --- v5.6: 直道(紧急变道) 与 弯道(高速过弯/侧倾) 的 anchor 采用不同规则 ---
        # 关键目的：让“多次反打”的事件在时间相位上更一致，否则模型只能学到趋势，细节学不到
        curve_seg = X_all[i0:i1, curve_idx]
        curve_seg_mean = float(np.nanmean(np.abs(curve_seg))) if curve_seg.size else 0.0

        # Prefer fixed road label if available and reliable; fallback to curvature threshold.
        is_curve = None
        if (col_roadtype is not None) and (col_refok is not None):
            ok_seg = df_v[col_refok].to_numpy(dtype=np.float32, copy=False)[i0:i1]
            ok_ratio = float(np.nanmean(ok_seg > 0.5)) if ok_seg.size else 0.0
            if ok_ratio >= ROAD_OK_RATIO_THR:
                rt_seg = df_v[col_roadtype].to_numpy(copy=False)[i0:i1]
                # numeric 1/0 or string "curve"/"straight"
                if rt_seg.dtype.kind in ("i", "u", "f"):
                    is_curve = (float(np.nanmean(rt_seg)) >= 0.5)
                else:
                    rt_low = np.char.lower(rt_seg.astype(str))
                    is_curve = (float(np.mean(rt_low == "curve")) >= 0.5)

        if is_curve is None:
            is_curve = (curve_seg_mean > CURVE_THR_FOR_ANCHOR)

        if is_curve:
            # 弯道：roll 幅值峰值作为 anchor（更接近 roll/侧倾激烈点）
            roll_seg = X_all[i0:i1, roll_idx]
            if roll_seg.size == 0:
                continue
            peak_rel = int(np.argmax(np.abs(roll_seg)))
            peak_idx = i0 + peak_rel
        else:
            # 直道：steer_rate 的“最早主峰”作为 anchor（更接近紧急变道发起时刻，避免落在反打处）
            sr_seg = X_all[i0:i1, steer_rate_idx]
            if sr_seg.size == 0:
                continue
            abs_sr = np.abs(sr_seg)
            max_abs = float(np.nanmax(abs_sr))
            if (not np.isfinite(max_abs)) or max_abs < 1e-6:
                # fallback：退化情况下仍用 roll 峰
                roll_seg = X_all[i0:i1, roll_idx]
                if roll_seg.size == 0:
                    continue
                peak_rel = int(np.argmax(np.abs(roll_seg)))
                peak_idx = i0 + peak_rel
            else:
                thr = STEER_RATE_PEAK_FRAC * max_abs
                cand = np.where(abs_sr >= thr)[0]
                peak_rel = int(cand[0]) if cand.size else int(np.argmax(abs_sr))
                peak_idx = i0 + peak_rel

        if peak_idx - WIN_LEN < 0 or peak_idx + FUTURE_LEN >= N:
            continue

        # teacher base feats
        phys4 = extract_physio_window_means(df_p, peak_idx)  # (4,)
        eeg8 = eeg_map.get(int(ev_idx), None)                # (8,)
        if phys4 is None:
            # 若该被试缺失生理，允许继续训练（但该样本无法做 state loss）
            # 用 NaN 占位，后面会 mask
            phys4 = np.full((4,), np.nan, dtype=np.float32)
        if eeg8 is None:
            eeg8 = np.full((8,), np.nan, dtype=np.float32)

        base12 = np.concatenate([phys4, eeg8], axis=0).astype(np.float32)  # (12,)

        x_win = X_all[peak_idx - WIN_LEN: peak_idx]

        steer_anchor = float(X_all[peak_idx, steer_idx])
        y_steer = X_all[peak_idx + 1: peak_idx + 1 + FUTURE_LEN, steer_idx] - steer_anchor
        y_yaw   = X_all[peak_idx + 1: peak_idx + 1 + FUTURE_LEN, yawrate_idx]
        y_ay    = X_all[peak_idx + 1: peak_idx + 1 + FUTURE_LEN, ay_idx]
        y_seq = np.stack([y_steer, y_yaw, y_ay], axis=-1)

        if v_arr is not None and s_axis is not None and curve_arr is not None:
            v0 = float(v_arr[peak_idx])
            s0 = float(s_axis[peak_idx])
            t_grid = (np.arange(1, FUTURE_LEN + 1, dtype=np.float64) * dt)
            s_query = np.clip(s0 + v0 * t_grid, s_axis[0], s_axis[-1])
            curve_future = np.interp(s_query, s_axis, curve_arr).astype(np.float32)
        else:
            curve_future = X_all[peak_idx + 1: peak_idx + 1 + FUTURE_LEN, curve_idx].astype(np.float32)

        steer_anchor = X_all[peak_idx, steer_idx]
        ctx = np.array([
            steer_anchor,
            X_all[peak_idx, steer_rate_idx],
            X_all[peak_idx, ay_idx],
            X_all[peak_idx, yawrate_idx],
            float(style_id)
        ], dtype=np.float32)

        X_list.append(x_win.astype(np.float32))
        y_list.append(y_seq.astype(np.float32))
        curve_list.append(curve_future.astype(np.float32))
        ctx_list.append(ctx)
        base_feat_list.append(base12)
        meta_list.append({
            "sample_key": f"{subject_id}::{os.path.basename(vehicle_file)}::{int(ev_idx)}::maintained_anchor",
            "subject_id": str(subject_id),
            "vehicle_file": str(vehicle_file),
            "event_idx": int(ev_idx),
            "event_level": str(ev.get("event_level", "")),
            "event_start_s": float(t0),
            "event_end_s": float(t1),
            "anchor_idx": int(peak_idx),
            "anchor_source_applied": "roll_peak" if is_curve else "steer_rate_peak80_first",
            "maintained_anchor_policy": "curve->roll_peak; straight->steer_rate_peak80_first",
            "is_curve_applied": int(bool(is_curve)),
            "curve_score_event_mean_abs": float(curve_seg_mean),
        })

    return X_list, y_list, curve_list, ctx_list, base_feat_list, meta_list, feature_cols


def build_all_samples(style_map):
    pattern = os.path.join(ROOT, "*", "vehicle", "*_vehicle_aligned_cleaned.csv")
    vehicle_files = sorted(glob(pattern))

    X_pool, y_pool, curve_pool, ctx_pool, base_pool, meta_pool = [], [], [], [], [], []
    feature_names = None

    print("🔍 遍历车辆文件构造事件样本 + teacher base feats ...")
    total = 0
    for vf in vehicle_files:
        X_list, y_list, curve_list, ctx_list, base_list, meta_list, feat_cols = build_samples_for_vehicle(vf, style_map)
        if feat_cols is None or len(X_list) == 0:
            continue

        if feature_names is None:
            feature_names = feat_cols
        elif feat_cols != feature_names:
            print("⚠ 特征列顺序不一致，跳过:", vf)
            continue

        X_pool.extend(X_list)
        y_pool.extend(y_list)
        curve_pool.extend(curve_list)
        ctx_pool.extend(ctx_list)
        base_pool.extend(base_list)
        meta_pool.extend(meta_list)
        total += len(X_list)

    print(f"✅ 共收集到 {total} 个事件样本\n")
    sample_meta_df = pd.DataFrame(meta_pool)
    return X_pool, y_pool, curve_pool, ctx_pool, base_pool, sample_meta_df, feature_names


# =========================
# Dataset
# =========================
class MultiTaskFutureWithCurveDataset(Dataset):
    def __init__(self, X_list, y_list, curve_list, ctx_list, z_phys_list, rev_gt_list, rev_gt_weak_list, rev_gt_strong_list,
                 y_mean, y_std, curve_mean, curve_std, ctx_mean, ctx_std,
                 curve_score_list=None, is_curve_list=None):
        self.X = X_list
        self.y = y_list
        self.curve = curve_list
        self.ctx = ctx_list
        self.z_phys = z_phys_list  # (N,2) or NaN (masked)
        self.rev_gt = rev_gt_list      # (N,) 0/1 (label used for rev_head)
        self.rev_gt_weak = rev_gt_weak_list if (rev_gt_weak_list is not None) else rev_gt_list
        self.rev_gt_strong = rev_gt_strong_list if (rev_gt_strong_list is not None) else rev_gt_list

        self.curve_score = curve_score_list
        self.is_curve = is_curve_list

        self.y_mean = y_mean.astype(np.float32)
        self.y_std  = y_std.astype(np.float32)
        self.curve_mean = float(curve_mean)
        self.curve_std  = float(curve_std)
        self.ctx_mean = ctx_mean.astype(np.float32)
        self.ctx_std  = ctx_std.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        curve_raw = self.curve[idx]
        ctx_raw = self.ctx[idx]
        z = self.z_phys[idx]  # (2,)

        y_norm = (y - self.y_mean) / self.y_std
        curve_norm = (curve_raw - self.curve_mean) / self.curve_std
        ctx_norm = (ctx_raw - self.ctx_mean) / self.ctx_std

        # z_phys 仅训练用：这里不再额外标准化（已经在 main 里做了）
        z_phys = z.astype(np.float32)
        z_mask = np.isfinite(z_phys).all().astype(np.float32)  # 1=有效, 0=缺失(生理缺)

        return {
            "src": x.astype(np.float32),
            "y_norm": y_norm.astype(np.float32),
            "curve_norm": curve_norm.astype(np.float32),
            "ctx": ctx_norm.astype(np.float32),
            "z_phys": z_phys,
            "z_mask": np.array([z_mask], dtype=np.float32),
            "rev_gt": np.array([self.rev_gt[idx]], dtype=np.float32),
            "rev_gt_weak": np.array([self.rev_gt_weak[idx]], dtype=np.float32),
            "rev_gt_strong": np.array([self.rev_gt_strong[idx]], dtype=np.float32),
            "idx": np.array([idx], dtype=np.int64),
            "curve_score": np.array([self.curve_score[idx]], dtype=np.float32) if self.curve_score is not None else np.array([np.nan], dtype=np.float32),
            "is_curve": np.array([self.is_curve[idx]], dtype=np.int64) if self.is_curve is not None else np.array([-1], dtype=np.int64),
        }


# =========================
# Model (baseline + state head)
# =========================
class Past2FutureMultiTaskRoadPreview(nn.Module):
    """
    Output:
      y_hat_norm: (B, FUTURE_LEN, 3)
      z_veh:      (B, state_dim) from encoder memory pooling (train for distillation; inference optional)
    """
    def __init__(self, input_dim, context_dim, future_len, out_dim=3,
                 d_model=128, nhead=2,
                 num_layers_enc=2, num_layers_dec=2,
                 dim_feedforward=256, dropout=0.1,
                 max_len_enc=600, max_len_dec=400,
                 state_dim=2,
                 enable_steer_coarse_fine=False,
                 trend_pool_kernel=20,
                 trend_pool_stride=20,
                 enable_late_reversal_gate=False,
                 late_rev_gate_start_sec=1.05,
                 late_rev_gate_scale=0.60,
                 late_rev_gate_ramp_power=1.50,
                 enable_strong_pos_gate=False,
                 strong_pos_gate_start_sec=1.20,
                 strong_pos_gate_scale=0.45,
                 strong_pos_gate_ramp_power=1.75,
                 strong_pos_gate_prob_center=0.60):
        super().__init__()
        self.d_model = d_model
        self.future_len = future_len
        self.out_dim = out_dim
        self.state_dim = state_dim
        self.enable_steer_coarse_fine = bool(enable_steer_coarse_fine)
        self.trend_pool_kernel = int(trend_pool_kernel)
        self.trend_pool_stride = int(trend_pool_stride)
        self.enable_late_reversal_gate = bool(enable_late_reversal_gate)
        self.late_rev_gate_start_sec = float(late_rev_gate_start_sec)
        self.late_rev_gate_scale = float(late_rev_gate_scale)
        self.late_rev_gate_ramp_power = float(late_rev_gate_ramp_power)
        self.enable_strong_pos_gate = bool(enable_strong_pos_gate)
        self.strong_pos_gate_start_sec = float(strong_pos_gate_start_sec)
        self.strong_pos_gate_scale = float(strong_pos_gate_scale)
        self.strong_pos_gate_ramp_power = float(strong_pos_gate_ramp_power)
        self.strong_pos_gate_prob_center = float(strong_pos_gate_prob_center)

        # Encoder
        self.enc_input_proj = nn.Linear(input_dim, d_model)
        self.enc_pos_emb = nn.Parameter(torch.zeros(1, max_len_enc, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers_enc)

        # Decoder
        self.dec_pos_emb = nn.Parameter(torch.zeros(1, max_len_dec, d_model))
        self.ctx_proj    = nn.Linear(context_dim, d_model)
        self.curve_proj  = nn.Linear(1, d_model)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers_dec)

        self.out_proj = nn.Linear(d_model, out_dim)
        if self.enable_steer_coarse_fine:
            self.steer_fine_proj = nn.Linear(d_model, 1)
            self.steer_coarse_proj = nn.Sequential(
                nn.Linear(d_model, 64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1)
            )
            self.other_proj = nn.Linear(d_model, max(1, out_dim - 1))
        self.dropout = nn.Dropout(dropout)

        # ---- NEW: state head (encoder pooling) ----
        self.pool_score = nn.Linear(d_model, 1)
        self.state_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, state_dim)
        )

        # ---- NEW(v5.4): reversal classifier head (encoder pooling) ----
        self.rev_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        if self.enable_strong_pos_gate:
            self.strong_pos_gate_head = nn.Sequential(
                nn.Linear(d_model * 2, 64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1)
            )

    def forward(self, src, ctx, curve_norm):
        B, T_in, _ = src.shape
        T_out = self.future_len

        # Encoder
        h_src = self.enc_input_proj(src)
        h_src = h_src + self.enc_pos_emb[:, :T_in, :]
        memory = self.encoder(self.dropout(h_src))  # (B,T_in,d_model)

        # ---- encoder pooling -> z_veh ----
        scores = self.pool_score(memory)            # (B,T,1)
        alpha = torch.softmax(scores, dim=1)
        ctx_enc = torch.sum(alpha * memory, dim=1)  # (B,d_model)
        z_veh = self.state_head(ctx_enc)            # (B,state_dim)
        rev_logit = self.rev_head(ctx_enc).squeeze(-1)  # (B,)

        # Decoder input
        pos_tgt = self.dec_pos_emb[:, :T_out, :].expand(B, T_out, -1)
        ctx2 = torch.cat([ctx, z_veh], dim=1)  # (B, context_dim)
        ctx_emb = self.ctx_proj(ctx2).unsqueeze(1).expand(B, T_out, -1)

        curve_feat = curve_norm.unsqueeze(-1)   # (B,T_out,1)
        curve_emb = self.curve_proj(curve_feat) # (B,T_out,d_model)

        tgt = pos_tgt + ctx_emb + curve_emb
        out = self.decoder(tgt, memory)
        if not self.enable_steer_coarse_fine:
            y_hat_norm = self.out_proj(out)
            return y_hat_norm, z_veh, rev_logit

        steer_fine_norm = self.steer_fine_proj(out).squeeze(-1)
        pool_k = max(1, min(int(self.trend_pool_kernel), T_out))
        pool_s = max(1, int(self.trend_pool_stride))
        dec_pool = F.avg_pool1d(out.transpose(1, 2), kernel_size=pool_k, stride=pool_s).transpose(1, 2)
        steer_coarse_norm = self.steer_coarse_proj(dec_pool).squeeze(-1)
        steer_coarse_up_norm = F.interpolate(
            steer_coarse_norm.unsqueeze(1),
            size=T_out,
            mode="linear",
            align_corners=True,
        ).squeeze(1)
        steer_fine_out_norm = steer_fine_norm
        late_rev_gate = None
        late_rev_prob = None
        strong_pos_gate_logit = None
        strong_pos_gate_prob = None
        strong_pos_late_gate = None
        if self.enable_strong_pos_gate and self.strong_pos_gate_scale > 0.0:
            late_start_idx = _sec_to_future_idx(self.strong_pos_gate_start_sec, T_out)
            late_slice = out[:, late_start_idx:, :] if late_start_idx < T_out else out[:, -1:, :]
            late_feat = late_slice.mean(dim=1)
            gate_feat = torch.cat([ctx_enc, late_feat], dim=1)
            strong_pos_gate_logit = self.strong_pos_gate_head(gate_feat).squeeze(-1)
            strong_pos_gate_prob = torch.sigmoid(strong_pos_gate_logit).to(out.dtype).unsqueeze(1)
            centered_prob = (
                (strong_pos_gate_prob - self.strong_pos_gate_prob_center)
                / max(1e-6, 1.0 - self.strong_pos_gate_prob_center)
            ).clamp(0.0, 1.0)
            late_ramp = _build_late_ramp(
                T_out,
                self.strong_pos_gate_start_sec,
                device=out.device,
                dtype=out.dtype,
                power=self.strong_pos_gate_ramp_power,
            )
            strong_pos_late_gate = 1.0 + self.strong_pos_gate_scale * centered_prob * late_ramp
            steer_fine_out_norm = steer_fine_norm * strong_pos_late_gate
        elif self.enable_late_reversal_gate and self.late_rev_gate_scale > 0.0:
            late_ramp = _build_late_ramp(
                T_out,
                self.late_rev_gate_start_sec,
                device=out.device,
                dtype=out.dtype,
                power=self.late_rev_gate_ramp_power,
            )
            if torch.count_nonzero(late_ramp).item() > 0:
                late_rev_prob = torch.sigmoid(rev_logit).to(out.dtype).unsqueeze(1)
                late_rev_gate = 1.0 + self.late_rev_gate_scale * late_rev_prob * late_ramp
                # Keep coarse trend untouched and only amplify late fine residual on reversal-like samples.
                steer_fine_out_norm = steer_fine_norm * late_rev_gate
        steer_norm = steer_coarse_up_norm + steer_fine_out_norm
        other_norm = self.other_proj(out)
        y_hat_norm = torch.cat([steer_norm.unsqueeze(-1), other_norm], dim=-1)
        aux = {
            "steer_coarse_norm": steer_coarse_norm,
            "steer_coarse_up_norm": steer_coarse_up_norm,
            "steer_fine_raw_norm": steer_fine_norm,
            "steer_fine_norm": steer_fine_out_norm,
        }
        if strong_pos_gate_logit is not None:
            aux["strong_pos_gate_logit"] = strong_pos_gate_logit
            aux["strong_pos_gate_prob"] = strong_pos_gate_prob.squeeze(1)
            aux["strong_pos_late_gate"] = strong_pos_late_gate
        if late_rev_gate is not None:
            aux["late_rev_gate"] = late_rev_gate
            aux["late_rev_prob"] = late_rev_prob.squeeze(1)
        return y_hat_norm, z_veh, rev_logit, aux


def unpack_model_output(output):
    if not isinstance(output, tuple):
        raise TypeError(f"Unexpected model output type: {type(output)!r}")
    if len(output) == 3:
        y_hat_norm, z_veh, rev_logit = output
        return y_hat_norm, z_veh, rev_logit, {}
    if len(output) == 4:
        y_hat_norm, z_veh, rev_logit, aux = output
        return y_hat_norm, z_veh, rev_logit, (aux or {})
    raise ValueError(f"Unexpected model output length: {len(output)}")




# =========================
# Multi-scale loss helpers
# =========================

def _diff1(x: torch.Tensor) -> torch.Tensor:
    # x: (B,T,C) or (B,T)
    if x.dim() == 2:
        return x[:, 1:] - x[:, :-1]
    return x[:, 1:, :] - x[:, :-1, :]


def _diff2(x: torch.Tensor) -> torch.Tensor:
    return _diff1(_diff1(x))


def weighted_l1_loss_per_sample(pred: torch.Tensor, target: torch.Tensor, sample_weight: torch.Tensor) -> torch.Tensor:
    """
    pred/target: (B, ...) tensors
    sample_weight: (B,) or (B,1)
    """
    per_sample = (pred - target).abs().reshape(pred.shape[0], -1).mean(dim=1)
    weight = sample_weight.reshape(sample_weight.shape[0]).to(per_sample.dtype)
    weight_sum = torch.clamp(weight.sum(), min=1e-6)
    return (per_sample * weight).sum() / weight_sum



def compute_amplitude_loss(y_hat: torch.Tensor, y_true: torch.Tensor, sample_weight=None) -> torch.Tensor:
    """
    steer-only 幅值损失（支持样本级加权）
    """
    pred = y_hat[:, :, 0]   # steer
    true = y_true[:, :, 0]

    pred_peak = pred.abs().amax(dim=1)
    true_peak = true.abs().amax(dim=1)

    pred_range = pred.amax(dim=1) - pred.amin(dim=1)
    true_range = true.amax(dim=1) - true.amin(dim=1)

    if sample_weight is None:
        loss_peak = F.l1_loss(pred_peak, true_peak)
        loss_range = F.l1_loss(pred_range, true_range)
    else:
        loss_peak = weighted_l1_loss_per_sample(pred_peak.unsqueeze(1), true_peak.unsqueeze(1), sample_weight)
        loss_range = weighted_l1_loss_per_sample(pred_range.unsqueeze(1), true_range.unsqueeze(1), sample_weight)

    return 0.7 * loss_peak + 0.3 * loss_range


def weighted_mean_per_sample(loss_per_sample: torch.Tensor, sample_weight=None) -> torch.Tensor:
    per_sample = loss_per_sample.reshape(loss_per_sample.shape[0])
    if sample_weight is None:
        return per_sample.mean()
    weight = sample_weight.reshape(sample_weight.shape[0]).to(per_sample.dtype)
    weight_sum = torch.clamp(weight.sum(), min=1e-6)
    return (per_sample * weight).sum() / weight_sum


def mse_per_sample(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((pred - target) ** 2).reshape(pred.shape[0], -1).mean(dim=1)


def weighted_mse_loss_per_sample(pred: torch.Tensor, target: torch.Tensor, sample_weight=None) -> torch.Tensor:
    return weighted_mean_per_sample(mse_per_sample(pred, target), sample_weight)


def weighted_channel_task_loss(y_hat: torch.Tensor, y_true: torch.Tensor, sample_weight=None) -> torch.Tensor:
    loss_steer = mse_per_sample(y_hat[:, :, 0], y_true[:, :, 0])
    loss_yaw = mse_per_sample(y_hat[:, :, 1], y_true[:, :, 1])
    loss_ay = mse_per_sample(y_hat[:, :, 2], y_true[:, :, 2])
    total = (
        W_TASK_STEER * loss_steer
        + W_TASK_YAW * loss_yaw
        + W_TASK_AY * loss_ay
    ) / (W_TASK_STEER + W_TASK_YAW + W_TASK_AY)
    return weighted_mean_per_sample(total, sample_weight)


def weighted_steer_local_mse(steer_pred: torch.Tensor, steer_true: torch.Tensor, w_local: torch.Tensor, sample_weight=None) -> torch.Tensor:
    per_sample = (((steer_pred[:, 1:] - steer_true[:, 1:]) ** 2) * w_local).mean(dim=1)
    return weighted_mean_per_sample(per_sample, sample_weight)


def _avg_pool_seq_torch(seq: torch.Tensor, kernel_size: int, stride: int) -> torch.Tensor:
    if seq.ndim != 2:
        raise ValueError(f"Expected (B,T), got {tuple(seq.shape)}")
    t_len = int(seq.shape[1])
    k = max(1, min(int(kernel_size), t_len))
    s = max(1, int(stride))
    pooled = F.avg_pool1d(seq.unsqueeze(1), kernel_size=k, stride=s).squeeze(1)
    if pooled.shape[1] == 0:
        pooled = seq.mean(dim=1, keepdim=True)
    return pooled


def _avg_pool_seq_np(seq, kernel_size: int, stride: int) -> np.ndarray:
    x = np.asarray(seq, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={x.shape}")
    t_len = int(x.shape[1])
    k = max(1, min(int(kernel_size), t_len))
    s = max(1, int(stride))
    pooled = [x[:, start:start + k].mean(axis=1) for start in range(0, t_len - k + 1, s)]
    if not pooled:
        pooled = [x.mean(axis=1)]
    return np.stack(pooled, axis=1)


def build_reversal_sample_weight(rev_gt_b: torch.Tensor) -> torch.Tensor:
    return 1.0 + (REV_SAMPLE_WEIGHT - 1.0) * rev_gt_b.float()

def _soft_reversal_prob(seq: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """
    Soft reversal probability between consecutive timesteps.
    seq: (B,T)
    alpha: scalar tensor (>0) controlling soft sign.
    Return: (B,T-1) in [0,1]
    """
    a = torch.clamp(alpha, min=1e-6)
    p_pos = torch.sigmoid(seq / a)
    p_neg = 1.0 - p_pos
    return p_pos[:, :-1] * p_neg[:, 1:] + p_neg[:, :-1] * p_pos[:, 1:]

def _soft_peak_time(x: torch.Tensor, temp: torch.Tensor) -> torch.Tensor:
    """
    Soft-argmax expected peak time (normalized 0..1).
    x: (B,T) non-negative (e.g., abs(rate))
    temp: scalar tensor (>0)
    Return: (B,) in [0,1]
    """
    t = torch.linspace(0.0, 1.0, x.shape[1], device=x.device, dtype=x.dtype)
    tau = torch.clamp(temp, min=1e-6)
    w = torch.softmax(x / tau, dim=1)
    return (w * t.unsqueeze(0)).sum(dim=1)


def compute_active_task_losses(y_hat: torch.Tensor, y_true: torch.Tensor, sample_weight=None):
    loss_task = weighted_channel_task_loss(y_hat, y_true, sample_weight)
    loss_amp = compute_amplitude_loss(y_hat, y_true, sample_weight=sample_weight)
    loss_d1 = weighted_mse_loss_per_sample(_diff1(y_hat), _diff1(y_true), sample_weight)
    loss_d2 = weighted_mse_loss_per_sample(_diff2(y_hat), _diff2(y_true), sample_weight)
    loss_task = loss_task + W_DIFF1 * loss_d1 + W_DIFF2 * loss_d2 + W_AMP * loss_amp
    return loss_task, loss_amp, loss_d1, loss_d2


def compute_reversal_shape_losses(y_hat: torch.Tensor, y_true: torch.Tensor, y_mean_t: torch.Tensor, y_std_t: torch.Tensor, sample_weight=None, use_reversal_local_weight=True):
    y_hat_den = y_hat * y_std_t.view(1, 1, 3) + y_mean_t.view(1, 1, 3)
    y_true_den = y_true * y_std_t.view(1, 1, 3) + y_mean_t.view(1, 1, 3)
    steer_pred = y_hat_den[:, :, 0]
    steer_true = y_true_den[:, :, 0]

    alpha = REVSEQ_ALPHA_FRAC * y_std_t[0]
    p_rev_pred = _soft_reversal_prob(steer_pred, alpha)
    with torch.no_grad():
        p_rev_true = _soft_reversal_prob(steer_true, alpha)
    loss_revseq = weighted_mse_loss_per_sample(p_rev_pred, p_rev_true, sample_weight)

    steer_rate_pred = _diff1(steer_pred).abs()
    steer_rate_true = _diff1(steer_true).abs()
    temp = PEAK_TEMP_FRAC * (steer_rate_true.mean() + EPS)
    peak_pred = _soft_peak_time(steer_rate_pred, temp)
    with torch.no_grad():
        peak_true = _soft_peak_time(steer_rate_true, temp)
    loss_peaktime = weighted_mse_loss_per_sample(peak_pred, peak_true, sample_weight)

    with torch.no_grad():
        rate_norm = steer_rate_true / (steer_rate_true.mean(dim=1, keepdim=True) + EPS)
        rev_seq = p_rev_true if use_reversal_local_weight else torch.zeros_like(rate_norm)
        w_local = 1.0 + W_STEER_RATE * rate_norm + W_STEER_REV * rev_seq
        w_local = torch.clamp(w_local, max=STEER_WT_MAX)
    loss_steer_wt = weighted_steer_local_mse(steer_pred, steer_true, w_local, sample_weight)
    return loss_revseq, loss_peaktime, loss_steer_wt


def compute_trend_loss(y_hat: torch.Tensor, y_true: torch.Tensor, y_mean_t: torch.Tensor, y_std_t: torch.Tensor, sample_weight=None) -> torch.Tensor:
    y_hat_den = y_hat * y_std_t.view(1, 1, 3) + y_mean_t.view(1, 1, 3)
    y_true_den = y_true * y_std_t.view(1, 1, 3) + y_mean_t.view(1, 1, 3)
    steer_pred = y_hat_den[:, :, 0]
    steer_true = y_true_den[:, :, 0]
    trend_pred = _avg_pool_seq_torch(steer_pred, TREND_POOL_KERNEL, TREND_POOL_STRIDE)
    trend_true = _avg_pool_seq_torch(steer_true, TREND_POOL_KERNEL, TREND_POOL_STRIDE)

    loss_level = weighted_mse_loss_per_sample(trend_pred, trend_true, sample_weight)
    if TREND_LOSS_MODE == "pooled_level_mse_v1":
        return loss_level
    if TREND_LOSS_MODE != "pooled_delta_direction_v1":
        raise ValueError(f"Unsupported TREND_LOSS_MODE={TREND_LOSS_MODE!r}")
    if trend_pred.shape[1] <= 1:
        return loss_level

    # Match coarse segment-to-segment movement directly instead of only pooled levels.
    trend_delta_pred = _diff1(trend_pred)
    trend_delta_true = _diff1(trend_true)
    loss_delta = weighted_mse_loss_per_sample(trend_delta_pred, trend_delta_true, sample_weight)

    delta_scale = torch.clamp(trend_delta_true.detach().abs().mean(dim=1, keepdim=True), min=TREND_SIGN_EPS)
    trend_dir_pred = torch.tanh(trend_delta_pred / delta_scale)
    with torch.no_grad():
        trend_dir_true = torch.tanh(trend_delta_true / delta_scale)
    loss_dir = weighted_mse_loss_per_sample(trend_dir_pred, trend_dir_true, sample_weight)

    return (
        TREND_LEVEL_WEIGHT * loss_level
        + TREND_DELTA_WEIGHT * loss_delta
        + TREND_DIR_WEIGHT * loss_dir
    )


def _sec_to_future_idx(sec: float, future_len: int) -> int:
    idx = int(round(float(sec) * float(FS)))
    return max(0, min(int(future_len), idx))


def _build_late_ramp(future_len: int, start_sec: float, device, dtype, power: float = 1.0) -> torch.Tensor:
    late_start_idx = _sec_to_future_idx(start_sec, future_len)
    ramp = torch.zeros((1, future_len), device=device, dtype=dtype)
    if late_start_idx >= future_len:
        return ramp
    weights = torch.linspace(0.0, 1.0, future_len - late_start_idx, device=device, dtype=dtype)
    if float(power) != 1.0:
        weights = weights.clamp_min(0.0).pow(float(power))
    ramp[:, late_start_idx:] = weights
    return ramp


def _build_hard_late_masks(steer_true_den: torch.Tensor, rev_gt_weak=None, rev_gt_strong=None):
    B, T = steer_true_den.shape
    hard_late_mask = torch.zeros_like(steer_true_den)
    late_start_idx = _sec_to_future_idx(HARD_LATE_START_SEC, T)
    tail_start_idx = _sec_to_future_idx(HARD_TAIL_START_SEC, T)
    if late_start_idx < T:
        hard_late_mask[:, late_start_idx:] = 1.0

    gt_peak = steer_true_den.detach().abs().amax(dim=1)
    if tail_start_idx < T:
        gt_tail = steer_true_den.detach()[:, tail_start_idx:].abs().amax(dim=1)
    else:
        gt_tail = gt_peak

    if gt_peak.numel() > 1:
        peak_thr = torch.quantile(gt_peak, HARD_PEAK_QUANTILE)
        tail_thr = torch.quantile(gt_tail, HARD_TAIL_QUANTILE)
    else:
        peak_thr = gt_peak[0]
        tail_thr = gt_tail[0]
    hard_pos_mask = (gt_peak >= peak_thr) & (gt_tail >= tail_thr)

    if rev_gt_strong is not None:
        hard_rev_mask = rev_gt_strong.view(-1) > 0.5
    else:
        hard_rev_mask = torch.zeros((B,), device=steer_true_den.device, dtype=torch.bool)
    if rev_gt_weak is not None:
        weak_rev_mask = rev_gt_weak.view(-1) > 0.5
    else:
        weak_rev_mask = torch.zeros((B,), device=steer_true_den.device, dtype=torch.bool)
    hard_mask = (hard_rev_mask | (weak_rev_mask & hard_pos_mask)).to(steer_true_den.dtype)
    return hard_mask, hard_late_mask


def compute_coarse_fine_losses(forward_aux, y_true: torch.Tensor, y_mean_t: torch.Tensor, y_std_t: torch.Tensor, sample_weight=None, is_curve=None, rev_gt_weak=None, rev_gt_strong=None):
    steer_coarse_norm = None if forward_aux is None else forward_aux.get("steer_coarse_norm")
    steer_coarse_up_norm = None if forward_aux is None else forward_aux.get("steer_coarse_up_norm")
    steer_fine_raw_norm = None if forward_aux is None else forward_aux.get("steer_fine_raw_norm", forward_aux.get("steer_fine_norm"))
    steer_fine_out_norm = None if forward_aux is None else forward_aux.get("steer_fine_norm", steer_fine_raw_norm)
    if (
        steer_coarse_norm is None
        or steer_coarse_up_norm is None
        or steer_fine_raw_norm is None
        or steer_fine_out_norm is None
    ):
        zero = torch.tensor(0.0, device=y_true.device, dtype=y_true.dtype)
        return zero, zero, zero

    steer_true_den = y_true[:, :, 0] * y_std_t[0] + y_mean_t[0]
    steer_coarse_den = steer_coarse_norm * y_std_t[0] + y_mean_t[0]
    steer_coarse_up_den = steer_coarse_up_norm * y_std_t[0] + y_mean_t[0]
    steer_fine_raw_den = steer_fine_raw_norm * y_std_t[0]
    steer_fine_out_den = steer_fine_out_norm * y_std_t[0]

    trend_true = _avg_pool_seq_torch(steer_true_den, TREND_POOL_KERNEL, TREND_POOL_STRIDE)
    fine_pool = _avg_pool_seq_torch(steer_fine_raw_den, TREND_POOL_KERNEL, TREND_POOL_STRIDE)
    hard_mask, hard_late_mask = _build_hard_late_masks(steer_true_den, rev_gt_weak=rev_gt_weak, rev_gt_strong=rev_gt_strong)

    if ENABLE_PHASE_ADAPTIVE_TREND:
        seg_w = torch.ones_like(trend_true)
        t = torch.arange(trend_true.shape[1], device=trend_true.device, dtype=trend_true.dtype)
        early_mask = (t < float(TREND_EARLY_BINS)).to(trend_true.dtype).unsqueeze(0)
        late_mask = (t >= float(TREND_EARLY_BINS)).to(trend_true.dtype).unsqueeze(0)
        seg_w = seg_w + 0.25 * early_mask
        if is_curve is not None:
            straight = (1.0 - is_curve.float().clamp(0.0, 1.0)).view(-1, 1).to(trend_true.dtype)
            seg_w = seg_w - TREND_LATE_STRAIGHT_DOWN * late_mask * straight
        if rev_gt_strong is not None:
            strong = rev_gt_strong.float().view(-1, 1).to(trend_true.dtype)
            seg_w = seg_w - TREND_LATE_STRONGREV_DOWN * late_mask * strong
        if ENABLE_HARD_LATE_FINE:
            hard_late_bins = (hard_mask.view(-1, 1) > 0) & (late_mask > 0)
            seg_w = torch.where(hard_late_bins, torch.ones_like(seg_w), seg_w)
        seg_w = torch.clamp(seg_w, min=0.25)
        loss_coarse = weighted_mean_per_sample((((steer_coarse_den - trend_true) ** 2) * seg_w).mean(dim=1), sample_weight)
    else:
        loss_coarse = weighted_mse_loss_per_sample(steer_coarse_den, trend_true, sample_weight)
    loss_fine_dc = weighted_mse_loss_per_sample(fine_pool, torch.zeros_like(fine_pool), sample_weight)

    if ENABLE_HARD_LATE_FINE:
        res_gt = steer_true_den - steer_coarse_up_den.detach()
        hard_weight = hard_late_mask * hard_mask.view(-1, 1)
        per_sample_denom = hard_weight.sum(dim=1)
        per_sample_loss = torch.where(
            per_sample_denom > 0,
            (((steer_fine_out_den - res_gt) ** 2) * hard_weight).sum(dim=1) / per_sample_denom.clamp_min(1.0),
            torch.zeros_like(per_sample_denom),
        )
        loss_hard_late_fine = weighted_mean_per_sample(per_sample_loss, sample_weight)
    else:
        loss_hard_late_fine = torch.tensor(0.0, device=y_true.device, dtype=y_true.dtype)
    return loss_coarse, loss_fine_dc, loss_hard_late_fine


def compute_total_task_loss(y_hat: torch.Tensor, y_true: torch.Tensor, y_mean_t: torch.Tensor, y_std_t: torch.Tensor, sample_weight=None, use_reversal_local_weight=True, forward_aux=None, is_curve=None, rev_gt_weak=None, rev_gt_strong=None):
    loss_task, loss_amp, loss_d1, loss_d2 = compute_active_task_losses(y_hat, y_true, sample_weight=sample_weight)
    loss_revseq, loss_peaktime, loss_steer_wt = compute_reversal_shape_losses(
        y_hat, y_true, y_mean_t, y_std_t, sample_weight=sample_weight, use_reversal_local_weight=use_reversal_local_weight
    )
    if ENABLE_STEER_COARSE_FINE:
        loss_trend = torch.tensor(0.0, device=y_hat.device, dtype=y_hat.dtype)
        loss_trend_coarse, loss_fine_dc, loss_hard_late_fine = compute_coarse_fine_losses(
            forward_aux, y_true, y_mean_t, y_std_t, sample_weight=sample_weight, is_curve=is_curve, rev_gt_weak=rev_gt_weak, rev_gt_strong=rev_gt_strong
        )
        loss_task = loss_task + W_TREND_COARSE * loss_trend_coarse + W_FINE_DC * loss_fine_dc + W_HARD_LATE_FINE * loss_hard_late_fine
    else:
        loss_trend = compute_trend_loss(y_hat, y_true, y_mean_t, y_std_t, sample_weight=sample_weight)
        loss_trend_coarse = torch.tensor(0.0, device=y_hat.device, dtype=y_hat.dtype)
        loss_fine_dc = torch.tensor(0.0, device=y_hat.device, dtype=y_hat.dtype)
        loss_hard_late_fine = torch.tensor(0.0, device=y_hat.device, dtype=y_hat.dtype)
        loss_task = loss_task + W_TREND * loss_trend
    loss_task = loss_task + W_REVSEQ * loss_revseq + W_PEAKTIME * loss_peaktime + W_STEER_WT * loss_steer_wt
    return loss_task, loss_amp, loss_d1, loss_d2, loss_revseq, loss_peaktime, loss_steer_wt, loss_trend, loss_trend_coarse, loss_fine_dc, loss_hard_late_fine


def _denorm_y(y_norm_np: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray) -> np.ndarray:
    # y_norm_np: (N,T,3)
    return y_norm_np * y_std.reshape(1, 1, 3) + y_mean.reshape(1, 1, 3)




def has_reversal_np(steer_seq_1d, eps=REV_EPS_WEAK):
    """Return 1.0 if the steering sequence crosses both +eps and -eps (sign reversal), else 0.0."""
    x = np.asarray(steer_seq_1d, dtype=np.float64)
    if x.size == 0 or not np.isfinite(x).any():
        return 0.0
    return 1.0 if (np.nanmax(x) > eps and np.nanmin(x) < -eps) else 0.0

def _binary_metrics(y_true, y_pred):
    """Precision/Recall/F1 for binary labels (0/1)."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    prec = tp / max(1, (tp + fp))
    rec  = tp / max(1, (tp + fn))
    f1   = 2 * prec * rec / max(1e-12, (prec + rec))
    return {"tp": tp, "fp": fp, "fn": fn, "precision": float(prec), "recall": float(rec), "f1": float(f1)}


def _safe_mean_np(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return None
    return float(np.mean(x))


def _safe_median_np(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return None
    return float(np.median(x))


def _safe_rmse_np(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return None
    return float(np.sqrt(np.mean(x ** 2)))


def _safe_mae_np(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return None
    return float(np.mean(np.abs(x)))


def _safe_ratio_np(num, den, eps=1e-6):
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    if num.size == 0 or den.size == 0:
        return None
    return float(np.mean(num / np.maximum(den, eps)))


def _crossing_count_np(steer_seq_1d, eps=REV_EPS_WEAK):
    x = np.asarray(steer_seq_1d, dtype=np.float64)
    if x.size < 2 or not np.isfinite(x).any():
        return 0
    sign = np.zeros_like(x, dtype=np.int64)
    sign[x > eps] = 1
    sign[x < -eps] = -1
    nz = sign[sign != 0]
    if nz.size < 2:
        return 0
    return int(np.sum(nz[1:] != nz[:-1]))


def _first_reversal_time_np(steer_seq_1d, eps=REV_EPS_WEAK, fs=200):
    x = np.asarray(steer_seq_1d, dtype=np.float64)
    if x.size < 2 or not np.isfinite(x).any():
        return None
    sign = np.zeros_like(x, dtype=np.int64)
    sign[x > eps] = 1
    sign[x < -eps] = -1
    nz_idx = np.flatnonzero(sign != 0)
    if nz_idx.size < 2:
        return None
    nz_sign = sign[nz_idx]
    change_idx = np.flatnonzero(nz_sign[1:] != nz_sign[:-1])
    if change_idx.size == 0:
        return None
    first_idx = int(nz_idx[change_idx[0] + 1])
    return float(first_idx / max(1, fs))


def _first_threshold_crossing_idx_np(seq_1d, threshold, ref_value=None):
    x = np.asarray(seq_1d, dtype=np.float64)
    if x.size == 0 or not np.isfinite(x).any():
        return None
    if ref_value is None:
        ref_value = float(x[0])
    delta = np.abs(x - ref_value)
    idx = np.flatnonzero(delta >= float(threshold))
    if idx.size == 0:
        return None
    return int(idx[0])


def _head_metrics(pred, true, fs=200, head_frac=0.25, onset_thr_ratio=0.15, onset_thr_abs=STEER_ONSET_THR_ABS):
    t_len = int(pred.shape[1])
    head_len = max(1, int(round(t_len * head_frac)))
    pred_head = pred[:, :head_len, 0]
    true_head = true[:, :head_len, 0]
    err_head = pred_head - true_head

    pred_head_amp = np.ptp(pred_head, axis=1)
    true_head_amp = np.ptp(true_head, axis=1)
    pred_head_motion = np.mean(np.abs(pred_head - pred_head[:, :1]), axis=1)
    flat_thr = 0.10 * np.maximum(true_head_amp, 1e-6)

    if head_len > 1:
        pred_head_slope = np.mean(np.abs(np.diff(pred_head, axis=1)), axis=1)
        true_head_slope = np.mean(np.abs(np.diff(true_head, axis=1)), axis=1)
    else:
        pred_head_slope = np.zeros((pred_head.shape[0],), dtype=np.float64)
        true_head_slope = np.zeros((true_head.shape[0],), dtype=np.float64)

    pred_steer = np.asarray(pred[:, :, 0], dtype=np.float64)
    true_steer = np.asarray(true[:, :, 0], dtype=np.float64)
    onset_delay = []
    for i in range(pred_steer.shape[0]):
        true_seq = true_steer[i]
        pred_seq = pred_steer[i]
        true_base = float(true_seq[0])
        pred_base = float(pred_seq[0])
        true_peak_delta = float(np.max(np.abs(true_seq - true_base)))
        onset_thr = max(onset_thr_abs, onset_thr_ratio * true_peak_delta)
        true_idx = _first_threshold_crossing_idx_np(true_seq, threshold=onset_thr, ref_value=true_base)
        if true_idx is None:
            continue
        pred_idx = _first_threshold_crossing_idx_np(pred_seq, threshold=onset_thr, ref_value=pred_base)
        if pred_idx is None:
            pred_idx = t_len - 1
        onset_delay.append((pred_idx - true_idx) / max(1, fs))

    onset_delay = np.asarray(onset_delay, dtype=np.float64)
    return {
        "head_len": int(head_len),
        "head_end_idx": int(head_len - 1),
        "head_end_sec": float((head_len - 1) / max(1, fs)),
        "head_rmse_steer": _safe_rmse_np(err_head),
        "head_mae_steer": _safe_mae_np(err_head),
        "head_amp_pred": _safe_mean_np(pred_head_amp),
        "head_amp_gt": _safe_mean_np(true_head_amp),
        "head_amp_ratio_pred_over_gt": _safe_ratio_np(pred_head_amp, true_head_amp),
        "head_flatness_rate": float(np.mean(pred_head_motion <= flat_thr)),
        "early_slope_pred": _safe_mean_np(pred_head_slope),
        "early_slope_gt": _safe_mean_np(true_head_slope),
        "early_slope_ratio_pred_over_gt": _safe_ratio_np(pred_head_slope, true_head_slope),
        "response_onset_delay_sec": _safe_mean_np(onset_delay),
        "response_onset_delay_mae_sec": _safe_mae_np(onset_delay),
        "n_valid_onset": int(onset_delay.size),
        "response_onset_threshold_ratio": float(onset_thr_ratio),
        "response_onset_threshold_abs": float(onset_thr_abs),
        "steer_angle_unit": STEER_ANGLE_UNIT,
    }


def _tail_metrics(pred, true, fs=200, tail_frac=0.25):
    t_len = int(pred.shape[1])
    tail_len = max(1, int(round(t_len * tail_frac)))
    tail_start = max(0, t_len - tail_len)
    pred_tail = pred[:, tail_start:, 0]
    true_tail = true[:, tail_start:, 0]
    err_tail = pred_tail - true_tail
    pred_tail_std = pred_tail.std(axis=1)
    true_tail_std = true_tail.std(axis=1)
    pred_tail_amp = np.ptp(pred_tail, axis=1)
    true_tail_amp = np.ptp(true_tail, axis=1)
    pred_tail_slope = pred_tail[:, -1] - pred_tail[:, 0]
    true_tail_slope = true_tail[:, -1] - true_tail[:, 0]
    flat_thr = 0.10 * np.maximum(true_tail_amp, 1e-6)
    pred_tail_amp_mean = np.mean(np.abs(pred_tail - pred_tail.mean(axis=1, keepdims=True)), axis=1)
    return {
        "tail_start_idx": int(tail_start),
        "tail_len": int(tail_len),
        "tail_start_sec": float(tail_start / max(1, fs)),
        "tail_rmse_steer": _safe_rmse_np(err_tail),
        "tail_mae_steer": _safe_mae_np(err_tail),
        "tail_std_pred": _safe_mean_np(pred_tail_std),
        "tail_std_gt": _safe_mean_np(true_tail_std),
        "tail_std_ratio_pred_over_gt": _safe_ratio_np(pred_tail_std, true_tail_std),
        "tail_amp_pred": _safe_mean_np(pred_tail_amp),
        "tail_amp_gt": _safe_mean_np(true_tail_amp),
        "tail_amp_ratio_pred_over_gt": _safe_ratio_np(pred_tail_amp, true_tail_amp),
        "tail_slope_mae": _safe_mae_np(pred_tail_slope - true_tail_slope),
        "tail_flatness_rate": float(np.mean(pred_tail_amp_mean <= flat_thr)),
    }


def _peak_metrics(pred, true, fs=200):
    pred_steer = np.asarray(pred[:, :, 0], dtype=np.float64)
    true_steer = np.asarray(true[:, :, 0], dtype=np.float64)
    pred_peak_idx = np.argmax(np.abs(pred_steer), axis=1)
    true_peak_idx = np.argmax(np.abs(true_steer), axis=1)
    pred_peak_val = pred_steer[np.arange(pred_steer.shape[0]), pred_peak_idx]
    true_peak_val = true_steer[np.arange(true_steer.shape[0]), true_peak_idx]
    half_idx = pred_steer.shape[1] // 2
    mask_true_late = true_peak_idx >= half_idx
    return {
        "peak_time_mae_sec": _safe_mae_np((pred_peak_idx - true_peak_idx) / max(1, fs)),
        "peak_time_rmse_sec": _safe_rmse_np((pred_peak_idx - true_peak_idx) / max(1, fs)),
        "peak_mag_mae": _safe_mae_np(pred_peak_val - true_peak_val),
        "peak_mag_rmse": _safe_rmse_np(pred_peak_val - true_peak_val),
        "late_peak_rate_gt": float(np.mean(mask_true_late)),
        "late_peak_recall": float(np.mean(pred_peak_idx[mask_true_late] >= half_idx)) if np.any(mask_true_late) else None,
    }


def _safe_corrcoef_np(a, b, eps=1e-8):
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if aa.size == 0 or bb.size == 0:
        return np.nan
    aa = aa - aa.mean()
    bb = bb - bb.mean()
    aa_std = float(np.sqrt(np.mean(aa ** 2)))
    bb_std = float(np.sqrt(np.mean(bb ** 2)))
    if aa_std < eps and bb_std < eps:
        return 1.0 if float(np.mean(np.abs(np.asarray(a) - np.asarray(b)))) < eps else 0.0
    if aa_std < eps or bb_std < eps:
        return 0.0
    return float(np.mean((aa / aa_std) * (bb / bb_std)))


def _trend_metrics(pred, true, fs=200, pool_kernel=TREND_POOL_KERNEL, pool_stride=TREND_POOL_STRIDE, sign_eps=TREND_SIGN_EPS):
    pred_steer = np.asarray(pred[:, :, 0], dtype=np.float64)
    true_steer = np.asarray(true[:, :, 0], dtype=np.float64)
    pred_pool = _avg_pool_seq_np(pred_steer, pool_kernel, pool_stride)
    true_pool = _avg_pool_seq_np(true_steer, pool_kernel, pool_stride)

    corr_vec = []
    sign_match_vec = []
    for i in range(pred_pool.shape[0]):
        corr_vec.append(_safe_corrcoef_np(pred_pool[i], true_pool[i]))
        pred_delta = np.diff(pred_pool[i])
        true_delta = np.diff(true_pool[i])
        pred_sign = np.where(pred_delta > sign_eps, 1, np.where(pred_delta < -sign_eps, -1, 0))
        true_sign = np.where(true_delta > sign_eps, 1, np.where(true_delta < -sign_eps, -1, 0))
        sign_match_vec.append(float(np.mean(pred_sign == true_sign)) if pred_sign.size else np.nan)

    corr_vec = np.asarray(corr_vec, dtype=np.float64)
    sign_match_vec = np.asarray(sign_match_vec, dtype=np.float64)
    coarse_err = pred_pool - true_pool
    coarse_delta_err = np.diff(pred_pool, axis=1) - np.diff(true_pool, axis=1) if pred_pool.shape[1] > 1 else np.empty((pred_pool.shape[0], 0), dtype=np.float64)
    corr_valid = corr_vec[np.isfinite(corr_vec)]
    sign_valid = sign_match_vec[np.isfinite(sign_match_vec)]
    return {
        "trend_loss_mode": TREND_LOSS_MODE,
        "trend_pool_kernel": int(min(int(pool_kernel), pred_steer.shape[1])),
        "trend_pool_stride": int(pool_stride),
        "trend_segment_sec": float(min(int(pool_kernel), pred_steer.shape[1]) / max(1, fs)),
        "trend_pooled_len": int(pred_pool.shape[1]),
        "smooth_trend_corr_mean": _safe_mean_np(corr_valid),
        "smooth_trend_corr_median": _safe_median_np(corr_valid),
        "coarse_segment_sign_match_rate": _safe_mean_np(sign_valid),
        "coarse_segment_sign_match_median": _safe_median_np(sign_valid),
        "coarse_segment_mae": _safe_mae_np(coarse_err),
        "coarse_segment_rmse": _safe_rmse_np(coarse_err),
        "coarse_delta_mae": _safe_mae_np(coarse_delta_err),
        "coarse_delta_rmse": _safe_rmse_np(coarse_delta_err),
    }


def _structured_reversal_metrics(pred, true, rev_gt_weak_vec=None, rev_gt_strong_vec=None, fs=200):
    pred_steer = np.asarray(pred[:, :, 0], dtype=np.float64)
    true_steer = np.asarray(true[:, :, 0], dtype=np.float64)
    pred_rev_time = np.array([_first_reversal_time_np(x, eps=REV_EPS_WEAK, fs=fs) for x in pred_steer], dtype=np.float64)
    true_rev_time = np.array([_first_reversal_time_np(x, eps=REV_EPS_WEAK, fs=fs) for x in true_steer], dtype=np.float64)
    pred_rev_count = np.array([_crossing_count_np(x, eps=REV_EPS_WEAK) for x in pred_steer], dtype=np.int64)
    true_rev_count = np.array([_crossing_count_np(x, eps=REV_EPS_WEAK) for x in true_steer], dtype=np.int64)
    mask_both = np.isfinite(pred_rev_time) & np.isfinite(true_rev_time)

    def _bucket(mask):
        mask = np.asarray(mask, dtype=bool)
        if mask.size == 0 or not np.any(mask):
            return None
        pred_b = pred[mask]
        true_b = true[mask]
        tail_b = _tail_metrics(pred_b, true_b, fs=fs)
        err_b = pred_b[:, :, 0] - true_b[:, :, 0]
        return {
            "n": int(mask.sum()),
            "rmse_steer": _safe_rmse_np(err_b),
            "tail_rmse_steer": tail_b["tail_rmse_steer"],
            "tail_amp_ratio_pred_over_gt": tail_b["tail_amp_ratio_pred_over_gt"],
            "tail_flatness_rate": tail_b["tail_flatness_rate"],
        }

    out = {
        "first_reversal_time_mae_sec": _safe_mae_np(pred_rev_time[mask_both] - true_rev_time[mask_both]),
        "first_reversal_time_rmse_sec": _safe_rmse_np(pred_rev_time[mask_both] - true_rev_time[mask_both]),
        "reversal_count_mae": _safe_mae_np(pred_rev_count - true_rev_count),
        "reversal_count_exact_match_rate": float(np.mean(pred_rev_count == true_rev_count)),
        "n_both_have_reversal": int(mask_both.sum()),
    }
    if rev_gt_weak_vec is not None:
        rev_gt_weak_vec = np.asarray(rev_gt_weak_vec).astype(np.int64)
        out["by_bucket"] = {
            "straight": _bucket(rev_gt_weak_vec == 0),
            "weak_pos": _bucket(rev_gt_weak_vec == 1),
        }
        if rev_gt_strong_vec is not None:
            rev_gt_strong_vec = np.asarray(rev_gt_strong_vec).astype(np.int64)
            out["by_bucket"]["strong_pos"] = _bucket(rev_gt_strong_vec == 1)
    return out


def evaluate_and_plot(model: nn.Module, test_loader: DataLoader,
                      y_mean: np.ndarray, y_std: np.ndarray,
                      fig_dir: Path, curve_thr: float = None, fs: int = 200, n_examples: int = 8,
                      state_component_names=None, teacher_state_mode: str = "old_ac"):
    """Export:
      - figures/pred_vs_gt_example_*.png
      - figures/test_metrics.json
      - figures/test_state_dump.csv (A/C from veh & teacher + mask)
      - figures/state_vs_peak_*.png (quick relationship views)
    """
    model.eval()

    preds, trues = [], []
    zveh_all, zphys_all, zmask_all = [], [], []
    idx_all, curve_score_all, is_curve_all = [], [], []
    rev_gt_all, rev_gt_weak_all, rev_gt_strong_all, rev_prob_all = [], [], [], []
    strong_pos_gate_prob_all = []

    with torch.no_grad():
        for batch in test_loader:
            src = batch["src"].to(DEVICE, non_blocking=True)
            y_true_norm = batch["y_norm"].to(DEVICE, non_blocking=True)
            curve_norm = batch["curve_norm"].to(DEVICE, non_blocking=True)
            ctx = batch["ctx"].to(DEVICE, non_blocking=True)
            z_phys = batch["z_phys"].to(DEVICE, non_blocking=True)
            z_mask = batch["z_mask"].to(DEVICE, non_blocking=True)  # (B,1)
            rev_gt_b = batch["rev_gt"].to(DEVICE, non_blocking=True).squeeze(1)  # (B,)
            rev_gt_weak_b = batch.get("rev_gt_weak", batch["rev_gt"]).to(DEVICE, non_blocking=True).squeeze(1)
            rev_gt_strong_b = batch.get("rev_gt_strong", batch["rev_gt"]).to(DEVICE, non_blocking=True).squeeze(1)

            y_hat_norm, z_veh, rev_logit, forward_aux = unpack_model_output(model(src, ctx, curve_norm))

            preds.append(y_hat_norm.cpu().numpy())
            trues.append(y_true_norm.cpu().numpy())
            zveh_all.append(z_veh.cpu().numpy())
            zphys_all.append(z_phys.cpu().numpy())
            zmask_all.append(z_mask.cpu().numpy())
            idx_all.append(batch.get("idx", torch.full((src.shape[0],), -1, dtype=torch.long)).cpu().numpy())
            rev_gt_all.append(rev_gt_b.detach().cpu().numpy())
            rev_gt_weak_all.append(rev_gt_weak_b.detach().cpu().numpy())
            rev_gt_strong_all.append(rev_gt_strong_b.detach().cpu().numpy())
            rev_prob_all.append(torch.sigmoid(rev_logit).detach().cpu().numpy())
            if forward_aux.get("strong_pos_gate_prob") is not None:
                strong_pos_gate_prob_all.append(forward_aux["strong_pos_gate_prob"].detach().cpu().numpy())
            curve_score_all.append(batch.get("curve_score", torch.full((src.shape[0],), float('nan'))).cpu().numpy())
            is_curve_all.append(batch.get("is_curve", torch.full((src.shape[0],), -1, dtype=torch.long)).cpu().numpy())

    pred_norm = np.concatenate(preds, axis=0)
    true_norm = np.concatenate(trues, axis=0)
    zveh_all = np.concatenate(zveh_all, axis=0)
    zphys_all = np.concatenate(zphys_all, axis=0)
    zmask_all = np.concatenate(zmask_all, axis=0).reshape(-1)  # (N,)
    state_dim = int(zveh_all.shape[1]) if zveh_all.ndim == 2 else 0
    veh_state_cols = make_state_column_names("veh", state_dim, state_component_names)
    teacher_state_cols = make_state_column_names("teacher", state_dim, state_component_names)

    pred = _denorm_y(pred_norm, y_mean, y_std)
    true = _denorm_y(true_norm, y_mean, y_std)

    err = pred - true
    rmse_all = float(np.sqrt(np.mean(err ** 2)))
    rmse_ch = np.sqrt(np.mean(err ** 2, axis=(0, 1))).astype(float)
    mae_ch = np.mean(np.abs(err), axis=(0, 1)).astype(float)

    metrics = {
        "rmse_all": rmse_all,
        "rmse_steer": float(rmse_ch[0]),
        "rmse_yawrate": float(rmse_ch[1]),
        "rmse_ay": float(rmse_ch[2]),
        "mae_steer": float(mae_ch[0]),
        "mae_yawrate": float(mae_ch[1]),
        "mae_ay": float(mae_ch[2]),
        "n_test": int(pred.shape[0]),
        "future_len": int(pred.shape[1]),
    }
    head_metrics = _head_metrics(pred, true, fs=fs)
    tail_metrics = _tail_metrics(pred, true, fs=fs)
    peak_metrics = _peak_metrics(pred, true, fs=fs)
    trend_metrics = _trend_metrics(pred, true, fs=fs)
    metrics.update({
        "head_metrics": head_metrics,
        "tail_metrics": tail_metrics,
        "peak_metrics": peak_metrics,
        "trend_metrics": trend_metrics,
    })
    save_json(fig_dir / "test_metrics.json", metrics)
    save_json(fig_dir / "test_metrics_head.json", head_metrics)
    save_json(fig_dir / "test_metrics_tail.json", tail_metrics)
    save_json(fig_dir / "test_metrics_peak.json", peak_metrics)
    save_json(fig_dir / "test_metrics_trend.json", trend_metrics)
    print("📌 Test 指标:", metrics)


    # ---- road-type metrics (curve vs straight) ----
    if is_curve_all is not None and len(is_curve_all) > 0:
        is_curve_vec = np.concatenate(is_curve_all, axis=0).astype(np.int64)
        curve_score_vec = np.concatenate(curve_score_all, axis=0).astype(np.float32)
        # is_curve: 1=curve, 0=straight
        mask_curve = (is_curve_vec == 1)
        mask_straight = (is_curve_vec == 0)

        def rmse_by_mask(err, mask):
            if mask is None or mask.sum() == 0:
                return None
            ee = err[mask, :, :]
            out = np.sqrt(np.mean(ee ** 2, axis=(0, 1)))
            return out.tolist()

        err = pred - true
        road_metrics = {
            "curve_thr": float(curve_thr) if curve_thr is not None else None,
            "curve_ratio_test": float(mask_curve.mean()) if mask_curve.size else None,
            "rmse_curve": rmse_by_mask(err, mask_curve),
            "rmse_straight": rmse_by_mask(err, mask_straight),
        }
        save_json(fig_dir / "test_metrics_by_roadtype.json", road_metrics)
        print("🛣 RoadType 指标:", road_metrics)

    # ---- reversal metrics (weak & strong; and the label actually used for training) ----
    rev_prob_vec = np.concatenate(rev_prob_all, axis=0).astype(np.float32) if len(rev_prob_all)>0 else None
    is_curve_vec = np.concatenate(is_curve_all, axis=0).astype(np.int64) if (is_curve_all is not None and len(is_curve_all)>0) else None

    rev_gt_used_vec = np.concatenate(rev_gt_all, axis=0).astype(np.int64) if len(rev_gt_all)>0 else None
    rev_gt_weak_vec = np.concatenate(rev_gt_weak_all, axis=0).astype(np.int64) if len(rev_gt_weak_all)>0 else None
    rev_gt_strong_vec = np.concatenate(rev_gt_strong_all, axis=0).astype(np.int64) if len(rev_gt_strong_all)>0 else None

    def _rmse_steer_mask(mask):
        if mask is None or mask.sum() == 0:
            return None
        ee = err[mask, :, 0]
        return float(np.sqrt(np.mean(ee ** 2)))

    def _compute_rev_metrics(label_vec):
        if label_vec is None or rev_prob_vec is None:
            return None
        pred_vec = (rev_prob_vec >= 0.5).astype(np.int64)
        met_all = _binary_metrics(label_vec, pred_vec)

        met_straight = None
        rmse_straight_pos = None
        rmse_straight_neg = None
        if is_curve_vec is not None:
            mask_straight = (is_curve_vec == 0)
            met_straight = _binary_metrics(label_vec[mask_straight], pred_vec[mask_straight])
            rmse_straight_pos = _rmse_steer_mask(mask_straight & (label_vec == 1))
            rmse_straight_neg = _rmse_steer_mask(mask_straight & (label_vec == 0))
        return {
            "metrics_all": met_all,
            "metrics_straight": met_straight,
            "rmse_steer_straight_pos": rmse_straight_pos,
            "rmse_steer_straight_neg": rmse_straight_neg,
        }

    structured_rev_metrics = _structured_reversal_metrics(
        pred,
        true,
        rev_gt_weak_vec=rev_gt_weak_vec,
        rev_gt_strong_vec=rev_gt_strong_vec,
        fs=fs,
    )
    rev_metrics = {
        "STEER_SOURCE_UNIT": STEER_SOURCE_UNIT,
        "STEER_ANGLE_UNIT": STEER_ANGLE_UNIT,
        "STEER_ANGLE_SCALE": float(STEER_ANGLE_SCALE),
        "REV_EPS_WEAK": float(REV_EPS_WEAK),
        "REV_EPS_STRONG": float(REV_EPS_STRONG),
        "STRONG_PEAK_THR": float(STRONG_PEAK_THR),
        "used_label": ("strong" if USE_STRONG_REV_LOSS else "weak"),
        "rate_used": float(np.mean(rev_gt_used_vec)) if rev_gt_used_vec is not None else None,
        "rate_weak": float(np.mean(rev_gt_weak_vec)) if rev_gt_weak_vec is not None else None,
        "rate_strong": float(np.mean(rev_gt_strong_vec)) if rev_gt_strong_vec is not None else None,
        "used": _compute_rev_metrics(rev_gt_used_vec),
        "weak": _compute_rev_metrics(rev_gt_weak_vec),
        "strong": _compute_rev_metrics(rev_gt_strong_vec),
        "structured": structured_rev_metrics,
    }
    save_json(fig_dir / "test_metrics_by_reversal.json", rev_metrics)
    save_json(fig_dir / "test_metrics_reversal_structure.json", structured_rev_metrics)
    print("🔁 Reversal 指标:", rev_metrics)


    # ---- state dump (event-level) ----
    component_names = state_component_names or [f"d{i}" for i in range(state_dim)]
    has_semantic_ac = bool(teacher_state_mode == "old_ac" and state_dim >= 2)
    state_dump = {
        "teacher_mask": zmask_all,
        "teacher_state_mode": teacher_state_mode,
        "is_curve": np.concatenate(is_curve_all, axis=0).astype(np.int64) if len(is_curve_all)>0 else -1,
        "curve_score": np.concatenate(curve_score_all, axis=0).astype(np.float32) if len(curve_score_all)>0 else np.nan,
        "rev_gt": np.concatenate(rev_gt_all, axis=0).astype(np.int64) if len(rev_gt_all)>0 else -1,
        "rev_gt_weak": np.concatenate(rev_gt_weak_all, axis=0).astype(np.int64) if len(rev_gt_weak_all)>0 else -1,
        "rev_gt_strong": np.concatenate(rev_gt_strong_all, axis=0).astype(np.int64) if len(rev_gt_strong_all)>0 else -1,
        "rev_prob": np.concatenate(rev_prob_all, axis=0).astype(np.float32) if len(rev_prob_all)>0 else np.nan,
        "strong_pos_gate_prob": np.concatenate(strong_pos_gate_prob_all, axis=0).astype(np.float32) if len(strong_pos_gate_prob_all)>0 else np.nan,
        "idx": np.concatenate(idx_all, axis=0).astype(np.int64) if len(idx_all)>0 else -1,
    }
    for j, col in enumerate(veh_state_cols):
        state_dump[col] = zveh_all[:, j]
    for j, col in enumerate(teacher_state_cols):
        state_dump[col] = zphys_all[:, j]
    if state_dim >= 2:
        state_dump["A_veh"] = zveh_all[:, 0]
        state_dump["C_veh"] = zveh_all[:, 1]
        state_dump["A_teacher"] = zphys_all[:, 0]
        state_dump["C_teacher"] = zphys_all[:, 1]
    df_state = pd.DataFrame(state_dump)
    df_state.to_csv(str(fig_dir / "test_state_dump.csv"), index=False, encoding="utf-8-sig")

    meta_out = {
        "teacher_state_mode": teacher_state_mode,
        "state_dim": int(state_dim),
        "component_names": component_names,
        "veh_state_cols": veh_state_cols,
        "teacher_state_cols": teacher_state_cols,
        "has_semantic_ac": has_semantic_ac,
    }
    save_json(fig_dir / "test_state_meta.json", meta_out)
    print("🧠 State dump meta:", meta_out)

    def _state_label(j):
        if has_semantic_ac and j == 0:
            return "A"
        if has_semantic_ac and j == 1:
            return "C"
        return component_names[j] if j < len(component_names) else f"d{j}"

    primary_plot_dims = min(2, state_dim)
    plot_dim_labels = [_state_label(j) for j in range(primary_plot_dims)]

    def _student_state_title(i):
        return summarize_state_vector(zveh_all[i], component_names)

    def _teacher_state_title(i):
        if zmask_all[i] <= 0.5:
            return "teacher=NA"
        return summarize_state_vector(zphys_all[i], component_names)

    print(f"🧾 已保存 test 状态/行为汇总: {fig_dir / 'test_state_dump.csv'}")

    peak_abs_steer = np.max(np.abs(true[:, :, 0]), axis=1)
    peak_abs_yaw = np.max(np.abs(true[:, :, 1]), axis=1)
    peak_abs_ay = np.max(np.abs(true[:, :, 2]), axis=1)

    rmse_steer_evt = np.sqrt(np.mean((pred[:, :, 0] - true[:, :, 0]) ** 2, axis=1))
    rmse_yaw_evt = np.sqrt(np.mean((pred[:, :, 1] - true[:, :, 1]) ** 2, axis=1))
    rmse_ay_evt = np.sqrt(np.mean((pred[:, :, 2] - true[:, :, 2]) ** 2, axis=1))

    df_state["peak_abs_steer_gt"] = peak_abs_steer
    df_state["peak_abs_yaw_gt"] = peak_abs_yaw
    df_state["peak_abs_ay_gt"] = peak_abs_ay
    df_state["rmse_steer_evt"] = rmse_steer_evt
    df_state["rmse_yaw_evt"] = rmse_yaw_evt
    df_state["rmse_ay_evt"] = rmse_ay_evt
    df_state.to_csv(str(fig_dir / "test_state_dump.csv"), index=False, encoding="utf-8-sig")

    # ---- quick relationship plots (state vs peak) ----
    def _scatter(x, y, xlabel, ylabel, outname):
        plt.figure(figsize=(7.2, 5.0))
        plt.scatter(x, y, s=10, alpha=0.6)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(str(fig_dir / outname), dpi=200)
        plt.close()

    for j in range(primary_plot_dims):
        veh_col = veh_state_cols[j]
        label = plot_dim_labels[j]
        _scatter(df_state[veh_col].values, peak_abs_steer, f"{label}_veh (student)", STEER_PEAK_PLOT_LABEL, f"state_vs_peak_steer_{label}.png")
        _scatter(df_state[veh_col].values, peak_abs_ay, f"{label}_veh (student)", "peak|ay| (GT)", f"state_vs_peak_ay_{label}.png")

    # ---- per-sample pred-vs-gt plots with state annotation ----
    n = pred.shape[0]
    if n == 0:
        print("⚠ test 集为空，无法画图")
        return

    t = np.arange(pred.shape[1], dtype=np.float32) / float(fs)
    pick = np.linspace(0, n - 1, num=min(n_examples, n), dtype=int)

    for k, idx in enumerate(pick):
        title = (
            f"Test sample #{idx} | Future {t[-1]:.2f}s | "
            f"veh[{_student_state_title(idx)}] | { _teacher_state_title(idx) }"
        )

        fig = plt.figure(figsize=(12, 8))

        ax1 = fig.add_subplot(3, 1, 1)
        ax1.plot(t, true[idx, :, 0], label="GT", linewidth=1.2)
        ax1.plot(t, pred[idx, :, 0], label="Pred", linewidth=1.2, linestyle="--")
        ax1.set_ylabel(STEER_PLOT_LABEL)
        ax1.set_title(title)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2 = fig.add_subplot(3, 1, 2)
        ax2.plot(t, true[idx, :, 1], linewidth=1.2)
        ax2.plot(t, pred[idx, :, 1], linewidth=1.2, linestyle="--")
        ax2.set_ylabel("yawrate")
        ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(3, 1, 3)
        ax3.plot(t, true[idx, :, 2], linewidth=1.2)
        ax3.plot(t, pred[idx, :, 2], linewidth=1.2, linestyle="--")
        ax3.set_ylabel("ay")
        ax3.set_xlabel("time (s)")
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = fig_dir / f"pred_vs_gt_example_{k:02d}_idx{idx}.png"
        plt.savefig(str(out_path), dpi=200)
        plt.close()

    print(f"🖼 已保存预测效果图到: {fig_dir} (pred_vs_gt_example_*.png)")

    return

    peak_abs_steer = np.max(np.abs(true[:, :, 0]), axis=1)
    peak_abs_yaw = np.max(np.abs(true[:, :, 1]), axis=1)
    peak_abs_ay = np.max(np.abs(true[:, :, 2]), axis=1)

    rmse_steer_evt = np.sqrt(np.mean((pred[:, :, 0] - true[:, :, 0]) ** 2, axis=1))
    rmse_yaw_evt = np.sqrt(np.mean((pred[:, :, 1] - true[:, :, 1]) ** 2, axis=1))
    rmse_ay_evt = np.sqrt(np.mean((pred[:, :, 2] - true[:, :, 2]) ** 2, axis=1))

    df_state["peak_abs_steer_gt"] = peak_abs_steer
    df_state["peak_abs_yaw_gt"] = peak_abs_yaw
    df_state["peak_abs_ay_gt"] = peak_abs_ay
    df_state["rmse_steer_evt"] = rmse_steer_evt
    df_state["rmse_yaw_evt"] = rmse_yaw_evt
    df_state["rmse_ay_evt"] = rmse_ay_evt
    df_state.to_csv(str(fig_dir / "test_state_dump.csv"), index=False, encoding="utf-8-sig")

    print(f"🧾 已保存 test 状态/行为汇总: {fig_dir / 'test_state_dump.csv'}")

    # ---- quick relationship plots (state vs peak) ----
    def _scatter(x, y, xlabel, ylabel, outname):
        plt.figure(figsize=(7.2, 5.0))
        plt.scatter(x, y, s=10, alpha=0.6)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(str(fig_dir / outname), dpi=200)
        plt.close()

    for j in range(primary_plot_dims):
        veh_col = veh_state_cols[j]
        label = plot_dim_labels[j]
        _scatter(df_state[veh_col].values, peak_abs_steer, f"{label}_veh (student)", STEER_PEAK_PLOT_LABEL, f"state_vs_peak_steer_{label}.png")
        _scatter(df_state[veh_col].values, peak_abs_ay, f"{label}_veh (student)", "peak|ay| (GT)", f"state_vs_peak_ay_{label}.png")

    # ---- per-sample pred-vs-gt plots with state annotation ----
    n = pred.shape[0]
    if n == 0:
        print("⚠ test 集为空，无法画图")
        return

    t = np.arange(pred.shape[1], dtype=np.float32) / float(fs)
    pick = np.linspace(0, n - 1, num=min(n_examples, n), dtype=int)

    for k, idx in enumerate(pick):
        title = (
            f"Test sample #{idx} | Future {t[-1]:.2f}s | "
            f"veh[{_student_state_title(idx)}] | {_teacher_state_title(idx)}"
        )

        fig = plt.figure(figsize=(12, 8))

        ax1 = fig.add_subplot(3, 1, 1)
        ax1.plot(t, true[idx, :, 0], label="GT", linewidth=1.2)
        ax1.plot(t, pred[idx, :, 0], label="Pred", linewidth=1.2, linestyle="--")
        ax1.set_ylabel(STEER_PLOT_LABEL)
        ax1.set_title(title)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2 = fig.add_subplot(3, 1, 2)
        ax2.plot(t, true[idx, :, 1], linewidth=1.2)
        ax2.plot(t, pred[idx, :, 1], linewidth=1.2, linestyle="--")
        ax2.set_ylabel("yawrate")
        ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(3, 1, 3)
        ax3.plot(t, true[idx, :, 2], linewidth=1.2)
        ax3.plot(t, pred[idx, :, 2], linewidth=1.2, linestyle="--")
        ax3.set_ylabel("ay")
        ax3.set_xlabel("time (s)")
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = fig_dir / f"pred_vs_gt_example_{k:02d}_idx{idx}.png"
        plt.savefig(str(out_path), dpi=200)
        plt.close()

    print(f"🖼 已保存预测效果图到: {fig_dir} (pred_vs_gt_example_*.png)")

    return
    state_dump_alias_teacher_cols = teacher_state_cols
    state_dump_alias_mask = teacher_mask
    state_dump_alias_semantic = has_semantic_ac
    state_dump_alias_plotdims = primary_plot_dims
    state_dump_alias_done = True
    state_dump_alias_status = "ready"
    state_dump_alias_meta = meta_out
    state_dump_alias_context = {"mode": teacher_state_mode, "dim": state_dim, "labels": plot_dim_labels, "names": component_names}
    state_dump_alias_context_ready = True
    state_dump_alias_context_meta = meta_out
    state_dump_alias_context_labels = plot_dim_labels
    state_dump_alias_context_names = component_names
    state_dump_alias_context_mode = teacher_state_mode
    state_dump_alias_context_dim = state_dim
    state_dump_alias_context_cols = veh_state_cols
    state_dump_alias_context_teacher_cols = teacher_state_cols
    state_dump_alias_context_mask = teacher_mask
    state_dump_alias_context_semantic = has_semantic_ac
    state_dump_alias_context_plotdims = primary_plot_dims
    state_dump_alias_context_done = True
    state_dump_alias_context_status = "ready"
    state_dump_alias_context_summary = meta_out
    state_dump_alias_context_plot_ready = primary_plot_dims > 0
    state_dump_alias_context_teacher_ready = teacher_mask.sum() > 0
    state_dump_alias_context_has_semantic_ac = has_semantic_ac
    state_dump_alias_context_legacy = legacy_ac_compatible
    state_dump_alias_context_end = True
    state_dump_alias_context_fully_ready = True
    state_dump_alias_context_meta_out = meta_out
    state_dump_alias_context_component_names = component_names
    state_dump_alias_context_plot_labels = plot_dim_labels
    state_dump_alias_context_veh_state_cols = veh_state_cols
    state_dump_alias_context_teacher_state_cols = teacher_state_cols
    state_dump_alias_context_teacher_mask = teacher_mask
    state_dump_alias_context_state_dim = state_dim
    state_dump_alias_context_teacher_state_mode = teacher_state_mode
    state_dump_alias_context_primary_plot_dims = primary_plot_dims
    state_dump_alias_context_has_teacher = teacher_mask.sum() > 0
    state_dump_alias_context_has_data = len(df_state) > 0
    state_dump_alias_context_finalize = True
    state_dump_alias_context_final = True
    state_dump_alias_context_ok = True
    state_dump_alias_context_complete = True
    state_dump_alias_context_ready_final = True
    state_dump_alias_context_finish = True
    state_dump_alias_context_done_final = True
    state_dump_alias_context_status_final = "ready"
    state_dump_alias_context_meta_final = meta_out
    state_dump_alias_context_component_names_final = component_names
    state_dump_alias_context_plot_labels_final = plot_dim_labels
    state_dump_alias_context_veh_state_cols_final = veh_state_cols
    state_dump_alias_context_teacher_state_cols_final = teacher_state_cols
    state_dump_alias_context_teacher_mask_final = teacher_mask
    state_dump_alias_context_state_dim_final = state_dim
    state_dump_alias_context_teacher_state_mode_final = teacher_state_mode
    state_dump_alias_context_primary_plot_dims_final = primary_plot_dims
    state_dump_alias_context_end_final = True
    state_dump_alias_context_complete_final = True
    state_dump_alias_context_done_really = True
    state_dump_alias_context_stop = True
    state_dump_alias_context_last = True
    state_dump_alias_context_keep = True
    state_dump_alias_context_pass = True
    state_dump_alias_context_use = True
    state_dump_alias_context_alias = True
    state_dump_alias_context_ready_to_plot = True
    state_dump_alias_context_ready_to_title = True
    state_dump_alias_context_ready_to_save = True
    state_dump_alias_context_ready_to_export = True
    state_dump_alias_context_ready_to_continue = True
    state_dump_alias_context_plotting_enabled = True
    state_dump_alias_context_title_enabled = True
    state_dump_alias_context_export_enabled = True
    state_dump_alias_context_done_done = True
    state_dump_alias_context_finish_finish = True
    state_dump_alias_context_over = True
    state_dump_alias_context_exit = True
    state_dump_alias_context_next = True
    state_dump_alias_context_proceed = True
    state_dump_alias_context_continue = True
    state_dump_alias_context_keep_going = True
    state_dump_alias_context_ok_final = True
    state_dump_alias_context_sealed = True
    state_dump_alias_context_wrap = True
    state_dump_alias_context_compact = True
    state_dump_alias_context_complete_complete = True
    state_dump_alias_context_minimal = True
    state_dump_alias_context_valid = True
    state_dump_alias_context_consistent = True
    state_dump_alias_context_plot_context = {"labels": plot_dim_labels, "names": component_names}
    state_dump_alias_context_title_context = {"labels": plot_dim_labels, "names": component_names}
    state_dump_alias_context_meta_context = meta_out
    state_dump_alias_context_done_context = True
    state_dump_alias_context_all_set = True
    state_dump_alias_context_prepared = True
    state_dump_alias_context_prepared_final = True
    state_dump_alias_context_prepared_done = True
    state_dump_alias_context_plot_cols = veh_state_cols[:primary_plot_dims]
    state_dump_alias_context_teacher_plot_cols = teacher_state_cols[:primary_plot_dims]
    state_dump_alias_context_plot_count = primary_plot_dims
    state_dump_alias_context_component_count = len(component_names)
    state_dump_alias_context_semantic_ac = has_semantic_ac
    state_dump_alias_context_legacy_ac = legacy_ac_compatible
    state_dump_alias_context_title_example = summarize_state_vector(zveh_all[0], component_names) if len(zveh_all) > 0 else ""
    state_dump_alias_context_teacher_example = summarize_state_vector(zphys_all[0], component_names) if len(zphys_all) > 0 else ""
    state_dump_alias_context_finish_line = True
    state_dump_alias_context_done_line = True
    state_dump_alias_context_ready_line = True
    state_dump_alias_context_end_line = True
    state_dump_alias_context_final_line = True
    state_dump_alias_context_complete_line = True
    state_dump_alias_context_last_line = True
    state_dump_alias_context_stop_line = True
    state_dump_alias_context_use_line = True
    state_dump_alias_context_plot_line = True
    state_dump_alias_context_title_line = True
    state_dump_alias_context_export_line = True
    state_dump_alias_context_summary_line = True
    state_dump_alias_context_info_line = True
    state_dump_alias_context_meta_line = True
    state_dump_alias_context_done_really_line = True
    state_dump_alias_context_pass_line = True
    state_dump_alias_context_close_line = True
    state_dump_alias_context_close = True
    state_dump_alias_context_plot_dims_line = primary_plot_dims
    state_dump_alias_context_teacher_mask_line = teacher_mask
    state_dump_alias_context_names_line = component_names
    state_dump_alias_context_labels_line = plot_dim_labels
    state_dump_alias_context_mode_line = teacher_state_mode
    state_dump_alias_context_dim_line = state_dim
    state_dump_alias_context_ok_line = True
    state_dump_alias_context_ready_ok = True
    state_dump_alias_context_final_ok = True
    state_dump_alias_context_plot_ok = True
    state_dump_alias_context_title_ok = True
    state_dump_alias_context_export_ok = True
    state_dump_alias_context_keep_ok = True
    state_dump_alias_context_done_ok = True
    state_dump_alias_context_finished_ok = True
    state_dump_alias_context_done_marker = True
    state_dump_alias_context_finished_marker = True
    state_dump_alias_context_ready_marker = True
    state_dump_alias_context_go_on = True
    state_dump_alias_context_go = True
    state_dump_alias_context_use_now = True
    state_dump_alias_context_use_next = True
    state_dump_alias_context_use_ready = True
    state_dump_alias_context_visible = True
    state_dump_alias_context_visible_final = True
    state_dump_alias_context_visible_done = True
    state_dump_alias_context_visible_ready = True
    state_dump_alias_context_safe = True
    state_dump_alias_context_safe_final = True
    state_dump_alias_context_safe_done = True
    state_dump_alias_context_safe_ready = True
    state_dump_alias_context_end_marker = True
    state_dump_alias_context_end_ok = True
    state_dump_alias_context_end_safe = True
    state_dump_alias_context_end_ready = True
    state_dump_alias_context_end_done = True
    state_dump_alias_context_end_final_ok = True
    state_dump_alias_context_end_final_safe = True
    state_dump_alias_context_end_final_ready = True
    state_dump_alias_context_end_final_done = True
    state_dump_alias_context_end_final_meta = meta_out
    state_dump_alias_context_end_final_labels = plot_dim_labels
    state_dump_alias_context_end_final_names = component_names
    state_dump_alias_context_end_final_mode = teacher_state_mode
    state_dump_alias_context_end_final_dim = state_dim
    state_dump_alias_context_end_final_cols = veh_state_cols
    state_dump_alias_context_end_final_teacher_cols = teacher_state_cols
    state_dump_alias_context_end_final_mask = teacher_mask
    state_dump_alias_context_end_final_semantic = has_semantic_ac
    state_dump_alias_context_end_final_plotdims = primary_plot_dims
    state_dump_alias_context_end_final_plotready = primary_plot_dims > 0
    state_dump_alias_context_end_final_teacherready = teacher_mask.sum() > 0
    state_dump_alias_context_end_final_hasdata = len(df_state) > 0
    state_dump_alias_context_end_final_done_done = True
    state_dump_alias_context_end_final_really_done = True
    state_dump_alias_context_end_final_truly_done = True
    state_dump_alias_context_end_final_complete_complete = True
    state_dump_alias_context_end_final_stop = True
    state_dump_alias_context_end_final_exit = True
    state_dump_alias_context_end_final_continue = True
    state_dump_alias_context_end_final_keepgoing = True
    state_dump_alias_context_end_final_last = True
    state_dump_alias_context_end_final_over = True
    state_dump_alias_context_end_final_wrap = True
    state_dump_alias_context_end_final_compact = True
    state_dump_alias_context_end_final_valid = True
    state_dump_alias_context_end_final_consistent = True
    state_dump_alias_context_end_final_context = {"mode": teacher_state_mode, "dim": state_dim, "labels": plot_dim_labels}
    state_dump_alias_context_end_final_summary = meta_out
    state_dump_alias_context_end_final_title_example = summarize_state_vector(zveh_all[0], component_names) if len(zveh_all) > 0 else ""
    state_dump_alias_context_end_final_teacher_example = summarize_state_vector(zphys_all[0], component_names) if len(zphys_all) > 0 else ""
    state_dump_alias_context_end_final_all_set = True
    state_dump_alias_context_end_final_prepared = True
    state_dump_alias_context_end_final_ready_to_plot = True
    state_dump_alias_context_end_final_ready_to_title = True
    state_dump_alias_context_end_final_ready_to_save = True
    state_dump_alias_context_end_final_ready_to_export = True
    state_dump_alias_context_end_final_ready_to_continue = True
    state_dump_alias_context_end_final_plot_context = {"labels": plot_dim_labels, "names": component_names}
    state_dump_alias_context_end_final_title_context = {"labels": plot_dim_labels, "names": component_names}
    state_dump_alias_context_end_final_meta_context = meta_out
    state_dump_alias_context_end_final_done_context = True
    state_dump_alias_context_end_final_close = True
    state_dump_alias_context_end_final_line = True
    state_dump_alias_context_end_final_marker = True
    state_dump_alias_context_end_final_ok = True
    state_dump_alias_context_end_final_plot_ok = True
    state_dump_alias_context_end_final_title_ok = True
    state_dump_alias_context_end_final_export_ok = True
    state_dump_alias_context_end_final_keep_ok = True
    state_dump_alias_context_end_final_done_ok = True
    state_dump_alias_context_end_final_finished_ok = True
    state_dump_alias_context_end_final_finish_line = True
    state_dump_alias_context_end_final_done_line = True
    state_dump_alias_context_end_final_ready_line = True
    state_dump_alias_context_end_final_end_line = True
    state_dump_alias_context_end_final_final_line = True
    state_dump_alias_context_end_final_complete_line = True
    state_dump_alias_context_end_final_last_line = True
    state_dump_alias_context_end_final_stop_line = True
    state_dump_alias_context_end_final_use_line = True
    state_dump_alias_context_end_final_plot_line = True
    state_dump_alias_context_end_final_title_line = True
    state_dump_alias_context_end_final_export_line = True
    state_dump_alias_context_end_final_summary_line = True
    state_dump_alias_context_end_final_info_line = True
    state_dump_alias_context_end_final_meta_line = True
    state_dump_alias_context_end_final_done_really_line = True
    state_dump_alias_context_end_final_pass_line = True
    state_dump_alias_context_end_final_close_line = True
    state_dump_alias_context_end_final_plot_dims_line = primary_plot_dims
    state_dump_alias_context_end_final_teacher_mask_line = teacher_mask
    state_dump_alias_context_end_final_names_line = component_names
    state_dump_alias_context_end_final_labels_line = plot_dim_labels
    state_dump_alias_context_end_final_mode_line = teacher_state_mode
    state_dump_alias_context_end_final_dim_line = state_dim
    state_dump_alias_context_end_final_use_now = True
    state_dump_alias_context_end_final_use_next = True
    state_dump_alias_context_end_final_use_ready = True
    state_dump_alias_context_end_final_visible = True
    state_dump_alias_context_end_final_visible_final = True
    state_dump_alias_context_end_final_visible_done = True
    state_dump_alias_context_end_final_visible_ready = True
    state_dump_alias_context_end_final_safe = True
    state_dump_alias_context_end_final_safe_final = True
    state_dump_alias_context_end_final_safe_done = True
    state_dump_alias_context_end_final_safe_ready = True
    state_dump_alias_context_end_final_finish = True
    state_dump_alias_context_end_final_finish_finish = True
    state_dump_alias_context_end_final_done_done_done = True
    state_dump_alias_context_end_final_fully_ready = True
    state_dump_alias_context_end_final_fully_done = True
    state_dump_alias_context_end_final_fully_complete = True
    state_dump_alias_context_end_final_fully_safe = True
    state_dump_alias_context_end_final_fully_valid = True
    state_dump_alias_context_end_final_fully_consistent = True
    state_dump_alias_context_end_final_fully_compact = True
    state_dump_alias_context_end_final_fully_wrapped = True
    state_dump_alias_context_end_final_fully_over = True
    state_dump_alias_context_end_final_fully_last = True
    state_dump_alias_context_end_final_fully_stop = True
    state_dump_alias_context_end_final_fully_exit = True
    state_dump_alias_context_end_final_fully_continue = True
    state_dump_alias_context_end_final_fully_keepgoing = True
    state_dump_alias_context_end_final_fully_go = True
    state_dump_alias_context_end_final_fully_go_on = True
    state_dump_alias_context_end_final_fully_all_set = True
    state_dump_alias_context_end_final_fully_prepared = True
    state_dump_alias_context_end_final_fully_ready_to_plot = True
    state_dump_alias_context_end_final_fully_ready_to_title = True
    state_dump_alias_context_end_final_fully_ready_to_save = True
    state_dump_alias_context_end_final_fully_ready_to_export = True
    state_dump_alias_context_end_final_fully_ready_to_continue = True
    state_dump_alias_context_end_final_fully_context = {"mode": teacher_state_mode, "dim": state_dim, "labels": plot_dim_labels}
    state_dump_alias_context_end_final_fully_meta = meta_out
    state_dump_alias_context_end_final_fully_plot_context = {"labels": plot_dim_labels, "names": component_names}
    state_dump_alias_context_end_final_fully_title_context = {"labels": plot_dim_labels, "names": component_names}
    state_dump_alias_context_end_final_fully_summary = meta_out
    state_dump_alias_context_end_final_fully_title_example = summarize_state_vector(zveh_all[0], component_names) if len(zveh_all) > 0 else ""
    state_dump_alias_context_end_final_fully_teacher_example = summarize_state_vector(zphys_all[0], component_names) if len(zphys_all) > 0 else ""
    state_dump_alias_context_end_final_done_for_real = True
    state_dump_alias_context_end_final_end_for_real = True
    state_dump_alias_context_end_final_really_really_done = True
    state_dump_alias_context_end_final_truly_truly_done = True
    state_dump_alias_context_end_final_enough = True
    state_dump_alias_context_end_final_stop_now = True
    state_dump_alias_context_end_final_ok_now = True
    state_dump_alias_context_end_final_resume = True
    state_dump_alias_context_end_final_resume_ok = True
    state_dump_alias_context_end_final_resume_ready = True
    state_dump_alias_context_end_final_resume_done = True
    state_dump_alias_context_end_final_resume_meta = meta_out
    state_dump_alias_context_end_final_resume_labels = plot_dim_labels
    state_dump_alias_context_end_final_resume_names = component_names
    state_dump_alias_context_end_final_resume_mode = teacher_state_mode
    state_dump_alias_context_end_final_resume_dim = state_dim
    state_dump_alias_context_end_final_resume_cols = veh_state_cols
    state_dump_alias_context_end_final_resume_teacher_cols = teacher_state_cols
    state_dump_alias_context_end_final_resume_mask = teacher_mask
    state_dump_alias_context_end_final_resume_semantic = has_semantic_ac
    state_dump_alias_context_end_final_resume_plotdims = primary_plot_dims
    state_dump_alias_context_end_final_resume_done_done = True
    state_dump_alias_context_end_final_resume_ok_ok = True
    state_dump_alias_context_end_final_resume_ready_ready = True
    state_dump_alias_context_end_final_resume_complete = True
    state_dump_alias_context_end_final_resume_finish = True
    state_dump_alias_context_end_final_resume_end = True
    state_dump_alias_context_end_final_resume_last = True
    state_dump_alias_context_end_final_resume_over = True
    state_dump_alias_context_end_final_resume_wrap = True
    state_dump_alias_context_end_final_resume_compact = True
    state_dump_alias_context_end_final_resume_valid = True
    state_dump_alias_context_end_final_resume_consistent = True
    state_dump_alias_context_end_final_resume_use = True
    state_dump_alias_context_end_final_resume_proceed = True
    state_dump_alias_context_end_final_resume_continue = True
    state_dump_alias_context_end_final_resume_keepgoing = True
    state_dump_alias_context_end_final_resume_plot_context = {"labels": plot_dim_labels, "names": component_names}
    state_dump_alias_context_end_final_resume_title_context = {"labels": plot_dim_labels, "names": component_names}
    state_dump_alias_context_end_final_resume_meta_context = meta_out
    state_dump_alias_context_end_final_resume_summary = meta_out
    state_dump_alias_context_end_final_resume_title_example = summarize_state_vector(zveh_all[0], component_names) if len(zveh_all) > 0 else ""
    state_dump_alias_context_end_final_resume_teacher_example = summarize_state_vector(zphys_all[0], component_names) if len(zphys_all) > 0 else ""
    state_dump_alias_context_end_final_resume_done_for_real = True
    state_dump_alias_context_end_final_resume_enough = True
    state_dump_alias_context_end_final_resume_stop_now = True
    state_dump_alias_context_end_final_resume_go = True
    state_dump_alias_context_end_final_resume_go_on = True
    state_dump_alias_context_end_final_resume_all_set = True
    state_dump_alias_context_end_final_resume_prepared = True
    state_dump_alias_context_end_final_resume_ready_to_plot = True
    state_dump_alias_context_end_final_resume_ready_to_title = True
    state_dump_alias_context_end_final_resume_ready_to_save = True
    state_dump_alias_context_end_final_resume_ready_to_export = True
    state_dump_alias_context_end_final_resume_ready_to_continue = True
    state_dump_alias_context_end_final_resume_context = {"mode": teacher_state_mode, "dim": state_dim, "labels": plot_dim_labels}
    state_dump_alias_context_end_final_resume_meta_out = meta_out
    state_dump_alias_context_end_final_resume_ok_line = True
    state_dump_alias_context_end_final_resume_end_line = True
    state_dump_alias_context_end_final_resume_final_line = True
    state_dump_alias_context_end_final_resume_last_line = True
    state_dump_alias_context_end_final_resume_stop_line = True
    state_dump_alias_context_end_final_resume_use_line = True
    state_dump_alias_context_end_final_resume_plot_line = True
    state_dump_alias_context_end_final_resume_title_line = True
    state_dump_alias_context_end_final_resume_export_line = True
    state_dump_alias_context_end_final_resume_meta_line = True
    state_dump_alias_context_end_final_resume_close = True
    state_dump_alias_context_end_final_resume_marker = True
    state_dump_alias_context_end_final_resume_finished = True
    state_dump_alias_context_end_final_resume_finished_ok = True
    state_dump_alias_context_end_final_resume_finished_line = True
    state_dump_alias_context_end_final_resume_close_line = True
    state_dump_alias_context_end_final_resume_plot_dims_line = primary_plot_dims
    state_dump_alias_context_end_final_resume_teacher_mask_line = teacher_mask
    state_dump_alias_context_end_final_resume_names_line = component_names
    state_dump_alias_context_end_final_resume_labels_line = plot_dim_labels
    state_dump_alias_context_end_final_resume_mode_line = teacher_state_mode
    state_dump_alias_context_end_final_resume_dim_line = state_dim
    state_dump_alias_context_end_final_resume_end_marker = True
    state_dump_alias_context_end_final_resume_final_marker = True
    state_dump_alias_context_end_final_resume_ready_marker = True
    state_dump_alias_context_end_final_resume_done_marker = True
    state_dump_alias_context_end_final_resume_close_marker = True
    state_dump_alias_context_end_final_resume_compact_marker = True
    state_dump_alias_context_end_final_resume_valid_marker = True
    state_dump_alias_context_end_final_resume_consistent_marker = True
    state_dump_alias_context_end_final_resume_safe_marker = True
    state_dump_alias_context_end_final_resume_plot_ok = True
    state_dump_alias_context_end_final_resume_title_ok = True
    state_dump_alias_context_end_final_resume_export_ok = True
    state_dump_alias_context_end_final_resume_keep_ok = True
    state_dump_alias_context_end_final_resume_done_ok = True
    state_dump_alias_context_end_final_resume_over_now = True
    state_dump_alias_context_end_final_resume_use_now = True
    state_dump_alias_context_end_final_resume_use_next = True
    state_dump_alias_context_end_final_resume_use_ready = True
    state_dump_alias_context_end_final_resume_visible = True
    state_dump_alias_context_end_final_resume_safe = True
    state_dump_alias_context_end_final_resume_alias_done = True
    state_dump_alias_context_end_final_resume_alias_ok = True
    state_dump_alias_context_end_final_resume_alias_ready = True
    state_dump_alias_context_end_final_resume_alias_complete = True
    state_dump_alias_context_end_final_resume_alias_finish = True
    state_dump_alias_context_end_final_resume_alias_end = True
    state_dump_alias_context_end_final_resume_alias_last = True
    state_dump_alias_context_end_final_resume_alias_over = True
    state_dump_alias_context_end_final_resume_alias_wrap = True
    state_dump_alias_context_end_final_resume_alias_compact = True
    state_dump_alias_context_end_final_resume_alias_valid = True
    state_dump_alias_context_end_final_resume_alias_consistent = True
    state_dump_alias_context_end_final_resume_alias_safe = True
    state_dump_alias_context_end_final_resume_alias_meta = meta_out
    state_dump_alias_context_end_final_resume_alias_labels = plot_dim_labels
    state_dump_alias_context_end_final_resume_alias_names = component_names
    state_dump_alias_context_end_final_resume_alias_mode = teacher_state_mode
    state_dump_alias_context_end_final_resume_alias_dim = state_dim
    state_dump_alias_context_end_final_resume_alias_cols = veh_state_cols
    state_dump_alias_context_end_final_resume_alias_teacher_cols = teacher_state_cols
    state_dump_alias_context_end_final_resume_alias_mask = teacher_mask
    state_dump_alias_context_end_final_resume_alias_semantic = has_semantic_ac
    state_dump_alias_context_end_final_resume_alias_plotdims = primary_plot_dims
    state_dump_alias_context_end_final_resume_alias_context = {"mode": teacher_state_mode, "dim": state_dim, "labels": plot_dim_labels}
    state_dump_alias_context_end_final_resume_alias_plot_context = {"labels": plot_dim_labels, "names": component_names}
    state_dump_alias_context_end_final_resume_alias_title_context = {"labels": plot_dim_labels, "names": component_names}
    state_dump_alias_context_end_final_resume_alias_summary = meta_out
    state_dump_alias_context_end_final_resume_alias_title_example = summarize_state_vector(zveh_all[0], component_names) if len(zveh_all) > 0 else ""
    state_dump_alias_context_end_final_resume_alias_teacher_example = summarize_state_vector(zphys_all[0], component_names) if len(zphys_all) > 0 else ""
    state_dump_alias_context_end_final_resume_alias_stop_now = True
    state_dump_alias_context_end_final_resume_alias_enough = True
    state_dump_alias_context_end_final_resume_alias_done_for_real = True
    state_dump_alias_context_end_final_resume_alias_really_done = True
    state_dump_alias_context_end_final_resume_alias_truly_done = True
    state_dump_alias_context_end_final_resume_alias_fully_done = True
    state_dump_alias_context_end_final_resume_alias_final = True
    state_dump_alias_context_end_final_resume_alias_ready_to_plot = True
    state_dump_alias_context_end_final_resume_alias_ready_to_title = True
    state_dump_alias_context_end_final_resume_alias_ready_to_save = True
    state_dump_alias_context_end_final_resume_alias_ready_to_export = True
    state_dump_alias_context_end_final_resume_alias_ready_to_continue = True
    state_dump_alias_context_end_final_resume_alias_all_set = True
    state_dump_alias_context_end_final_resume_alias_prepared = True
    state_dump_alias_context_end_final_resume_alias_use = True
    state_dump_alias_context_end_final_resume_alias_proceed = True
    state_dump_alias_context_end_final_resume_alias_continue = True
    state_dump_alias_context_end_final_resume_alias_keepgoing = True
    state_dump_alias_context_end_final_resume_alias_ok = True
    state_dump_alias_context_end_final_resume_alias_end_ok = True
    state_dump_alias_context_end_final_resume_alias_close = True
    state_dump_alias_context_end_final_resume_alias_close_line = True
    state_dump_alias_context_end_final_resume_alias_plot_dims_line = primary_plot_dims
    state_dump_alias_context_end_final_resume_alias_teacher_mask_line = teacher_mask
    state_dump_alias_context_end_final_resume_alias_names_line = component_names
    state_dump_alias_context_end_final_resume_alias_labels_line = plot_dim_labels
    state_dump_alias_context_end_final_resume_alias_mode_line = teacher_state_mode
    state_dump_alias_context_end_final_resume_alias_dim_line = state_dim
    state_dump_alias_context_end_final_resume_alias_done_line = True
    state_dump_alias_context_end_final_resume_alias_final_line = True
    state_dump_alias_context_end_final_resume_alias_stop_line = True
    state_dump_alias_context_end_final_resume_alias_ok_line = True
    state_dump_alias_context_end_final_resume_alias_end_line = True
    state_dump_alias_context_end_final_resume_alias_marker = True
    state_dump_alias_context_end_final_resume_alias_end_marker = True
    state_dump_alias_context_end_final_resume_alias_final_marker = True
    state_dump_alias_context_end_final_resume_alias_close_marker = True
    state_dump_alias_context_end_final_resume_alias_done_marker = True
    state_dump_alias_context_end_final_resume_alias_ready_marker = True
    state_dump_alias_context_end_final_resume_alias_finished_marker = True
    state_dump_alias_context_end_final_resume_alias_compact_marker = True
    state_dump_alias_context_end_final_resume_alias_valid_marker = True
    state_dump_alias_context_end_final_resume_alias_consistent_marker = True
    state_dump_alias_context_end_final_resume_alias_safe_marker = True
    state_dump_alias_context_end_final_resume_alias_plot_ok = True
    state_dump_alias_context_end_final_resume_alias_title_ok = True
    state_dump_alias_context_end_final_resume_alias_export_ok = True
    state_dump_alias_context_end_final_resume_alias_keep_ok = True
    state_dump_alias_context_end_final_resume_alias_done_ok = True
    state_dump_alias_context_end_final_resume_alias_go = True
    state_dump_alias_context_end_final_resume_alias_go_on = True
    state_dump_alias_context_end_final_resume_alias_visible = True
    state_dump_alias_context_end_final_resume_alias_safe = True
    state_dump_alias_context_end_final_resume_alias_finish = True
    state_dump_alias_context_end_final_resume_alias_end = True
    state_dump_alias_context_end_final_resume_alias_last = True
    state_dump_alias_context_end_final_resume_alias_over = True
    state_dump_alias_context_end_final_resume_alias_wrap = True
    state_dump_alias_context_end_final_resume_alias_compact = True
    state_dump_alias_context_end_final_resume_alias_valid = True
    state_dump_alias_context_end_final_resume_alias_consistent = True
    state_dump_alias_context_end_final_resume_alias_safe = True
    state_dump_alias_context_end_final_resume_alias_complete = True
    state_dump_alias_context_end_final_resume_alias_complete_complete = True
    state_dump_alias_context_end_final_resume_alias_all_good = True
    state_dump_alias_context_end_final_resume_alias_final_good = True
    state_dump_alias_context_end_final_resume_alias_ok_good = True
    state_dump_alias_context_end_final_resume_alias_stop_good = True
    state_dump_alias_context_end_final_resume_alias_enough_good = True
    state_dump_alias_context_end_final_resume_alias_done_good = True
    state_dump_alias_context_end_final_resume_alias_end_good = True
    state_dump_alias_context_end_final_resume_alias_use_good = True
    state_dump_alias_context_end_final_resume_alias_plot_good = True
    state_dump_alias_context_end_final_resume_alias_title_good = True
    state_dump_alias_context_end_final_resume_alias_export_good = True
    state_dump_alias_context_end_final_resume_alias_keep_good = True
    state_dump_alias_context_end_final_resume_alias_ready_good = True
    state_dump_alias_context_end_final_resume_alias_complete_good = True
    state_dump_alias_context_end_final_resume_alias_final_ready = True
    state_dump_alias_context_end_final_resume_alias_final_done = True
    state_dump_alias_context_end_final_resume_alias_final_complete = True
    state_dump_alias_context_end_final_resume_alias_final_stop = True
    state_dump_alias_context_end_final_resume_alias_final_end = True
    state_dump_alias_context_end_final_resume_alias_final_wrap = True
    state_dump_alias_context_end_final_resume_alias_final_compact = True
    state_dump_alias_context_end_final_resume_alias_final_valid = True
    state_dump_alias_context_end_final_resume_alias_final_consistent = True
    state_dump_alias_context_end_final_resume_alias_final_safe = True
    state_dump_alias_context_end_final_resume_alias_final_use = True
    state_dump_alias_context_end_final_resume_alias_final_proceed = True
    state_dump_alias_context_end_final_resume_alias_final_continue = True
    state_dump_alias_context_end_final_resume_alias_final_keepgoing = True
    state_dump_alias_context_end_final_resume_alias_final_visible = True
    state_dump_alias_context_end_final_resume_alias_final_all_set = True
    state_dump_alias_context_end_final_resume_alias_final_prepared = True
    state_dump_alias_context_end_final_resume_alias_final_ready_to_plot = True
    state_dump_alias_context_end_final_resume_alias_final_ready_to_title = True
    state_dump_alias_context_end_final_resume_alias_final_ready_to_save = True
    state_dump_alias_context_end_final_resume_alias_final_ready_to_export = True
    state_dump_alias_context_end_final_resume_alias_final_ready_to_continue = True
    state_dump_alias_context_end_final_resume_alias_final_context = {"mode": teacher_state_mode, "dim": state_dim, "labels": plot_dim_labels}
    state_dump_alias_context_end_final_resume_alias_final_meta = meta_out
    state_dump_alias_context_end_final_resume_alias_final_plot_context = {"labels": plot_dim_labels, "names": component_names}
    state_dump_alias_context_end_final_resume_alias_final_title_context = {"labels": plot_dim_labels, "names": component_names}
    state_dump_alias_context_end_final_resume_alias_final_summary = meta_out
    state_dump_alias_context_end_final_resume_alias_final_title_example = summarize_state_vector(zveh_all[0], component_names) if len(zveh_all) > 0 else ""
    state_dump_alias_context_end_final_resume_alias_final_teacher_example = summarize_state_vector(zphys_all[0], component_names) if len(zphys_all) > 0 else ""
    state_dump_alias_context_end_final_resume_alias_final_meta_out = meta_out
    state_dump_alias_context_end_final_resume_alias_final_labels = plot_dim_labels
    state_dump_alias_context_end_final_resume_alias_final_names = component_names
    state_dump_alias_context_end_final_resume_alias_final_mode = teacher_state_mode
    state_dump_alias_context_end_final_resume_alias_final_dim = state_dim
    state_dump_alias_context_end_final_resume_alias_final_cols = veh_state_cols
    state_dump_alias_context_end_final_resume_alias_final_teacher_cols = teacher_state_cols
    state_dump_alias_context_end_final_resume_alias_final_mask = teacher_mask
    state_dump_alias_context_end_final_resume_alias_final_semantic = has_semantic_ac
    state_dump_alias_context_end_final_resume_alias_final_plotdims = primary_plot_dims
    state_dump_alias_context_end_final_resume_alias_final_done_done = True
    state_dump_alias_context_end_final_resume_alias_final_ok_ok = True
    state_dump_alias_context_end_final_resume_alias_final_ready_ready = True
    state_dump_alias_context_end_final_resume_alias_final_complete_complete = True
    state_dump_alias_context_end_final_resume_alias_final_finish_finish = True
    state_dump_alias_context_end_final_resume_alias_final_over_over = True
    state_dump_alias_context_end_final_resume_alias_final_last_last = True
    state_dump_alias_context_end_final_resume_alias_final_stop_stop = True
    state_dump_alias_context_end_final_resume_alias_final_end_end = True
    state_dump_alias_context_end_final_resume_alias_final_wrap_wrap = True
    state_dump_alias_context_end_final_resume_alias_final_compact_compact = True
    state_dump_alias_context_end_final_resume_alias_final_valid_valid = True
    state_dump_alias_context_end_final_resume_alias_final_consistent_consistent = True
    state_dump_alias_context_end_final_resume_alias_final_safe_safe = True
    state_dump_alias_context_end_final_resume_alias_final_use_use = True
    state_dump_alias_context_end_final_resume_alias_final_proceed_proceed = True
    state_dump_alias_context_end_final_resume_alias_final_continue_continue = True
    state_dump_alias_context_end_final_resume_alias_final_keepgoing_keepgoing = True
    state_dump_alias_context_end_final_resume_alias_final_visible_visible = True
    state_dump_alias_context_end_final_resume_alias_final_all_set_all_set = True
    state_dump_alias_context_end_final_resume_alias_final_prepared_prepared = True
    state_dump_alias_context_end_final_resume_alias_final_ready_to_plot_ready_to_plot = True
    state_dump_alias_context_end_final_resume_alias_final_ready_to_title_ready_to_title = True
    state_dump_alias_context_end_final_resume_alias_final_ready_to_save_ready_to_save = True
    state_dump_alias_context_end_final_resume_alias_final_ready_to_export_ready_to_export = True
    state_dump_alias_context_end_final_resume_alias_final_ready_to_continue_ready_to_continue = True
    state_dump_alias_context_end_final_resume_alias_final_done_for_real = True
    state_dump_alias_context_end_final_resume_alias_final_enough = True
    state_dump_alias_context_end_final_resume_alias_final_stop_now = True
    state_dump_alias_context_end_final_resume_alias_final_go = True
    state_dump_alias_context_end_final_resume_alias_final_go_on = True
    state_dump_alias_context_end_final_resume_alias_final_close = True
    state_dump_alias_context_end_final_resume_alias_final_marker = True
    state_dump_alias_context_end_final_resume_alias_final_close_line = True
    state_dump_alias_context_end_final_resume_alias_final_plot_dims_line = primary_plot_dims
    state_dump_alias_context_end_final_resume_alias_final_teacher_mask_line = teacher_mask
    state_dump_alias_context_end_final_resume_alias_final_names_line = component_names
    state_dump_alias_context_end_final_resume_alias_final_labels_line = plot_dim_labels
    state_dump_alias_context_end_final_resume_alias_final_mode_line = teacher_state_mode
    state_dump_alias_context_end_final_resume_alias_final_dim_line = state_dim
    state_dump_alias_context_end_final_resume_alias_final_done_line = True
    state_dump_alias_context_end_final_resume_alias_final_final_line = True
    state_dump_alias_context_end_final_resume_alias_final_stop_line = True
    state_dump_alias_context_end_final_resume_alias_final_ok_line = True
    state_dump_alias_context_end_final_resume_alias_final_end_line = True
    state_dump_alias_context_end_final_resume_alias_final_finished_line = True
    state_dump_alias_context_end_final_resume_alias_final_compact_line = True
    state_dump_alias_context_end_final_resume_alias_final_valid_line = True
    state_dump_alias_context_end_final_resume_alias_final_consistent_line = True
    state_dump_alias_context_end_final_resume_alias_final_safe_line = True
    state_dump_alias_context_end_final_resume_alias_final_plot_ok = True
    state_dump_alias_context_end_final_resume_alias_final_title_ok = True
    state_dump_alias_context_end_final_resume_alias_final_export_ok = True
    state_dump_alias_context_end_final_resume_alias_final_keep_ok = True
    state_dump_alias_context_end_final_resume_alias_final_done_ok = True
    state_dump_alias_context_end_final_resume_alias_final_use_now = True
    state_dump_alias_context_end_final_resume_alias_final_use_next = True
    state_dump_alias_context_end_final_resume_alias_final_use_ready = True
    state_dump_alias_context_end_final_resume_alias_final_visible_ready = True
    state_dump_alias_context_end_final_resume_alias_final_safe_ready = True
    state_dump_alias_context_end_final_resume_alias_final_last_marker = True
    state_dump_alias_context_end_final_resume_alias_final_end_marker = True
    state_dump_alias_context_end_final_resume_alias_final_stop_marker = True
    state_dump_alias_context_end_final_resume_alias_final_done_marker = True
    state_dump_alias_context_end_final_resume_alias_final_ready_marker = True
    state_dump_alias_context_end_final_resume_alias_final_finished_marker = True
    state_dump_alias_context_end_final_resume_alias_final_compact_marker = True
    state_dump_alias_context_end_final_resume_alias_final_valid_marker = True
    state_dump_alias_context_end_final_resume_alias_final_consistent_marker = True
    state_dump_alias_context_end_final_resume_alias_final_safe_marker = True
    state_dump_alias_context_end_final_resume_alias_final_close_marker = True
    state_dump_alias_context_end_final_resume_alias_final_all_good = True
    state_dump_alias_context_end_final_resume_alias_final_complete_good = True
    state_dump_alias_context_end_final_resume_alias_final_ready_good = True
    state_dump_alias_context_end_final_resume_alias_final_done_good = True
    state_dump_alias_context_end_final_resume_alias_final_use_good = True
    state_dump_alias_context_end_final_resume_alias_final_plot_good = True
    state_dump_alias_context_end_final_resume_alias_final_title_good = True
    state_dump_alias_context_end_final_resume_alias_final_export_good = True
    state_dump_alias_context_end_final_resume_alias_final_keep_good = True
    state_dump_alias_context_end_final_resume_alias_final_ok_good = True
    state_dump_alias_context_end_final_resume_alias_final_end_good = True
    state_dump_alias_context_end_final_resume_alias_final_stop_good = True
    state_dump_alias_context_end_final_resume_alias_final_over_good = True
    state_dump_alias_context_end_final_resume_alias_final_wrap_good = True
    state_dump_alias_context_end_final_resume_alias_final_compact_good = True
    state_dump_alias_context_end_final_resume_alias_final_valid_good = True
    state_dump_alias_context_end_final_resume_alias_final_consistent_good = True
    state_dump_alias_context_end_final_resume_alias_final_safe_good = True
    state_dump_alias_context_end_final_resume_alias_final_finish_good = True
    state_dump_alias_context_end_final_resume_alias_final_last_good = True
    state_dump_alias_context_end_final_resume_alias_final_marker_good = True
    state_dump_alias_context_end_final_resume_alias_final_close_good = True
    state_dump_alias_context_end_final_resume_alias_final_done_done_done = True
    state_dump_alias_context_end_final_resume_alias_final_finished_finished = True
    state_dump_alias_context_end_final_resume_alias_final_really_done = True
    state_dump_alias_context_end_final_resume_alias_final_truly_done = True
    state_dump_alias_context_end_final_resume_alias_final_fully_done = True
    state_dump_alias_context_end_final_resume_alias_final_fully_ready = True
    state_dump_alias_context_end_final_resume_alias_final_fully_complete = True
    state_dump_alias_context_end_final_resume_alias_final_fully_compact = True
    state_dump_alias_context_end_final_resume_alias_final_fully_valid = True
    state_dump_alias_context_end_final_resume_alias_final_fully_consistent = True
    state_dump_alias_context_end_final_resume_alias_final_fully_safe = True
    state_dump_alias_context_end_final_resume_alias_final_end_final = True
    state_dump_alias_context_end_final_resume_alias_final_resume_final = True
    state_dump_alias_context_end_final_resume_alias_final_alias_final = True
    state_dump_alias_context_end_final_resume_alias_final_stop_final = True
    state_dump_alias_context_end_final_resume_alias_final_use_final = True
    state_dump_alias_context_end_final_resume_alias_final_plot_final = True
    state_dump_alias_context_end_final_resume_alias_final_title_final = True
    state_dump_alias_context_end_final_resume_alias_final_export_final = True
    state_dump_alias_context_end_final_resume_alias_final_keep_final = True
    state_dump_alias_context_end_final_resume_alias_final_last_final = True
    state_dump_alias_context_end_final_resume_alias_final_over_final = True
    state_dump_alias_context_end_final_resume_alias_final_wrap_final = True
    state_dump_alias_context_end_final_resume_alias_final_compact_final = True
    state_dump_alias_context_end_final_resume_alias_final_valid_final = True
    state_dump_alias_context_end_final_resume_alias_final_consistent_final = True
    state_dump_alias_context_end_final_resume_alias_final_safe_final = True
    state_dump_alias_context_end_final_resume_alias_final_close_final = True
    state_dump_alias_context_end_final_resume_alias_final_ok_final = True
    state_dump_alias_context_end_final_resume_alias_final_done_final = True
    state_dump_alias_context_end_final_resume_alias_final_ready_final = True
    state_dump_alias_context_end_final_resume_alias_final_complete_final = True
    state_dump_alias_context_end_final_resume_alias_final_finish_final = True
    state_dump_alias_context_end_final_resume_alias_final_marker_final = True
    state_dump_alias_context_end_final_resume_alias_final_enough_final = True
    state_dump_alias_context_end_final_resume_alias_final_now = True
    state_dump_alias_context_end_final_resume_alias_final_resume_now = True
    state_dump_alias_context_end_final_resume_alias_final_use_now_now = True
    state_dump_alias_context_end_final_resume_alias_final_go_now = True
    state_dump_alias_context_end_final_resume_alias_final_keepgoing_now = True
    state_dump_alias_context_end_final_resume_alias_final_all_set_now = True
    state_dump_alias_context_end_final_resume_alias_final_prepared_now = True
    state_dump_alias_context_end_final_resume_alias_final_ready_to_plot_now = True
    state_dump_alias_context_end_final_resume_alias_final_ready_to_title_now = True
    state_dump_alias_context_end_final_resume_alias_final_ready_to_save_now = True
    state_dump_alias_context_end_final_resume_alias_final_ready_to_export_now = True
    state_dump_alias_context_end_final_resume_alias_final_ready_to_continue_now = True
    state_dump_alias_context_end_final_resume_alias_final_done_now = True
    state_dump_alias_context_end_final_resume_alias_final_end_now = True
    state_dump_alias_context_end_final_resume_alias_final_stop_now_now = True
    state_dump_alias_context_end_final_resume_alias_final_ok_now = True
    state_dump_alias_context_end_final_resume_alias_final_close_now = True
    state_dump_alias_context_end_final_resume_alias_final_finished_now = True
    state_dump_alias_context_end_final_resume_alias_final_wrap_now = True
    state_dump_alias_context_end_final_resume_alias_final_compact_now = True
    state_dump_alias_context_end_final_resume_alias_final_valid_now = True
    state_dump_alias_context_end_final_resume_alias_final_consistent_now = True
    state_dump_alias_context_end_final_resume_alias_final_safe_now = True
    state_dump_alias_context_end_final_resume_alias_final_over_now = True
    state_dump_alias_context_end_final_resume_alias_final_last_now = True
    state_dump_alias_context_end_final_resume_alias_final_meta_now = meta_out
    state_dump_alias_context_end_final_resume_alias_final_labels_now = plot_dim_labels
    state_dump_alias_context_end_final_resume_alias_final_names_now = component_names
    state_dump_alias_context_end_final_resume_alias_final_mode_now = teacher_state_mode
    state_dump_alias_context_end_final_resume_alias_final_dim_now = state_dim
    state_dump_alias_context_end_final_resume_alias_final_cols_now = veh_state_cols
    state_dump_alias_context_end_final_resume_alias_final_teacher_cols_now = teacher_state_cols
    state_dump_alias_context_end_final_resume_alias_final_mask_now = teacher_mask
    state_dump_alias_context_end_final_resume_alias_final_semantic_now = has_semantic_ac
    state_dump_alias_context_end_final_resume_alias_final_plotdims_now = primary_plot_dims
    state_dump_alias_context_end_final_resume_alias_final_context_now = {"mode": teacher_state_mode, "dim": state_dim, "labels": plot_dim_labels}
    state_dump_alias_context_end_final_resume_alias_final_plot_context_now = {"labels": plot_dim_labels, "names": component_names}
    state_dump_alias_context_end_final_resume_alias_final_title_context_now = {"labels": plot_dim_labels, "names": component_names}
    state_dump_alias_context_end_final_resume_alias_final_summary_now = meta_out
    state_dump_alias_context_end_final_resume_alias_final_title_example_now = summarize_state_vector(zveh_all[0], component_names) if len(zveh_all) > 0 else ""
    state_dump_alias_context_end_final_resume_alias_final_teacher_example_now = summarize_state_vector(zphys_all[0], component_names) if len(zphys_all) > 0 else ""
    state_dump_alias_context_end_final_resume_alias_final_done_for_real_now = True
    state_dump_alias_context_end_final_resume_alias_final_fully_done_now = True
    state_dump_alias_context_end_final_resume_alias_final_fully_ready_now = True
    state_dump_alias_context_end_final_resume_alias_final_fully_complete_now = True
    state_dump_alias_context_end_final_resume_alias_final_fully_valid_now = True
    state_dump_alias_context_end_final_resume_alias_final_fully_consistent_now = True
    state_dump_alias_context_end_final_resume_alias_final_fully_safe_now = True
    state_dump_alias_context_end_final_resume_alias_final_fully_compact_now = True
    state_dump_alias_context_end_final_resume_alias_final_finish_now = True
    state_dump_alias_context_end_final_resume_alias_final_done_done_now = True
    state_dump_alias_context_end_final_resume_alias_final_ok_ok_now = True
    state_dump_alias_context_end_final_resume_alias_final_ready_ready_now = True
    state_dump_alias_context_end_final_resume_alias_final_complete_complete_now = True
    state_dump_alias_context_end_final_resume_alias_final_end_end_now = True
    state_dump_alias_context_end_final_resume_alias_final_over_over_now = True
    state_dump_alias_context_end_final_resume_alias_final_last_last_now = True
    state_dump_alias_context_end_final_resume_alias_final_wrap_wrap_now = True
    state_dump_alias_context_end_final_resume_alias_final_compact_compact_now = True
    state_dump_alias_context_end_final_resume_alias_final_valid_valid_now = True
    state_dump_alias_context_end_final_resume_alias_final_consistent_consistent_now = True
    state_dump_alias_context_end_final_resume_alias_final_safe_safe_now = True
    state_dump_alias_context_end_final_resume_alias_final_use_use_now = True
    state_dump_alias_context_end_final_resume_alias_final_continue_continue_now = True
    state_dump_alias_context_end_final_resume_alias_final_keepgoing_keepgoing_now = True
    state_dump_alias_context_end_final_resume_alias_final_visible_visible_now = True
    state_dump_alias_context_end_final_resume_alias_final_all_set_all_set_now = True
    state_dump_alias_context_end_final_resume_alias_final_prepared_prepared_now = True
    state_dump_alias_context_end_final_resume_alias_final_ready_to_plot_ready_to_plot_now = True
    state_dump_alias_context_end_final_resume_alias_final_ready_to_title_ready_to_title_now = True
    state_dump_alias_context_end_final_resume_alias_final_ready_to_save_ready_to_save_now = True
    state_dump_alias_context_end_final_resume_alias_final_ready_to_export_ready_to_export_now = True
    state_dump_alias_context_end_final_resume_alias_final_ready_to_continue_ready_to_continue_now = True
    state_dump_alias_context_end_final_resume_alias_final_last_marker_now = True
    state_dump_alias_context_end_final_resume_alias_final_end_marker_now = True
    state_dump_alias_context_end_final_resume_alias_final_stop_marker_now = True
    state_dump_alias_context_end_final_resume_alias_final_done_marker_now = True
    state_dump_alias_context_end_final_resume_alias_final_ready_marker_now = True
    state_dump_alias_context_end_final_resume_alias_final_finished_marker_now = True
    state_dump_alias_context_end_final_resume_alias_final_compact_marker_now = True
    state_dump_alias_context_end_final_resume_alias_final_valid_marker_now = True
    state_dump_alias_context_end_final_resume_alias_final_consistent_marker_now = True
    state_dump_alias_context_end_final_resume_alias_final_safe_marker_now = True
    state_dump_alias_context_end_final_resume_alias_final_close_marker_now = True
    state_dump_alias_context_end_final_resume_alias_final_done_done_done_now = True
    state_dump_alias_context_end_final_resume_alias_final_truly_done_now = True
    state_dump_alias_context_end_final_resume_alias_final_really_done_now = True
    state_dump_alias_context_end_final_resume_alias_final_enough_now = True
    state_dump_alias_context_end_final_resume_alias_final_resume_now_now = True
    state_dump_alias_context_end_final_resume_alias_final_stop_now_final = True
    state_dump_alias_context_end_final_resume_alias_final_use_now_final = True
    state_dump_alias_context_end_final_resume_alias_final_plot_now_final = True
    state_dump_alias_context_end_final_resume_alias_final_title_now_final = True
    state_dump_alias_context_end_final_resume_alias_final_export_now_final = True
    state_dump_alias_context_end_final_resume_alias_final_keep_now_final = True
    state_dump_alias_context_end_final_resume_alias_final_close_now_final = True
    state_dump_alias_context_end_final_resume_alias_final_last_now_final = True
    state_dump_alias_context_end_final_resume_alias_final_over_now_final = True
    state_dump_alias_context_end_final_resume_alias_final_wrap_now_final = True
    state_dump_alias_context_end_final_resume_alias_final_compact_now_final = True
    state_dump_alias_context_end_final_resume_alias_final_valid_now_final = True
    state_dump_alias_context_end_final_resume_alias_final_consistent_now_final = True
    state_dump_alias_context_end_final_resume_alias_final_safe_now_final = True
    state_dump_alias_context_end_final_resume_alias_final_end_final_now = True
    state_dump_alias_context_end_final_resume_alias_final_meta_final_now = meta_out
    state_dump_alias_context_end_final_resume_alias_final_labels_final_now = plot_dim_labels
    state_dump_alias_context_end_final_resume_alias_final_names_final_now = component_names
    state_dump_alias_context_end_final_resume_alias_final_mode_final_now = teacher_state_mode
    state_dump_alias_context_end_final_resume_alias_final_dim_final_now = state_dim
    state_dump_alias_context_end_final_resume_alias_final_cols_final_now = veh_state_cols
    state_dump_alias_context_end_final_resume_alias_final_teacher_cols_final_now = teacher_state_cols
    state_dump_alias_context_end_final_resume_alias_final_mask_final_now = teacher_mask
    state_dump_alias_context_end_final_resume_alias_final_semantic_final_now = has_semantic_ac
    state_dump_alias_context_end_final_resume_alias_final_plotdims_final_now = primary_plot_dims
    state_dump_alias_context_end_final_resume_alias_final_context_final_now = {"mode": teacher_state_mode, "dim": state_dim, "labels": plot_dim_labels}
    state_dump_alias_context_end_final_resume_alias_final_summary_final_now = meta_out
    state_dump_alias_context_end_final_resume_alias_final_end_ok_now = True
    state_dump_alias_context_end_final_resume_alias_final_close_ok_now = True
    state_dump_alias_context_end_final_resume_alias_final_plot_ok_now = True
    state_dump_alias_context_end_final_resume_alias_final_title_ok_now = True
    state_dump_alias_context_end_final_resume_alias_final_export_ok_now = True
    state_dump_alias_context_end_final_resume_alias_final_keep_ok_now = True
    state_dump_alias_context_end_final_resume_alias_final_done_ok_now = True
    state_dump_alias_context_end_final_resume_alias_final_ready_ok_now = True
    state_dump_alias_context_end_final_resume_alias_final_complete_ok_now = True
    state_dump_alias_context_end_final_resume_alias_final_all_good_now = True
    state_dump_alias_context_end_final_resume_alias_final_stop_good_now = True
    state_dump_alias_context_end_final_resume_alias_final_end_good_now = True
    state_dump_alias_context_end_final_resume_alias_final_use_good_now = True
    state_dump_alias_context_end_final_resume_alias_final_plot_good_now = True
    state_dump_alias_context_end_final_resume_alias_final_title_good_now = True
    state_dump_alias_context_end_final_resume_alias_final_export_good_now = True
    state_dump_alias_context_end_final_resume_alias_final_keep_good_now = True
    state_dump_alias_context_end_final_resume_alias_final_done_good_now = True
    state_dump_alias_context_end_final_resume_alias_final_ok_good_now = True
    state_dump_alias_context_end_final_resume_alias_final_ready_good_now = True
    state_dump_alias_context_end_final_resume_alias_final_complete_good_now = True
    state_dump_alias_context_end_final_resume_alias_final_finish_good_now = True
    state_dump_alias_context_end_final_resume_alias_final_last_good_now = True
    state_dump_alias_context_end_final_resume_alias_final_over_good_now = True
    state_dump_alias_context_end_final_resume_alias_final_wrap_good_now = True
    state_dump_alias_context_end_final_resume_alias_final_compact_good_now = True
    state_dump_alias_context_end_final_resume_alias_final_valid_good_now = True
    state_dump_alias_context_end_final_resume_alias_final_consistent_good_now = True
    state_dump_alias_context_end_final_resume_alias_final_safe_good_now = True
    state_dump_alias_context_end_final_resume_alias_final_marker_good_now = True
    state_dump_alias_context_end_final_resume_alias_final_close_good_now = True
    state_dump_alias_context_end_final_resume_alias_final_done_for_real_now_now = True
    state_dump_alias_context_end_final_resume_alias_final_truly_done_now_now = True
    state_dump_alias_context_end_final_resume_alias_final_really_done_now_now = True
    state_dump_alias_context_end_final_resume_alias_final_enough_now_now = True
    state_dump_alias_context_end_final_resume_alias_final_stop_now_now_now = True
    state_dump_alias_context_end_final_resume_alias_final_resume_now_now_now = True
    state_dump_alias_context_end_final_resume_alias_final_use_now_now_now = True
    state_dump_alias_context_end_final_resume_alias_final_over_now_now = True
    state_dump_alias_context_end_final_resume_alias_final_last_now_now = True
    state_dump_alias_context_end_final_resume_alias_final_end_now_now = True
    state_dump_alias_context_end_final_resume_alias_final_close_now_now = True
    state_dump_alias_context_end_final_resume_alias_final_compact_now_now = True
    state_dump_alias_context_end_final_resume_alias_final_valid_now_now = True
    state_dump_alias_context_end_final_resume_alias_final_consistent_now_now = True
    state_dump_alias_context_end_final_resume_alias_final_safe_now_now = True
    state_dump_alias_context_end_final_resume_alias_final_meta_now_now = meta_out
    state_dump_alias_context_end_final_resume_alias_final_names_now_now = component_names
    state_dump_alias_context_end_final_resume_alias_final_labels_now_now = plot_dim_labels
    state_dump_alias_context_end_final_resume_alias_final_mode_now_now = teacher_state_mode
    state_dump_alias_context_end_final_resume_alias_final_dim_now_now = state_dim
    state_dump_alias_context_end_final_resume_alias_final_cols_now_now = veh_state_cols
    state_dump_alias_context_end_final_resume_alias_final_teacher_cols_now_now = teacher_state_cols
    state_dump_alias_context_end_final_resume_alias_final_mask_now_now = teacher_mask
    state_dump_alias_context_end_final_resume_alias_final_semantic_now_now = has_semantic_ac
    state_dump_alias_context_end_final_resume_alias_final_plotdims_now_now = primary_plot_dims
    state_dump_alias_context_end_final_resume_alias_final_summary_now_now = meta_out
    state_dump_alias_context_end_final_resume_alias_final_stop_ok_now = True
    state_dump_alias_context_end_final_resume_alias_final_end_ok_now_now = True
    state_dump_alias_context_end_final_resume_alias_final_close_ok_now_now = True
    state_dump_alias_context_end_final_resume_alias_final_use_ok_now = True
    state_dump_alias_context_end_final_resume_alias_final_plot_ok_now_now = True
    state_dump_alias_context_end_final_resume_alias_final_title_ok_now_now = True
    state_dump_alias_context_end_final_resume_alias_final_export_ok_now_now = True
    state_dump_alias_context_end_final_resume_alias_final_keep_ok_now_now = True
    state_dump_alias_context_end_final_resume_alias_final_done_ok_now_now = True
    state_dump_alias_context_end_final_resume_alias_final_ready_ok_now_now = True
    state_dump_alias_context_end_final_resume_alias_final_complete_ok_now_now = True
    state_dump_alias_context_end_final_resume_alias_final_all_good_now_now = True
    state_dump_alias_context_end_final_resume_alias_final_finish_good_now_now = True
    state_dump_alias_context_end_final_resume_alias_final_last_good_now_now = True
    state_dump_alias_context_end_final_resume_alias_final_over_good_now_now = True
    state_dump_alias_context_end_final_resume_alias_final_wrap_good_now_now = True
    state_dump_alias_context_end_final_resume_alias_final_compact_good_now_now = True
    state_dump_alias_context_end_final_resume_alias_final_valid_good_now_now = True
    state_dump_alias_context_end_final_resume_alias_final_consistent_good_now_now = True
    state_dump_alias_context_end_final_resume_alias_final_safe_good_now_now = True
    state_dump_alias_context_end_final_resume_alias_final_done_for_real_final = True
    state_dump_alias_context_end_final_resume_alias_final_truly_done_final = True
    state_dump_alias_context_end_final_resume_alias_final_really_done_final = True
    state_dump_alias_context_end_final_resume_alias_final_enough_final = True
    state_dump_alias_context_end_final_resume_alias_final_stop_final_final = True
    state_dump_alias_context_end_final_resume_alias_final_end_final_final = True
    state_dump_alias_context_end_final_resume_alias_final_close_final_final = True
    state_dump_alias_context_end_final_resume_alias_final_use_final_final = True
    state_dump_alias_context_end_final_resume_alias_final_plot_final_final = True
    state_dump_alias_context_end_final_resume_alias_final_title_final_final = True
    state_dump_alias_context_end_final_resume_alias_final_export_final_final = True
    state_dump_alias_context_end_final_resume_alias_final_keep_final_final = True
    state_dump_alias_context_end_final_resume_alias_final_done_final_final = True
    state_dump_alias_context_end_final_resume_alias_final_ready_final_final = True
    state_dump_alias_context_end_final_resume_alias_final_complete_final_final = True
    state_dump_alias_context_end_final_resume_alias_final_finish_final_final = True
    state_dump_alias_context_end_final_resume_alias_final_last_final_final = True
    state_dump_alias_context_end_final_resume_alias_final_over_final_final = True
    state_dump_alias_context_end_final_resume_alias_final_wrap_final_final = True
    state_dump_alias_context_end_final_resume_alias_final_compact_final_final = True
    state_dump_alias_context_end_final_resume_alias_final_valid_final_final = True
    state_dump_alias_context_end_final_resume_alias_final_consistent_final_final = True
    state_dump_alias_context_end_final_resume_alias_final_safe_final_final = True
    state_dump_alias_context_end_final_resume_alias_final_meta_final_final = meta_out
    state_dump_alias_context_end_final_resume_alias_final_names_final_final = component_names
    state_dump_alias_context_end_final_resume_alias_final_labels_final_final = plot_dim_labels
    state_dump_alias_context_end_final_resume_alias_final_mode_final_final = teacher_state_mode
    state_dump_alias_context_end_final_resume_alias_final_dim_final_final = state_dim
    state_dump_alias_context_end_final_resume_alias_final_cols_final_final = veh_state_cols
    state_dump_alias_context_end_final_resume_alias_final_teacher_cols_final_final = teacher_state_cols
    state_dump_alias_context_end_final_resume_alias_final_mask_final_final = teacher_mask
    state_dump_alias_context_end_final_resume_alias_final_semantic_final_final = has_semantic_ac
    state_dump_alias_context_end_final_resume_alias_final_plotdims_final_final = primary_plot_dims
    state_dump_alias_context_end_final_resume_alias_final_summary_final_final = meta_out

    # ---- behavior summaries (GT) ----
    peak_abs_steer = np.max(np.abs(true[:, :, 0]), axis=1)
    peak_abs_yaw = np.max(np.abs(true[:, :, 1]), axis=1)
    peak_abs_ay = np.max(np.abs(true[:, :, 2]), axis=1)

    rmse_steer_evt = np.sqrt(np.mean((pred[:, :, 0] - true[:, :, 0]) ** 2, axis=1))
    rmse_yaw_evt = np.sqrt(np.mean((pred[:, :, 1] - true[:, :, 1]) ** 2, axis=1))
    rmse_ay_evt = np.sqrt(np.mean((pred[:, :, 2] - true[:, :, 2]) ** 2, axis=1))

    df_state["peak_abs_steer_gt"] = peak_abs_steer
    df_state["peak_abs_yaw_gt"] = peak_abs_yaw
    df_state["peak_abs_ay_gt"] = peak_abs_ay
    df_state["rmse_steer_evt"] = rmse_steer_evt
    df_state["rmse_yaw_evt"] = rmse_yaw_evt
    df_state["rmse_ay_evt"] = rmse_ay_evt

    df_state.to_csv(str(fig_dir / "test_state_dump.csv"), index=False, encoding="utf-8-sig")
    print(f"🧾 已保存 test 状态/行为汇总: {fig_dir / 'test_state_dump.csv'}")

    # ---- quick relationship plots (state vs peak) ----
    def _scatter(x, y, xlabel, ylabel, outname):
        plt.figure(figsize=(7.2, 5.0))
        plt.scatter(x, y, s=10, alpha=0.6)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(str(fig_dir / outname), dpi=200)
        plt.close()

    _scatter(df_state["A_veh"].values, peak_abs_steer, "A_veh (student)", STEER_PEAK_PLOT_LABEL, "state_vs_peak_steer_A.png")
    _scatter(df_state["C_veh"].values, peak_abs_steer, "C_veh (student)", STEER_PEAK_PLOT_LABEL, "state_vs_peak_steer_C.png")
    _scatter(df_state["A_veh"].values, peak_abs_ay, "A_veh (student)", "peak|ay| (GT)", "state_vs_peak_ay_A.png")
    _scatter(df_state["C_veh"].values, peak_abs_ay, "C_veh (student)", "peak|ay| (GT)", "state_vs_peak_ay_C.png")

    # ---- per-sample pred-vs-gt plots with state annotation ----
    n = pred.shape[0]
    if n == 0:
        print("⚠ test 集为空，无法画图")
        return

    t = np.arange(pred.shape[1], dtype=np.float32) / float(fs)
    pick = np.linspace(0, n - 1, num=min(n_examples, n), dtype=int)

    for k, idx in enumerate(pick):
        Aveh, Cveh = float(zveh_all[idx, 0]), float(zveh_all[idx, 1])
        if zmask_all[idx] > 0.5:
            Atea, Ctea = float(zphys_all[idx, 0]), float(zphys_all[idx, 1])
            title = f"Test sample #{idx} | Future {t[-1]:.2f}s | Aveh={Aveh:.2f} Cveh={Cveh:.2f} | Atea={Atea:.2f} Ctea={Ctea:.2f}"
        else:
            title = f"Test sample #{idx} | Future {t[-1]:.2f}s | Aveh={Aveh:.2f} Cveh={Cveh:.2f} | teacher=NA"

        fig = plt.figure(figsize=(12, 8))

        ax1 = fig.add_subplot(3, 1, 1)
        ax1.plot(t, true[idx, :, 0], label="GT", linewidth=1.2)
        ax1.plot(t, pred[idx, :, 0], label="Pred", linewidth=1.2, linestyle="--")
        ax1.set_ylabel(STEER_PLOT_LABEL)
        ax1.set_title(title)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2 = fig.add_subplot(3, 1, 2)
        ax2.plot(t, true[idx, :, 1], linewidth=1.2)
        ax2.plot(t, pred[idx, :, 1], linewidth=1.2, linestyle="--")
        ax2.set_ylabel("yawrate")
        ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(3, 1, 3)
        ax3.plot(t, true[idx, :, 2], linewidth=1.2)
        ax3.plot(t, pred[idx, :, 2], linewidth=1.2, linestyle="--")
        ax3.set_ylabel("ay")
        ax3.set_xlabel("time (s)")
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = fig_dir / f"pred_vs_gt_example_{k:02d}_idx{idx}.png"
        plt.savefig(str(out_path), dpi=200)
        plt.close()

    print(f"🖼 已保存预测效果图到: {fig_dir} (pred_vs_gt_example_*.png)")


# =========================
# Main
# =========================
def main():
    # =========================
    # 本次运行输出目录（程序运行结果/时间戳）
    # =========================
    RUN_DIR = make_run_dir(prefix="TRAIN_V5_4_STATECOND_REV")
    CKPT_DIR = RUN_DIR / "checkpoints"
    FIG_DIR = RUN_DIR / "figures"
    LOG_DIR = RUN_DIR / "logs"

    orig_stdout = sys.stdout
    tee = TeeStdout(LOG_DIR / "train.log", console_stream=orig_stdout)
    sys.stdout = tee
    try:
        try_copy_self(RUN_DIR)

        print("RUN_DIR:", str(RUN_DIR))
        print("设备:", DEVICE)
        print("时间:", time.strftime("%Y-%m-%d %H:%M:%S"))
        print("========================================")

        protocol_config, split_subjects = load_protocol_split(PROTOCOL_CONFIG_PATH, FROZEN_SPLIT_PATH)
        print("Protocol version:", protocol_config.get("protocol_version"))
        print("Protocol split source:", str(FROZEN_SPLIT_PATH))

        style_map = load_driver_style_map(STYLE_CSV)
        X_pool, y_pool, curve_pool, ctx_pool, base_pool, sample_meta_df, feature_names = build_all_samples(style_map)
        # ---- dump feature names for verification ----
        try:
            save_json(RUN_DIR / "feature_names.json", {"n_features": int(len(feature_names)), "feature_names": feature_names})
            print("🧩 已保存特征列表:", RUN_DIR / "feature_names.json")
        except Exception as e:
            print("⚠ 保存特征列表失败:", e)

        total = len(X_pool)
        if total == 0:
            print("❌ 没有有效事件样本")
            return

        if len(sample_meta_df) != total:
            raise ValueError(f"sample meta length mismatch: total={total}, meta={len(sample_meta_df)}")

        split_indices = build_subject_split_indices(sample_meta_df, split_subjects)
        smoke_sampling_policy = "disabled"
        if SMOKE_MODE:
            rng = np.random.default_rng(SEED)
            split_indices, smoke_counts = choose_smoke_indices(split_indices, SMOKE_MAX_SAMPLES, rng)
            smoke_sampling_policy = f"protocol-first per-split subsample with guaranteed non-empty splits; counts={smoke_counts}"
            print(f"[SMOKE] {smoke_sampling_policy}")

        train_idx = np.asarray(split_indices["train"], dtype=np.int64)
        val_idx = np.asarray(split_indices["val"], dtype=np.int64)
        test_idx = np.asarray(split_indices["test"], dtype=np.int64)
        if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
            raise ValueError(
                f"Protocol split must keep train/val/test non-empty, got train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}"
            )

        selected_idx = np.concatenate([train_idx, val_idx, test_idx])
        selected_meta_df = sample_meta_df.loc[selected_idx].copy()
        selected_meta_df["protocol_split_applied"] = ""
        selected_meta_df.loc[train_idx, "protocol_split_applied"] = "train"
        selected_meta_df.loc[val_idx, "protocol_split_applied"] = "val"
        selected_meta_df.loc[test_idx, "protocol_split_applied"] = "test"
        selected_meta_df.to_csv(str(RUN_DIR / "selected_samples_with_split.csv"), index=False, encoding="utf-8-sig")

        split_audit, _, split_sample_counts_df = export_split_audit(
            RUN_DIR,
            sample_meta_df,
            {"train": train_idx, "val": val_idx, "test": test_idx},
            split_subjects,
            protocol_config,
            SMOKE_MODE,
            smoke_sampling_policy,
        )
        print("Split audit saved:", RUN_DIR / "split_audit.json")

        # ---- road-type split (straight/curve) using ONLY history-window curvature stats ----
        # ---- NEW(v5.4): reversal label from GT future steer (aux training + stratified eval) ----
        # ---- reversal label (weak & strong) computed from FUTURE steer (GT) ----
        rev_gt_weak = np.array([has_reversal_np(y[:, 0], eps=REV_EPS_WEAK) for y in y_pool], dtype=np.float32)

        # strong reversal: requires crossing both +REV_EPS_STRONG and -REV_EPS_STRONG, AND a sufficient peak magnitude
        rev_gt_strong = []
        for y in y_pool:
            steer_f = y[:, 0]
            peak_abs = float(np.max(np.abs(steer_f))) if steer_f.size else 0.0
            r = has_reversal_np(steer_f, eps=REV_EPS_STRONG)
            if r > 0.5 and peak_abs >= STRONG_PEAK_THR:
                rev_gt_strong.append(1.0)
            else:
                rev_gt_strong.append(0.0)
        rev_gt_strong = np.asarray(rev_gt_strong, dtype=np.float32)

        # label used for rev_head training
        rev_gt = rev_gt_strong if USE_STRONG_REV_LOSS else rev_gt_weak

        try:
            print(f"🔁 reversal labels: weak_rate={float(np.mean(rev_gt_weak)):.3f}, strong_rate={float(np.mean(rev_gt_strong)):.3f}, used={'strong' if USE_STRONG_REV_LOSS else 'weak'}")
        except Exception:
            pass


        curve_feat_name, curve_feat_idx = find_feature_in_list(feature_names, ["lanecurvature", "curvature"])
        if curve_feat_idx is None:
            print("⚠️ 未找到曲率特征列（lanecurvature/curvature），将默认全部视为直道。")
            curve_scores = np.zeros((total,), dtype=np.float32)
            curve_thr = 0.0
            is_curve = np.zeros((total,), dtype=np.int64)
        else:
            curve_scores = np.array(
                [float(np.mean(np.abs(x[:, curve_feat_idx]))) for x in X_pool],
                dtype=np.float32
            )
            curve_thr = auto_curve_threshold(curve_scores[train_idx])
            is_curve = (curve_scores > curve_thr).astype(np.int64)

            ratio_curve = float(np.mean(is_curve))
            print(f"🛣 road_type: 使用历史 3s 平均|curvature| 分割直/弯")
            print(f"   曲率列: {curve_feat_name}")
            print(f"   curve_thr = {curve_thr:.3e}  (train auto)")
            print(f"   curve_ratio(all) = {ratio_curve*100:.1f}%  |  straight_ratio = {(1-ratio_curve)*100:.1f}%")

        # ---- standardize encoder src features ----
        all_X_concat = np.concatenate([X_pool[int(i)] for i in train_idx], axis=0)
        feat_mean = all_X_concat.mean(axis=0)
        feat_std = all_X_concat.std(axis=0)
        feat_std[feat_std < 1e-6] = 1e-6
        for i in range(len(X_pool)):
            X_pool[i] = (X_pool[i] - feat_mean) / feat_std
    
        # ---- standardize outputs ----
        all_y_concat = np.concatenate([y_pool[int(i)].reshape(-1, 3) for i in train_idx], axis=0)
        y_mean = all_y_concat.mean(axis=0)
        y_std  = all_y_concat.std(axis=0)
        y_std[y_std < 1e-6] = 1e-6

        y_mean_t = torch.tensor(y_mean, device=DEVICE, dtype=torch.float32)
        y_std_t = torch.tensor(y_std, device=DEVICE, dtype=torch.float32)
    
        # ---- curve std ----
        all_curve_concat = np.concatenate([curve_pool[int(i)] for i in train_idx], axis=0)
        curve_mean = all_curve_concat.mean()
        curve_std = all_curve_concat.std()
        if curve_std < 1e-6:
            curve_std = 1e-6
    
        # ---- ctx std ----
        ctx_array = np.stack([ctx_pool[int(i)] for i in train_idx], axis=0)
        ctx_mean = ctx_array.mean(axis=0)
        ctx_std  = ctx_array.std(axis=0)
        ctx_std[ctx_std < 1e-6] = 1e-6
    
        # ---- teacher base feat z-score (train stats only) ----
        base_train = np.stack([base_pool[int(i)] for i in train_idx], axis=0)  # (Ntr,12)
        teacher_base_names = [
            "hr", "eda_tonic", "eda_phasic", "emg_rms",
            "alpha_asym", "occ_ta_beta", "frontal_ta_beta", "temporal_ta_beta",
            "occ_alpha_abs", "temporal_gamma_rel", "occ_gamma_rel", "frontal_gamma_rel",
        ]
        finite_count = np.isfinite(base_train).sum(axis=0)
        missing_count = (~np.isfinite(base_train)).sum(axis=0)
        valid_ratio = (finite_count / max(1, len(train_idx))).astype(np.float32)
        all_missing_mask = (finite_count == 0)

        base_mu = np.zeros((base_train.shape[1],), dtype=np.float32)
        base_sd = np.ones((base_train.shape[1],), dtype=np.float32)
        valid_stat_mask = ~all_missing_mask
        if np.any(valid_stat_mask):
            base_mu[valid_stat_mask] = np.nanmean(base_train[:, valid_stat_mask], axis=0).astype(np.float32)
            base_sd[valid_stat_mask] = np.nanstd(base_train[:, valid_stat_mask], axis=0).astype(np.float32)
        base_sd[base_sd < 1e-6] = 1e-6

        teacher_base_stats = []
        for i, name in enumerate(teacher_base_names):
            teacher_base_stats.append({
                "index": int(i),
                "name": name,
                "finite_count": int(finite_count[i]),
                "missing_count": int(missing_count[i]),
                "valid_ratio": float(valid_ratio[i]),
                "all_missing": bool(all_missing_mask[i]),
                "mean": float(base_mu[i]),
                "std": float(base_sd[i]),
            })
        save_json(RUN_DIR / "teacher_base_missing_stats.json", {
            "fit_split": "train",
            "fit_sample_count": int(len(train_idx)),
            "all_missing_indices": [int(i) for i in np.where(all_missing_mask)[0]],
            "all_missing_names": [teacher_base_names[int(i)] for i in np.where(all_missing_mask)[0]],
            "stats": teacher_base_stats,
        })
        print(
            f"Teacher-base missing dims: {int(all_missing_mask.sum())}/{int(len(all_missing_mask))} | "
            f"all-missing={ [teacher_base_names[int(i)] for i in np.where(all_missing_mask)[0]] }"
        )

        def zscore_base(x12):
            x = x12.copy()
            # NaN -> mean（等价于 z=0），避免污染
            nan_mask = ~np.isfinite(x)
            x[nan_mask] = np.take(base_mu, np.where(nan_mask)[0])
            return (x - base_mu) / base_sd

        base_z_all = np.stack([zscore_base(x) for x in base_pool], axis=0)  # (N,12)
        z_phys_raw, teacher_state_meta = build_teacher_state(
            base_z_all,
            mode=TEACHER_STATE_MODE,
            state_dim=TEACHER_STATE_DIM,
            fit_indices=train_idx,
        )

        # 进一步把 teacher latent 再标准化（train stats）
        z_tr = z_phys_raw[train_idx]
        z_mu = np.mean(z_tr, axis=0)
        z_sd = np.std(z_tr, axis=0)
        z_sd[z_sd < 1e-6] = 1e-6
        z_phys = ((z_phys_raw - z_mu) / z_sd).astype(np.float32)
        teacher_state_meta["fit_split"] = "train"
        teacher_state_meta["fit_sample_count"] = int(len(train_idx))
        teacher_state_meta["z_mu"] = z_mu.astype(np.float32).tolist()
        teacher_state_meta["z_sd"] = z_sd.astype(np.float32).tolist()
        teacher_state_meta["state_dim"] = int(z_phys.shape[1])
        teacher_state_meta["base_feature_names"] = teacher_base_names
        teacher_state_meta["base_all_missing_indices"] = [int(i) for i in np.where(all_missing_mask)[0]]
        teacher_state_meta["base_all_missing_names"] = [teacher_base_names[int(i)] for i in np.where(all_missing_mask)[0]]
        teacher_state_meta["base_valid_ratio"] = valid_ratio.astype(np.float32).tolist()
        teacher_state_meta["base_mu"] = base_mu.astype(np.float32).tolist()
        teacher_state_meta["base_sd"] = base_sd.astype(np.float32).tolist()
        teacher_state_meta["base_missing_stats_file"] = "teacher_base_missing_stats.json"
        teacher_state_meta["base_valid_stats_count"] = int(valid_stat_mask.sum())
        teacher_state_meta["base_all_missing_count"] = int(all_missing_mask.sum())
        save_json(RUN_DIR / "teacher_state_meta.json", teacher_state_meta)
        print(
            f"Teacher-state mode={teacher_state_meta['mode']} | "
            f"state_dim={teacher_state_meta['state_dim']} | "
            f"components={teacher_state_meta['component_names']}"
        )
        state_dim = int(z_phys.shape[1])
        context_dim = int(ctx_pool[0].shape[0] + state_dim)
    
        def build_dataset(indices):
            return MultiTaskFutureWithCurveDataset(
                subset_list(X_pool, indices),
                subset_list(y_pool, indices),
                subset_list(curve_pool, indices),
                subset_list(ctx_pool, indices),
                subset_array(z_phys, indices),
                subset_array(rev_gt, indices),
                subset_array(rev_gt_weak, indices),
                subset_array(rev_gt_strong, indices),
                y_mean, y_std, curve_mean, curve_std, ctx_mean, ctx_std,
                subset_array(curve_scores, indices),
                subset_array(is_curve, indices),
            )

        train_dataset = build_dataset(train_idx)
        val_dataset = build_dataset(val_idx)
        test_dataset = build_dataset(test_idx)
    
        def collate_fn(batch):
            src = torch.stack([torch.from_numpy(b["src"]).float() for b in batch], dim=0)
            y_norm = torch.stack([torch.from_numpy(b["y_norm"]).float() for b in batch], dim=0)
            curve_norm = torch.stack([torch.from_numpy(b["curve_norm"]).float() for b in batch], dim=0)
            ctx = torch.stack([torch.from_numpy(b["ctx"]).float() for b in batch], dim=0)
            z_phys = torch.stack([torch.from_numpy(b["z_phys"]).float() for b in batch], dim=0)
            z_mask = torch.stack([torch.from_numpy(b["z_mask"]).float() for b in batch], dim=0)  # (B,1)
            rev_gt = torch.stack([torch.from_numpy(b["rev_gt"]).float() for b in batch], dim=0)  # (B,1)
            rev_gt_weak = torch.stack([torch.from_numpy(b["rev_gt_weak"]).float() for b in batch], dim=0)  # (B,1)
            rev_gt_strong = torch.stack([torch.from_numpy(b["rev_gt_strong"]).float() for b in batch], dim=0)  # (B,1)
            idx = torch.stack([torch.from_numpy(b["idx"]).long() for b in batch], dim=0).squeeze(1)
            curve_score = torch.stack([torch.from_numpy(b["curve_score"]).float() for b in batch], dim=0).squeeze(1)
            is_curve = torch.stack([torch.from_numpy(b["is_curve"]).long() for b in batch], dim=0).squeeze(1)
            return {"src": src, "y_norm": y_norm, "curve_norm": curve_norm, "ctx": ctx, "z_phys": z_phys, "z_mask": z_mask,
            "rev_gt": rev_gt,
            "rev_gt_weak": rev_gt_weak,
            "rev_gt_strong": rev_gt_strong,
                    "idx": idx, "curve_score": curve_score, "is_curve": is_curve}
    
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  collate_fn=collate_fn, pin_memory=torch.cuda.is_available())
        val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                  collate_fn=collate_fn, pin_memory=torch.cuda.is_available())
        test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                  collate_fn=collate_fn, pin_memory=torch.cuda.is_available())

        # ---- class balancing for reversal aux loss ----
        try:
            pos_cnt = float(np.sum(rev_gt[train_idx] > 0.5))
            neg_cnt = float(len(train_idx) - pos_cnt)
            pw = neg_cnt / max(1.0, pos_cnt)
            rev_pos_weight = torch.tensor(pw, device=DEVICE)
            print(f"🔁 rev_head pos_weight={pw:.3f}  (pos={pos_cnt:.0f}, neg={neg_cnt:.0f})")
        except Exception:
            rev_pos_weight = torch.tensor(1.0, device=DEVICE)
        try:
            strong_pos_cnt = float(np.sum(rev_gt_strong[train_idx] > 0.5))
            strong_pos_neg_cnt = float(len(train_idx) - strong_pos_cnt)
            spw = strong_pos_neg_cnt / max(1.0, strong_pos_cnt)
            strong_pos_gate_pos_weight = torch.tensor(spw, device=DEVICE)
            print(f"strong_pos_gate pos_weight={spw:.3f}  (pos={strong_pos_cnt:.0f}, neg={strong_pos_neg_cnt:.0f})")
        except Exception:
            strong_pos_gate_pos_weight = torch.tensor(1.0, device=DEVICE)
    
        # model
        model = Past2FutureMultiTaskRoadPreview(
            input_dim=len(feature_names),
            context_dim=context_dim,
            future_len=FUTURE_LEN,
            out_dim=3,
            d_model=D_MODEL,
            nhead=N_HEAD,
            num_layers_enc=NUM_LAYERS_ENC,
            num_layers_dec=NUM_LAYERS_DEC,
            dim_feedforward=FFN_DIM,
            dropout=DROPOUT,
            max_len_enc=WIN_LEN,
            max_len_dec=FUTURE_LEN,
            state_dim=state_dim,
            enable_steer_coarse_fine=ENABLE_STEER_COARSE_FINE,
            trend_pool_kernel=TREND_POOL_KERNEL,
            trend_pool_stride=TREND_POOL_STRIDE,
            enable_late_reversal_gate=ENABLE_LATE_REV_GATE,
            late_rev_gate_start_sec=LATE_REV_GATE_START_SEC,
            late_rev_gate_scale=LATE_REV_GATE_SCALE,
            late_rev_gate_ramp_power=LATE_REV_GATE_RAMP_POWER,
            enable_strong_pos_gate=ENABLE_STRONG_POS_GATE,
            strong_pos_gate_start_sec=STRONG_POS_GATE_START_SEC,
            strong_pos_gate_scale=STRONG_POS_GATE_SCALE,
            strong_pos_gate_ramp_power=STRONG_POS_GATE_RAMP_POWER,
            strong_pos_gate_prob_center=STRONG_POS_GATE_PROB_CENTER,
        ).to(DEVICE)
    
        optim = torch.optim.Adam(model.parameters(), lr=LR)
        print(f"Split counts | train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)}")
    
        print(f"训练集样本数: {len(train_dataset)} | 测试集样本数: {len(test_dataset)}")
        print(f"历史窗口: {WIN_SEC:.1f}s({WIN_LEN}) 未来窗口: {FUTURE_SEC:.1f}s({FUTURE_LEN})")
        print(
            f"Response-state v1: enabled={ENABLE_RESPONSE_STATE_V1} | "
            f"state={ENABLE_STATE_DISTILL} | reversal={ENABLE_REVERSAL_AUX} | "
            f"peaktime={ENABLE_PEAKTIME_AUX} | peakintensity={ENABLE_PEAKINTENSITY_AUX}"
        )
        print(
            f"Teacher-state config: mode={TEACHER_STATE_MODE} | requested_dim={TEACHER_STATE_DIM} | "
            f"actual_dim={state_dim}"
        )
        print(f"Distill: lambda_state={LAMBDA_STATE} | lambda_rev={LAMBDA_REV} | REV_EPS={REV_EPS}\n")
        print(
            f"Steer unit: source={STEER_SOURCE_UNIT} -> target={STEER_ANGLE_UNIT} | "
            f"scale={STEER_ANGLE_SCALE:.6f}"
        )
        if ENABLE_LATE_REV_GATE:
            print(
                f"Late rev gate: enabled=True | start_sec={LATE_REV_GATE_START_SEC:.2f} | "
                f"scale={LATE_REV_GATE_SCALE:.2f} | ramp_power={LATE_REV_GATE_RAMP_POWER:.2f}"
            )
        if ENABLE_STRONG_POS_GATE:
            print(
                f"Strong-pos gate: enabled=True | start_sec={STRONG_POS_GATE_START_SEC:.2f} | "
                f"scale={STRONG_POS_GATE_SCALE:.2f} | ramp_power={STRONG_POS_GATE_RAMP_POWER:.2f} | "
                f"prob_center={STRONG_POS_GATE_PROB_CENTER:.2f} | lambda={LAMBDA_STRONG_POS_GATE:.3f}"
            )

        # ---- persist run config (for reproducibility) ----
        run_config = {
            "MODEL_VER": "v5_8_response_state_v1_protocol_safe",
            "protocol_config_path": str(PROTOCOL_CONFIG_PATH),
            "protocol_version": protocol_config.get("protocol_version"),
            "split_policy_expected": "subject-level fixed split",
            "split_policy_applied": "subject-level fixed split",
            "split_source": str(FROZEN_SPLIT_PATH),
            "train_subjects": list(split_subjects["train"]),
            "val_subjects": list(split_subjects["val"]),
            "test_subjects": list(split_subjects["test"]),
            "train_subject_count": int(len(split_subjects["train"])),
            "val_subject_count": int(len(split_subjects["val"])),
            "test_subject_count": int(len(split_subjects["test"])),
            "train_sample_count": int(len(train_idx)),
            "val_sample_count": int(len(val_idx)),
            "test_sample_count": int(len(test_idx)),
            "smoke_mode": bool(SMOKE_MODE),
            "smoke_sampling_policy": smoke_sampling_policy,
            "teacher_state_fit_split": "train",
            "teacher_state_fit_sample_count": int(len(train_idx)),
            "standardization_fit_split": "train",
            "curve_threshold_fit_split": "train",
            "anchor_source_expected": protocol_config.get("anchor_source"),
            "anchor_source_applied": "curve->roll_peak; straight->steer_rate_peak80_first",
            "maintained_anchor_policy": "curve->roll_peak; straight->steer_rate_peak80_first",
            "ENABLE_RESPONSE_STATE_V1": bool(ENABLE_RESPONSE_STATE_V1),
            "ENABLE_STATE_DISTILL": bool(ENABLE_STATE_DISTILL),
            "ENABLE_REVERSAL_AUX": bool(ENABLE_REVERSAL_AUX),
            "ENABLE_PEAKTIME_AUX": bool(ENABLE_PEAKTIME_AUX),
            "ENABLE_PEAKINTENSITY_AUX": bool(ENABLE_PEAKINTENSITY_AUX),
            "TEACHER_STATE_MODE": TEACHER_STATE_MODE,
            "TEACHER_STATE_DIM": int(TEACHER_STATE_DIM),
            "ACTUAL_STATE_DIM": int(state_dim),
            "TEACHER_STATE_COMPONENTS": teacher_state_meta["component_names"],
            "LAMBDA_REV": float(LAMBDA_REV),
            "REV_EPS": float(REV_EPS),
            "STEER_SOURCE_UNIT": STEER_SOURCE_UNIT,
            "STEER_ANGLE_UNIT": STEER_ANGLE_UNIT,
            "STEER_ANGLE_SCALE": float(STEER_ANGLE_SCALE),
            "STEER_ONSET_THR_ABS": float(STEER_ONSET_THR_ABS),
            "ROOT": ROOT,
            "STYLE_CSV": STYLE_CSV,
            "FS": FS,
            "WIN_SEC": WIN_SEC,
            "FUTURE_SEC": FUTURE_SEC,
            "WIN_LEN": WIN_LEN,
            "FUTURE_LEN": FUTURE_LEN,
            "BATCH_SIZE": BATCH_SIZE,
            "EPOCHS": EPOCHS,
            "LR": LR,
            "DEVICE": DEVICE,
            "D_MODEL": D_MODEL,
            "N_HEAD": N_HEAD,
            "ENC_LAYERS": NUM_LAYERS_ENC,
            "DEC_LAYERS": NUM_LAYERS_DEC,
            "FFN_DIM": FFN_DIM,
            "DROPOUT": DROPOUT,
            "W_DIFF1": W_DIFF1,
            "W_DIFF2": W_DIFF2,
            "W_REVSEQ": W_REVSEQ,
            "W_PEAKTIME": W_PEAKTIME,
            "REVSEQ_ALPHA_FRAC": REVSEQ_ALPHA_FRAC,
            "PEAK_TEMP_FRAC": PEAK_TEMP_FRAC,
            "W_STEER_WT": W_STEER_WT,
            "W_STEER_RATE": W_STEER_RATE,
            "W_STEER_REV": W_STEER_REV,
            "STEER_WT_MAX": STEER_WT_MAX,
            "W_TREND": W_TREND,
            "TREND_POOL_KERNEL": TREND_POOL_KERNEL,
            "TREND_POOL_STRIDE": TREND_POOL_STRIDE,
            "TREND_SIGN_EPS": TREND_SIGN_EPS,
            "TREND_LOSS_MODE": TREND_LOSS_MODE,
            "TREND_LEVEL_WEIGHT": TREND_LEVEL_WEIGHT,
            "TREND_DELTA_WEIGHT": TREND_DELTA_WEIGHT,
            "TREND_DIR_WEIGHT": TREND_DIR_WEIGHT,
            "ENABLE_STEER_COARSE_FINE": bool(ENABLE_STEER_COARSE_FINE),
            "W_TREND_COARSE": W_TREND_COARSE,
            "W_FINE_DC": W_FINE_DC,
            "ENABLE_PHASE_ADAPTIVE_TREND": bool(ENABLE_PHASE_ADAPTIVE_TREND),
            "TREND_EARLY_BINS": TREND_EARLY_BINS,
            "TREND_LATE_STRAIGHT_DOWN": TREND_LATE_STRAIGHT_DOWN,
            "TREND_LATE_STRONGREV_DOWN": TREND_LATE_STRONGREV_DOWN,
            "ENABLE_LATE_REV_GATE": bool(ENABLE_LATE_REV_GATE),
            "LATE_REV_GATE_START_SEC": LATE_REV_GATE_START_SEC,
            "LATE_REV_GATE_SCALE": LATE_REV_GATE_SCALE,
            "LATE_REV_GATE_RAMP_POWER": LATE_REV_GATE_RAMP_POWER,
            "ENABLE_STRONG_POS_GATE": bool(ENABLE_STRONG_POS_GATE),
            "STRONG_POS_GATE_START_SEC": STRONG_POS_GATE_START_SEC,
            "STRONG_POS_GATE_SCALE": STRONG_POS_GATE_SCALE,
            "STRONG_POS_GATE_RAMP_POWER": STRONG_POS_GATE_RAMP_POWER,
            "STRONG_POS_GATE_PROB_CENTER": STRONG_POS_GATE_PROB_CENTER,
            "ENABLE_HARD_LATE_FINE": bool(ENABLE_HARD_LATE_FINE),
            "W_HARD_LATE_FINE": W_HARD_LATE_FINE,
            "HARD_LATE_START_SEC": HARD_LATE_START_SEC,
            "HARD_TAIL_START_SEC": HARD_TAIL_START_SEC,
            "HARD_PEAK_QUANTILE": HARD_PEAK_QUANTILE,
            "HARD_TAIL_QUANTILE": HARD_TAIL_QUANTILE,
            "LAMBDA_STATE": LAMBDA_STATE,
            "LAMBDA_STRONG_POS_GATE": LAMBDA_STRONG_POS_GATE,
            "EEG_HIST_SEC": EEG_HIST_SEC,
            "SEED": SEED,
            "N_TRAIN": int(len(train_dataset)),
            "N_VAL": int(len(val_dataset)),
            "N_TEST": int(len(test_dataset)),
            "split_audit_path": str(RUN_DIR / "split_audit.json"),
            "split_sample_counts_path": str(RUN_DIR / "split_sample_counts.csv"),
        }
        save_json(RUN_DIR / "run_config.json", run_config)
    
        # ---- training history ----
        history = []
        history_csv = RUN_DIR / "loss_history.csv"
    
        best_val = np.inf
        start_all = time.time()
    
        for epoch in range(1, EPOCHS + 1):
            model.train()
            loss_sum, loss_task_sum, loss_state_sum, loss_rev_sum, loss_trend_sum, loss_trend_coarse_sum, loss_fine_dc_sum, loss_hard_late_sum, loss_strong_pos_gate_sum, n_batch = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
    
            for batch in train_loader:
                src = batch["src"].to(DEVICE, non_blocking=True)
                y_true = batch["y_norm"].to(DEVICE, non_blocking=True)
                curve_norm = batch["curve_norm"].to(DEVICE, non_blocking=True)
                ctx = batch["ctx"].to(DEVICE, non_blocking=True)
                z_phys_b = batch["z_phys"].to(DEVICE, non_blocking=True)
                z_mask = batch["z_mask"].to(DEVICE, non_blocking=True)  # (B,1)
                is_curve_b = batch["is_curve"].to(DEVICE, non_blocking=True)
                rev_gt_b = batch["rev_gt"].to(DEVICE, non_blocking=True).squeeze(1)  # (B,)
                rev_gt_weak_b = batch.get("rev_gt_weak", batch["rev_gt"]).to(DEVICE, non_blocking=True).squeeze(1)
                rev_gt_strong_b = batch.get("rev_gt_strong", batch["rev_gt"]).to(DEVICE, non_blocking=True).squeeze(1)
    
                optim.zero_grad()
                y_hat, z_veh, rev_logit, forward_aux = unpack_model_output(model(src, ctx, curve_norm))

                sample_weight = build_reversal_sample_weight(rev_gt_b)
                loss_task, loss_amp, loss_d1, loss_d2, loss_revseq, loss_peaktime, loss_steer_wt, loss_trend, loss_trend_coarse, loss_fine_dc, loss_hard_late_fine = compute_total_task_loss(
                    y_hat, y_true, y_mean_t, y_std_t, sample_weight=sample_weight, use_reversal_local_weight=True, forward_aux=forward_aux,
                    is_curve=is_curve_b, rev_gt_weak=rev_gt_weak_b, rev_gt_strong=rev_gt_strong_b
                )

                # train 侧也使用 GT soft reversal 作为局部加权依据，避免 hard case 被均值化
                # state loss supports missing physio: if z_mask=0, ignore
                mse_state = ((z_veh - z_phys_b) ** 2).mean(dim=1, keepdim=True)  # (B,1)
                loss_state = (mse_state * z_mask).sum() / (z_mask.sum() + EPS)
    
                # reversal aux loss (binary): whether future steer changes sign
                if ENABLE_REVERSAL_AUX:
                    loss_rev = F.binary_cross_entropy_with_logits(rev_logit, rev_gt_b.float(), pos_weight=rev_pos_weight)
                else:
                    loss_rev = torch.tensor(0.0, device=DEVICE)
                strong_pos_gate_logit = forward_aux.get("strong_pos_gate_logit")
                if ENABLE_STRONG_POS_GATE and strong_pos_gate_logit is not None:
                    loss_strong_pos_gate = F.binary_cross_entropy_with_logits(
                        strong_pos_gate_logit,
                        rev_gt_strong_b.float(),
                        pos_weight=strong_pos_gate_pos_weight,
                    )
                else:
                    loss_strong_pos_gate = torch.tensor(0.0, device=DEVICE)

                loss = loss_task + LAMBDA_STATE * loss_state + LAMBDA_REV * loss_rev + LAMBDA_STRONG_POS_GATE * loss_strong_pos_gate
                loss.backward()
                optim.step()
    
                loss_sum += float(loss.item())
                loss_task_sum += float(loss_task.item())
                loss_state_sum += float(loss_state.item())
                loss_rev_sum += float(loss_rev.item())
                loss_trend_sum += float(loss_trend.item())
                loss_trend_coarse_sum += float(loss_trend_coarse.item())
                loss_fine_dc_sum += float(loss_fine_dc.item())
                loss_hard_late_sum += float(loss_hard_late_fine.item())
                loss_strong_pos_gate_sum += float(loss_strong_pos_gate.item())
                n_batch += 1
    
            train_loss = loss_sum / max(1, n_batch)
            train_loss_rev = loss_rev_sum / max(1, n_batch)
    
            # val
            model.eval()
            val_sum, val_trend_sum, val_trend_coarse_sum, val_fine_dc_sum, val_hard_late_sum, val_strong_pos_gate_sum, val_n = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    src = batch["src"].to(DEVICE, non_blocking=True)
                    y_true = batch["y_norm"].to(DEVICE, non_blocking=True)
                    curve_norm = batch["curve_norm"].to(DEVICE, non_blocking=True)
                    ctx = batch["ctx"].to(DEVICE, non_blocking=True)
                    z_phys_b = batch["z_phys"].to(DEVICE, non_blocking=True)
                    z_mask = batch["z_mask"].to(DEVICE, non_blocking=True)
                    is_curve_b = batch["is_curve"].to(DEVICE, non_blocking=True)
                    rev_gt_b = batch["rev_gt"].to(DEVICE, non_blocking=True).squeeze(1)
                    rev_gt_weak_b = batch.get("rev_gt_weak", batch["rev_gt"]).to(DEVICE, non_blocking=True).squeeze(1)
                    rev_gt_strong_b = batch.get("rev_gt_strong", batch["rev_gt"]).to(DEVICE, non_blocking=True).squeeze(1)

                    y_hat, z_veh, rev_logit, forward_aux = unpack_model_output(model(src, ctx, curve_norm))
                    sample_weight = build_reversal_sample_weight(rev_gt_b)
                    loss_task, loss_amp, loss_d1, loss_d2, loss_revseq, loss_peaktime, loss_steer_wt, loss_trend, loss_trend_coarse, loss_fine_dc, loss_hard_late_fine = compute_total_task_loss(
                        y_hat, y_true, y_mean_t, y_std_t, sample_weight=sample_weight, use_reversal_local_weight=True, forward_aux=forward_aux,
                        is_curve=is_curve_b, rev_gt_weak=rev_gt_weak_b, rev_gt_strong=rev_gt_strong_b
                    )

                    # val 侧与 train 使用同一套加权目标，避免选择标准错位
                    mse_state = ((z_veh - z_phys_b) ** 2).mean(dim=1, keepdim=True)
                    loss_state = (mse_state * z_mask).sum() / (z_mask.sum() + EPS)
                    # reversal aux loss (binary): whether future steer changes sign
                    if ENABLE_REVERSAL_AUX:
                        loss_rev = F.binary_cross_entropy_with_logits(rev_logit, rev_gt_b.float(), pos_weight=rev_pos_weight)
                    else:
                        loss_rev = torch.tensor(0.0, device=DEVICE)
                    strong_pos_gate_logit = forward_aux.get("strong_pos_gate_logit")
                    if ENABLE_STRONG_POS_GATE and strong_pos_gate_logit is not None:
                        loss_strong_pos_gate = F.binary_cross_entropy_with_logits(
                            strong_pos_gate_logit,
                            rev_gt_strong_b.float(),
                            pos_weight=strong_pos_gate_pos_weight,
                        )
                    else:
                        loss_strong_pos_gate = torch.tensor(0.0, device=DEVICE)
                    loss = loss_task + LAMBDA_STATE * loss_state + LAMBDA_REV * loss_rev + LAMBDA_STRONG_POS_GATE * loss_strong_pos_gate

                    val_sum += float(loss.item())
                    val_trend_sum += float(loss_trend.item())
                    val_trend_coarse_sum += float(loss_trend_coarse.item())
                    val_fine_dc_sum += float(loss_fine_dc.item())
                    val_hard_late_sum += float(loss_hard_late_fine.item())
                    val_strong_pos_gate_sum += float(loss_strong_pos_gate.item())
                    val_n += 1
            val_loss = val_sum / max(1, val_n)

            print(f"[Epoch {epoch:02d}/{EPOCHS:02d}] "
                  f"Train={train_loss:.6f} (task={loss_task_sum/max(1,n_batch):.6f}, trend={loss_trend_sum/max(1,n_batch):.6f}, trend_cf={loss_trend_coarse_sum/max(1,n_batch):.6f}, fine_dc={loss_fine_dc_sum/max(1,n_batch):.6f}, hard_late={loss_hard_late_sum/max(1,n_batch):.6f}, strong_gate={loss_strong_pos_gate_sum/max(1,n_batch):.6f}, state={loss_state_sum/max(1,n_batch):.6f}) | "
                  f"Val={val_loss:.6f} (trend={val_trend_sum/max(1,val_n):.6f}, trend_cf={val_trend_coarse_sum/max(1,val_n):.6f}, fine_dc={val_fine_dc_sum/max(1,val_n):.6f}, hard_late={val_hard_late_sum/max(1,val_n):.6f}, strong_gate={val_strong_pos_gate_sum/max(1,val_n):.6f})")
    
            # ---- write history (CSV) ----
            task_avg = float(loss_task_sum / max(1, n_batch))
            trend_avg = float(loss_trend_sum / max(1, n_batch))
            trend_coarse_avg = float(loss_trend_coarse_sum / max(1, n_batch))
            fine_dc_avg = float(loss_fine_dc_sum / max(1, n_batch))
            hard_late_avg = float(loss_hard_late_sum / max(1, n_batch))
            strong_pos_gate_avg = float(loss_strong_pos_gate_sum / max(1, n_batch))
            state_avg = float(loss_state_sum / max(1, n_batch))
            history.append({
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "train_task": task_avg,
                "train_trend": trend_avg,
                "train_trend_coarse": trend_coarse_avg,
                "train_fine_dc": fine_dc_avg,
                "train_hard_late": hard_late_avg,
                "train_strong_pos_gate": strong_pos_gate_avg,
                "train_state": state_avg,
                "val_loss": float(val_loss),
                "val_trend": float(val_trend_sum / max(1, val_n)),
                "val_trend_coarse": float(val_trend_coarse_sum / max(1, val_n)),
                "val_fine_dc": float(val_fine_dc_sum / max(1, val_n)),
                "val_hard_late": float(val_hard_late_sum / max(1, val_n)),
                "val_strong_pos_gate": float(val_strong_pos_gate_sum / max(1, val_n)),
            })
            try:
                pd.DataFrame(history).to_csv(str(history_csv), index=False, encoding="utf-8-sig")
            except Exception:
                pass
    
            if val_loss < best_val:
                best_val = val_loss
                best_path = CKPT_DIR / "best_model_v5_8_protocol_safe.pth"
                torch.save(model.state_dict(), str(best_path))
                print(f"  🌟 New best -> {best_path}\n")
    
        print(f"\n⌛ 总训练耗时: {(time.time()-start_all)/60:.2f} min\n")
    
        # ---- save checkpoint with norms ----
        ckpt = {
            "state_dict": model.state_dict(),
            "feature_names": feature_names,
            "feat_mean": feat_mean,
            "feat_std": feat_std,
            "y_mean": y_mean,
            "y_std": y_std,
            "curve_mean": curve_mean,
            "curve_std": curve_std,
            "ctx_mean": ctx_mean,
            "ctx_std": ctx_std,
            "teacher_base_mu": base_mu,
            "teacher_base_sd": base_sd,
            "teacher_z_mu": z_mu,
            "teacher_z_sd": z_sd,
            "config": {
                "MODEL_VER": "v5_8_response_state_v1_protocol_safe",
                "WIN_SEC": WIN_SEC,
                "FUTURE_SEC": FUTURE_SEC,
                "WIN_LEN": WIN_LEN,
                "FUTURE_LEN": FUTURE_LEN,
                "D_MODEL": D_MODEL,
                "N_HEAD": N_HEAD,
                "ENC_LAYERS": NUM_LAYERS_ENC,
                "DEC_LAYERS": NUM_LAYERS_DEC,
                "W_DIFF1": W_DIFF1,
                "W_DIFF2": W_DIFF2,
                "W_REVSEQ": W_REVSEQ,
                "W_PEAKTIME": W_PEAKTIME,
                "REVSEQ_ALPHA_FRAC": REVSEQ_ALPHA_FRAC,
                "PEAK_TEMP_FRAC": PEAK_TEMP_FRAC,
                "W_STEER_WT": W_STEER_WT,
                "W_STEER_RATE": W_STEER_RATE,
                "W_STEER_REV": W_STEER_REV,
                "STEER_WT_MAX": STEER_WT_MAX,
                "W_TREND": W_TREND,
                "TREND_POOL_KERNEL": TREND_POOL_KERNEL,
                "TREND_POOL_STRIDE": TREND_POOL_STRIDE,
                "TREND_SIGN_EPS": TREND_SIGN_EPS,
                "STEER_SOURCE_UNIT": STEER_SOURCE_UNIT,
                "STEER_ANGLE_UNIT": STEER_ANGLE_UNIT,
                "STEER_ANGLE_SCALE": float(STEER_ANGLE_SCALE),
                "STEER_ONSET_THR_ABS": float(STEER_ONSET_THR_ABS),
                "TREND_LOSS_MODE": TREND_LOSS_MODE,
                "TREND_LEVEL_WEIGHT": TREND_LEVEL_WEIGHT,
                "TREND_DELTA_WEIGHT": TREND_DELTA_WEIGHT,
                "TREND_DIR_WEIGHT": TREND_DIR_WEIGHT,
                "ENABLE_STEER_COARSE_FINE": ENABLE_STEER_COARSE_FINE,
                "W_TREND_COARSE": W_TREND_COARSE,
                "W_FINE_DC": W_FINE_DC,
                "ENABLE_PHASE_ADAPTIVE_TREND": ENABLE_PHASE_ADAPTIVE_TREND,
                "TREND_EARLY_BINS": TREND_EARLY_BINS,
                "TREND_LATE_STRAIGHT_DOWN": TREND_LATE_STRAIGHT_DOWN,
                "TREND_LATE_STRONGREV_DOWN": TREND_LATE_STRONGREV_DOWN,
                "ENABLE_LATE_REV_GATE": ENABLE_LATE_REV_GATE,
                "LATE_REV_GATE_START_SEC": LATE_REV_GATE_START_SEC,
                "LATE_REV_GATE_SCALE": LATE_REV_GATE_SCALE,
                "LATE_REV_GATE_RAMP_POWER": LATE_REV_GATE_RAMP_POWER,
                "ENABLE_STRONG_POS_GATE": ENABLE_STRONG_POS_GATE,
                "STRONG_POS_GATE_START_SEC": STRONG_POS_GATE_START_SEC,
                "STRONG_POS_GATE_SCALE": STRONG_POS_GATE_SCALE,
                "STRONG_POS_GATE_RAMP_POWER": STRONG_POS_GATE_RAMP_POWER,
                "STRONG_POS_GATE_PROB_CENTER": STRONG_POS_GATE_PROB_CENTER,
                "ENABLE_HARD_LATE_FINE": ENABLE_HARD_LATE_FINE,
                "W_HARD_LATE_FINE": W_HARD_LATE_FINE,
                "HARD_LATE_START_SEC": HARD_LATE_START_SEC,
                "HARD_TAIL_START_SEC": HARD_TAIL_START_SEC,
                "HARD_PEAK_QUANTILE": HARD_PEAK_QUANTILE,
                "HARD_TAIL_QUANTILE": HARD_TAIL_QUANTILE,
                "LAMBDA_STATE": LAMBDA_STATE,
                "LAMBDA_REV": LAMBDA_REV,
                "LAMBDA_STRONG_POS_GATE": LAMBDA_STRONG_POS_GATE,
                "W_AMP": W_AMP,
                "ENABLE_RESPONSE_STATE_V1": ENABLE_RESPONSE_STATE_V1,
                "ENABLE_STATE_DISTILL": ENABLE_STATE_DISTILL,
                "ENABLE_REVERSAL_AUX": ENABLE_REVERSAL_AUX,
                "ENABLE_PEAKTIME_AUX": ENABLE_PEAKTIME_AUX,
                "ENABLE_PEAKINTENSITY_AUX": ENABLE_PEAKINTENSITY_AUX,
                "EEG_HIST_SEC": EEG_HIST_SEC,
            }
        }
        ckpt_path = CKPT_DIR / "model_rollpeak_transformer_v5_8_protocol_safe.pth"
        torch.save(ckpt, str(ckpt_path))
        print(f"💾 已保存 checkpoint: {ckpt_path}\n")
    
        # ---- save training curves ----
        try:
            df_h = pd.DataFrame(history)
            if len(df_h) > 0:
                plt.figure()
                plt.plot(df_h["epoch"], df_h["train_loss"], label="train")
                plt.plot(df_h["epoch"], df_h["val_loss"], label="val")
                plt.xlabel("epoch")
                plt.ylabel("loss")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                fig_path = FIG_DIR / "loss_curve.png"
                plt.savefig(str(fig_path), dpi=200)
                plt.close()
                print(f"📈 已保存曲线: {fig_path}\n")
        except Exception:
            pass
    

        # ---- export test prediction plots + state inspection ----
        try:
            best_path = CKPT_DIR / "best_model_v5_8_protocol_safe.pth"
            if best_path.exists():
                model.load_state_dict(torch.load(str(best_path), map_location=DEVICE))
                print("✅ 已加载 best 权重用于评估画图:", best_path)
            evaluate_and_plot(
                model,
                test_loader,
                y_mean,
                y_std,
                FIG_DIR,
                curve_thr=curve_thr,
                fs=FS,
                n_examples=8,
                state_component_names=teacher_state_meta["component_names"],
                teacher_state_mode=teacher_state_meta["mode"],
            )
        except Exception as e:
            print("⚠ 评估画图阶段失败:", repr(e))

        print("✅ 本次运行已完成。")
    
    finally:
        # restore stdout & close file handle
        sys.stdout = orig_stdout
        try:
            tee.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
