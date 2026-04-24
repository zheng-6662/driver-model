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
ROOT = r"F:\数据集处理\data_process\datasetprocess\多模态数据\被试数据集合"
STYLE_CSV = r"F:\数据集处理\data_process\datasetprocess\多模态数据\driver_style_cluster_result.xlsx"

FS = 200
WIN_SEC = 3.0
FUTURE_SEC = 2.0
WIN_LEN = int(WIN_SEC * FS)         # 600
FUTURE_LEN = int(FUTURE_SEC * FS)   # 400

BATCH_SIZE = 64
EPOCHS = 40
LR = 1e-3
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

# Reversal-aware losses (focus on multi-reversal steering)
W_REVSEQ = 0.0        # time-wise reversal probability loss（精简版：关闭，先建立稳定基线）
W_PEAKTIME = 0.0      # steer-rate peak timing alignment loss
REVSEQ_ALPHA_FRAC = 0.25  # alpha = frac * steer_std (soft sign for reversal)
PEAK_TEMP_FRAC = 0.35     # temp = frac * mean|steer_rate| (soft-argmax)

# Steer local-detail weighting (emphasize reversals + high steer-rate)
W_STEER_WT = 0.50       # weighted steer MSE added to task loss
W_STEER_RATE = 1.00     # weight scale from |steer_rate|
W_STEER_REV = 0.0      # weight scale from reversal prob（精简版：关闭，避免局部反打片段过度加权）
STEER_WT_MAX = 4.0      # cap for stability

# Distillation
LAMBDA_STATE = 0.0   # 可从 0.03~0.30 调
W_TASK_STEER = 1.50   # steer 主任务权重
W_TASK_YAW   = 1.00   # yawrate 主任务权重
W_TASK_AY    = 0.70   # ay 主任务权重
W_AMP        = 0.30   # 幅值损失权重
REV_SAMPLE_WEIGHT = 1.80   # 强反打样本整体 loss 加权（建议先用 1.5~2.0）
REV_ZERO_EPS      = 1e-4    # 过零检测小阈值，避免数值噪声误判
LAMBDA_REV  = 0.0   # 反转辅助任务权重（精简版：关闭 rev_head 分类辅助任务）
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
ROAD_OK_RATIO_THR = 0.7  # use road_type_fixed when ref_nn_ok ratio >= this

# Event center alignment: keep event detection unchanged, but re-define final slicing center inside candidate event window.
EVENT_CENTER_MODE = "local_steer_rate_peak"  # ["legacy_center", "local_steer_rate_peak", "local_yawrate_peak", "local_ay_peak"]
EVENT_CENTER_SMOOTH_MS = 40.0        # local smoothing before peak picking
EVENT_CENTER_PROM_FRAC = 0.05         # min prominence fraction of local abs peak wrt window max
EVENT_CENTER_MIN_VALID_FRAC = 0.02    # if max(abs(signal)) < this*robust_scale => fallback
EVENT_CENTER_DEBUG_PLOTS = 12         # number of random debug center-alignment plots to save

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

def compute_teacher_state(base_feat_z):
    """
    base_feat_z: (B, 12)  已经按 train-set 统计做过 z-score 的基础特征
    dims:
      0..3  : HR, tonic, phasic, emg
      4..11 : eeg [alpha_asym, occ_ta_beta, frontal_ta_beta, temporal_ta_beta,
                   occ_alpha_abs, temporal_gamma_rel, occ_gamma_rel, frontal_gamma_rel]
    Output:
      z_phys_raw (B,2): [A,C] (未再标准化，可后面再对 A/C 做标准化)
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

    # ---- Arousal A ----
    A = (
        0.70 * hr +
        0.40 * tonic +
        0.80 * phasic +
        0.30 * gamma_mean +
        (-0.30) * occ_aabs +
        0.10 * alpha_asym
    )

    # ---- Cognitive Load C ----
    C = (
        0.70 * emg +
        0.50 * ta_mean
    )

    z = np.stack([A, C], axis=1).astype(np.float32)  # (B,2)
    return z




# =========================
# Event center alignment helpers
# =========================
def smooth_signal_1d(x, kernel_size):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x.copy()
    k = max(1, int(kernel_size))
    if k <= 1:
        return x.copy()
    if k % 2 == 0:
        k += 1
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    ker = np.ones(k, dtype=np.float64) / float(k)
    return np.convolve(xp, ker, mode="valid")


def _fallback_center(event_window, old_center_idx, reason):
    i0 = int(event_window["start_idx"])
    i1 = int(event_window["end_idx"])
    n = max(0, i1 - i0)
    fallback_used = True
    fallback_reason = reason

    if np.isfinite(old_center_idx) and i0 <= int(old_center_idx) < i1:
        return int(old_center_idx), fallback_used, fallback_reason + "|use_old_center"
    if n > 0:
        abs_mid = i0 + n // 2
        return int(abs_mid), fallback_used, fallback_reason + "|use_window_mid"
    return int(max(i0, 0)), fallback_used, fallback_reason + "|use_window_start"


def pick_local_abs_peak(signal, fs, smooth_ms=120.0, prom_frac=0.20, min_valid_frac=0.20):
    """Pick a robust main peak from a 1D signal INSIDE an event window only.
    Returns relative peak index and debug dict.
    """
    x = np.asarray(signal, dtype=np.float64)
    info = {
        "fallback": False,
        "fallback_reason": "",
        "n_candidates": 0,
        "smooth_kernel": 1,
        "used_max_abs": False,
    }
    if x.size == 0:
        info["fallback"] = True
        info["fallback_reason"] = "empty_signal"
        return None, info

    abs_x = np.abs(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0))
    k = max(1, int(round(float(smooth_ms) * fs / 1000.0)))
    abs_s = smooth_signal_1d(abs_x, k)
    info["smooth_kernel"] = int(k if k % 2 == 1 else k + 1)

    max_abs = float(np.max(abs_s)) if abs_s.size else 0.0
    q90 = float(np.quantile(abs_s, 0.90)) if abs_s.size else 0.0
    robust_scale = max(q90, 1e-8)
    if (not np.isfinite(max_abs)) or max_abs < min_valid_frac * robust_scale:
        info["fallback"] = True
        info["fallback_reason"] = "signal_too_small"
        return None, info

    prom_thr = float(prom_frac) * max_abs
    cand = []
    for i in range(1, len(abs_s) - 1):
        if abs_s[i] >= abs_s[i - 1] and abs_s[i] >= abs_s[i + 1]:
            local_base = max(min(abs_s[i - 1], abs_s[i + 1]), 0.0)
            prom = abs_s[i] - local_base
            if prom >= prom_thr:
                cand.append((i, float(abs_s[i]), float(prom)))
    info["n_candidates"] = int(len(cand))

    if cand:
        # choose the strongest local peak; tie-break by earlier one
        cand = sorted(cand, key=lambda z: (-z[1], z[0]))
        return int(cand[0][0]), info

    if abs_s.size:
        info["fallback"] = True
        info["fallback_reason"] = "no_prominent_local_peak"
        info["used_max_abs"] = True
        return int(np.argmax(abs_s)), info

    info["fallback"] = True
    info["fallback_reason"] = "invalid_smoothed_signal"
    return None, info


def compute_legacy_center(signal_dict, event_window, event_type):
    i0 = int(event_window["start_idx"])
    i1 = int(event_window["end_idx"])
    roll = signal_dict.get("roll")
    steer_rate = signal_dict.get("steer_rate")
    out = {"fallback": False, "fallback_reason": "", "source": "legacy"}

    if event_type == "curve":
        seg = None if roll is None else np.asarray(roll[i0:i1], dtype=np.float64)
        if seg is None or seg.size == 0:
            center, fb, rs = _fallback_center(event_window, np.nan, "legacy_curve_empty_roll")
            out["fallback"] = fb; out["fallback_reason"] = rs
            return center, out
        return int(i0 + np.argmax(np.abs(seg))), out

    seg = None if steer_rate is None else np.asarray(steer_rate[i0:i1], dtype=np.float64)
    if seg is not None and seg.size > 0:
        abs_sr = np.abs(np.nan_to_num(seg, nan=0.0))
        max_abs = float(np.max(abs_sr)) if abs_sr.size else 0.0
        if np.isfinite(max_abs) and max_abs >= 1e-6:
            thr = STEER_RATE_PEAK_FRAC * max_abs
            cand = np.where(abs_sr >= thr)[0]
            peak_rel = int(cand[0]) if cand.size else int(np.argmax(abs_sr))
            return int(i0 + peak_rel), out

    seg_roll = None if roll is None else np.asarray(roll[i0:i1], dtype=np.float64)
    if seg_roll is not None and seg_roll.size > 0:
        out["fallback"] = True
        out["fallback_reason"] = "legacy_straight_bad_steer_rate|use_roll_peak"
        return int(i0 + np.argmax(np.abs(seg_roll))), out

    center, fb, rs = _fallback_center(event_window, np.nan, "legacy_straight_empty_signals")
    out["fallback"] = fb; out["fallback_reason"] = rs
    return center, out


def compute_event_center(signal_dict, event_window, event_type, mode, fs):
    """Compute final event center INSIDE candidate event window only.

    signal_dict keys can contain: steer_rate, yawrate, ay, roll.
    event_window: {start_idx, end_idx, old_center_idx, event_id/sample_id(optional)}
    event_type: 'curve' or 'straight'
    mode: legacy_center/local_steer_rate_peak/local_yawrate_peak/local_ay_peak
    returns: center_idx, debug_info
    """
    i0 = int(event_window["start_idx"])
    i1 = int(event_window["end_idx"])
    old_center_idx = int(event_window.get("old_center_idx", (i0 + i1) // 2))
    debug = {
        "mode": str(mode),
        "event_type": str(event_type),
        "fallback": False,
        "fallback_reason": "",
        "peak_signal_name": "",
        "candidate_window_start": i0,
        "candidate_window_end": i1,
        "old_center_idx": old_center_idx,
    }

    legacy_center, legacy_info = compute_legacy_center(signal_dict, event_window, event_type)
    debug["legacy_center_idx"] = int(legacy_center)
    debug["legacy_fallback"] = bool(legacy_info.get("fallback", False))
    debug["legacy_fallback_reason"] = str(legacy_info.get("fallback_reason", ""))

    if mode == "legacy_center":
        debug["peak_signal_name"] = "legacy"
        debug["fallback"] = bool(legacy_info.get("fallback", False))
        debug["fallback_reason"] = str(legacy_info.get("fallback_reason", ""))
        return int(legacy_center), debug

    key_map = {
        "local_steer_rate_peak": "steer_rate",
        "local_yawrate_peak": "yawrate",
        "local_ay_peak": "ay",
    }
    sig_key = key_map.get(mode, None)
    if sig_key is None:
        center, fb, rs = _fallback_center(event_window, legacy_center, f"unknown_mode:{mode}")
        debug["peak_signal_name"] = "unknown"
        debug["fallback"] = fb
        debug["fallback_reason"] = rs
        return int(center), debug

    debug["peak_signal_name"] = sig_key
    sig = signal_dict.get(sig_key, None)
    if sig is None:
        center, fb, rs = _fallback_center(event_window, legacy_center, f"missing_signal:{sig_key}")
        debug["fallback"] = fb
        debug["fallback_reason"] = rs
        return int(center), debug

    seg = np.asarray(sig[i0:i1], dtype=np.float64)
    peak_rel, peak_info = pick_local_abs_peak(seg, fs=fs, smooth_ms=EVENT_CENTER_SMOOTH_MS, prom_frac=EVENT_CENTER_PROM_FRAC, min_valid_frac=EVENT_CENTER_MIN_VALID_FRAC)
    debug["n_peak_candidates"] = int(peak_info.get("n_candidates", 0))
    debug["smooth_kernel"] = int(peak_info.get("smooth_kernel", 1))

    if peak_rel is None:
        center, fb, rs = _fallback_center(event_window, legacy_center, peak_info.get("fallback_reason", f"invalid_{sig_key}"))
        debug["fallback"] = fb
        debug["fallback_reason"] = rs
        return int(center), debug

    center_idx = int(i0 + peak_rel)
    if center_idx < i0 or center_idx >= i1:
        center, fb, rs = _fallback_center(event_window, legacy_center, f"peak_out_of_window:{sig_key}")
        debug["fallback"] = fb
        debug["fallback_reason"] = rs
        return int(center), debug

    debug["fallback"] = bool(peak_info.get("fallback", False))
    debug["fallback_reason"] = str(peak_info.get("fallback_reason", ""))
    return int(center_idx), debug


def plot_event_center_debug(debug_rows, out_dir, n_plot=12, seed=2025):
    if debug_rows is None or len(debug_rows) == 0:
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)
    n_plot = int(min(max(0, n_plot), len(debug_rows)))
    if n_plot <= 0:
        return
    sel = rng.choice(len(debug_rows), size=n_plot, replace=False)
    for rank, idx in enumerate(sel):
        row = debug_rows[int(idx)]
        traces = row.get("plot_traces", None)
        if traces is None:
            continue
        t = traces.get("t", None)
        if t is None:
            continue
        fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
        axes[0].plot(t, traces.get("steer", np.zeros_like(t)), label="steer")
        axes[0].plot(t, traces.get("steer_rate", np.zeros_like(t)), label="steer_rate", alpha=0.8)
        axes[0].legend(loc="upper right")
        axes[0].set_ylabel("steer / rate")

        axes[1].plot(t, traces.get("yawrate", np.zeros_like(t)))
        axes[1].set_ylabel("yawrate")

        axes[2].plot(t, traces.get("ay", np.zeros_like(t)))
        axes[2].set_ylabel("ay")

        axes[3].plot(t, traces.get("roll", np.zeros_like(t)))
        axes[3].set_ylabel("roll")
        axes[3].set_xlabel("time (s)")

        win_start_s = traces.get("window_start_s", 0.0)
        win_end_s = traces.get("window_end_s", 0.0)
        old_center_s = traces.get("old_center_s", 0.0)
        final_center_s = traces.get("final_center_s", 0.0)
        for ax in axes:
            ax.axvspan(win_start_s, win_end_s, color="orange", alpha=0.12)
            ax.axvline(old_center_s, linestyle="--", alpha=0.9, label="old_center")
            ax.axvline(final_center_s, linestyle="-", alpha=0.9, label="final_center")
        axes[0].legend(loc="upper left")
        title = f"{row.get('sample_id','sample')} | {row.get('event_type','?')} | mode={row.get('event_center_mode','?')} | shift={row.get('center_shift_samples',0)} samp"
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        fig.savefig(out_dir / f"event_center_debug_{rank:02d}_{row.get('event_id','ev')}.png", dpi=150)
        plt.close(fig)

# =========================
# Build samples (vehicle + teacher base feats)
# =========================
def build_samples_for_vehicle(vehicle_file, style_map, event_center_mode=EVENT_CENTER_MODE):
    df_v, df_e = load_vehicle_and_events(vehicle_file)
    if df_v is None:
        return [], [], [], [], [], None, []

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
        return [], [], [], [], [], None
    if col_ay is None or col_yawrate is None or col_curve is None:
        return [], [], [], [], [], None

    base_cols = [c for c in [
        col_roll, col_yawrate, col_ay, col_ax, col_v,
        col_z, col_lane, col_curve, col_yaw, col_steer
    ] if c is not None]

    df_feat = df_v[base_cols].copy()

    if col_v is not None:
        df_feat[col_v] = df_feat[col_v] / 3.6

    if col_ay is not None:
        df_feat["LTR_est"] = df_v[col_ay] * LTR_COEFF

    steer = df_v[col_steer].to_numpy(dtype=np.float32)
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

    X_list, y_list, curve_list, ctx_list, base_feat_list = [], [], [], [], []
    center_debug_rows = []

    df_e = df_e[df_e["event_level"].isin(STRONG_LABELS)]
    if len(df_e) == 0:
        return [], [], [], [], [], None, []

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

        event_type = "curve" if is_curve else "straight"
        signal_dict = {
            "roll": X_all[:, roll_idx] if roll_idx is not None else None,
            "steer_rate": X_all[:, steer_rate_idx] if steer_rate_idx is not None else None,
            "yawrate": X_all[:, yawrate_idx] if yawrate_idx is not None else None,
            "ay": X_all[:, ay_idx] if ay_idx is not None else None,
            "steer": X_all[:, steer_idx] if steer_idx is not None else None,
        }
        old_center_idx, legacy_meta = compute_legacy_center(
            signal_dict,
            {"start_idx": i0, "end_idx": i1, "old_center_idx": (i0 + i1) // 2},
            event_type
        )
        final_center_idx, center_meta = compute_event_center(
            signal_dict=signal_dict,
            event_window={
                "start_idx": i0,
                "end_idx": i1,
                "old_center_idx": old_center_idx,
                "event_id": int(ev_idx),
            },
            event_type=event_type,
            mode=event_center_mode,
            fs=FS,
        )

        if final_center_idx - WIN_LEN < 0 or final_center_idx + FUTURE_LEN >= N:
            continue

        # teacher base feats
        phys4 = extract_physio_window_means(df_p, final_center_idx)  # (4,)
        eeg8 = eeg_map.get(int(ev_idx), None)                # (8,)
        if phys4 is None:
            # 若该被试缺失生理，允许继续训练（但该样本无法做 state loss）
            # 用 NaN 占位，后面会 mask
            phys4 = np.full((4,), np.nan, dtype=np.float32)
        if eeg8 is None:
            eeg8 = np.full((8,), np.nan, dtype=np.float32)

        base12 = np.concatenate([phys4, eeg8], axis=0).astype(np.float32)  # (12,)

        x_win = X_all[final_center_idx - WIN_LEN: final_center_idx]

        steer_anchor = float(X_all[final_center_idx, steer_idx])
        y_steer = X_all[final_center_idx + 1: final_center_idx + 1 + FUTURE_LEN, steer_idx] - steer_anchor
        y_yaw   = X_all[final_center_idx + 1: final_center_idx + 1 + FUTURE_LEN, yawrate_idx]
        y_ay    = X_all[final_center_idx + 1: final_center_idx + 1 + FUTURE_LEN, ay_idx]
        y_seq = np.stack([y_steer, y_yaw, y_ay], axis=-1)

        if v_arr is not None and s_axis is not None and curve_arr is not None:
            v0 = float(v_arr[final_center_idx])
            s0 = float(s_axis[final_center_idx])
            t_grid = (np.arange(1, FUTURE_LEN + 1, dtype=np.float64) * dt)
            s_query = np.clip(s0 + v0 * t_grid, s_axis[0], s_axis[-1])
            curve_future = np.interp(s_query, s_axis, curve_arr).astype(np.float32)
        else:
            curve_future = X_all[final_center_idx + 1: final_center_idx + 1 + FUTURE_LEN, curve_idx].astype(np.float32)

        steer_anchor = X_all[final_center_idx, steer_idx]
        ctx = np.array([
            steer_anchor,
            X_all[final_center_idx, steer_rate_idx],
            X_all[final_center_idx, ay_idx],
            X_all[final_center_idx, yawrate_idx],
            float(style_id)
        ], dtype=np.float32)

        sample_id = f"{Path(vehicle_file).stem}__ev{int(ev_idx)}"
        center_debug_rows.append({
            "event_id": int(ev_idx),
            "sample_id": sample_id,
            "subject_id": str(subject_id),
            "vehicle_file": str(vehicle_file),
            "event_type": event_type,
            "candidate_window_start": int(i0),
            "candidate_window_end": int(i1),
            "old_center_idx": int(old_center_idx),
            "final_center_idx": int(final_center_idx),
            "center_shift_samples": int(final_center_idx - old_center_idx),
            "center_shift_ms": float((final_center_idx - old_center_idx) * 1000.0 / FS),
            "event_center_mode": str(event_center_mode),
            "fallback": bool(center_meta.get("fallback", False)),
            "fallback_reason": str(center_meta.get("fallback_reason", "")),
            "legacy_fallback": bool(center_meta.get("legacy_fallback", False)),
            "legacy_fallback_reason": str(center_meta.get("legacy_fallback_reason", "")),
            "peak_signal_name": str(center_meta.get("peak_signal_name", "")),
            "plot_traces": {
                "t": (np.arange(max(0, i0 - WIN_LEN), min(N, i1 + FUTURE_LEN)) - max(0, i0 - WIN_LEN)) / FS,
                "steer": X_all[max(0, i0 - WIN_LEN): min(N, i1 + FUTURE_LEN), steer_idx].astype(np.float32),
                "steer_rate": X_all[max(0, i0 - WIN_LEN): min(N, i1 + FUTURE_LEN), steer_rate_idx].astype(np.float32),
                "yawrate": X_all[max(0, i0 - WIN_LEN): min(N, i1 + FUTURE_LEN), yawrate_idx].astype(np.float32),
                "ay": X_all[max(0, i0 - WIN_LEN): min(N, i1 + FUTURE_LEN), ay_idx].astype(np.float32),
                "roll": X_all[max(0, i0 - WIN_LEN): min(N, i1 + FUTURE_LEN), roll_idx].astype(np.float32),
                "window_start_s": float((i0 - max(0, i0 - WIN_LEN)) / FS),
                "window_end_s": float((i1 - max(0, i0 - WIN_LEN)) / FS),
                "old_center_s": float((old_center_idx - max(0, i0 - WIN_LEN)) / FS),
                "final_center_s": float((final_center_idx - max(0, i0 - WIN_LEN)) / FS),
            },
        })

        X_list.append(x_win.astype(np.float32))
        y_list.append(y_seq.astype(np.float32))
        curve_list.append(curve_future.astype(np.float32))
        ctx_list.append(ctx)
        base_feat_list.append(base12)

    return X_list, y_list, curve_list, ctx_list, base_feat_list, feature_cols, center_debug_rows


def build_all_samples(style_map, event_center_mode=EVENT_CENTER_MODE):
    pattern = os.path.join(ROOT, "*", "vehicle", "*_vehicle_aligned_cleaned.csv")
    vehicle_files = sorted(glob(pattern))

    X_pool, y_pool, curve_pool, ctx_pool, base_pool = [], [], [], [], []
    center_debug_all = []
    feature_names = None

    print("🔍 遍历车辆文件构造事件样本 + teacher base feats ...")
    total = 0
    for vf in vehicle_files:
        X_list, y_list, curve_list, ctx_list, base_list, feat_cols, center_rows = build_samples_for_vehicle(
            vf, style_map, event_center_mode=event_center_mode
        )
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
        center_debug_all.extend(center_rows)
        total += len(X_list)

    print(f"✅ 共收集到 {total} 个事件样本\n")
    return X_pool, y_pool, curve_pool, ctx_pool, base_pool, feature_names, center_debug_all


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
      z_veh:      (B, 2)  from encoder memory pooling (train for distillation; inference optional)
    """
    def __init__(self, input_dim, context_dim, future_len, out_dim=3,
                 d_model=128, nhead=2,
                 num_layers_enc=2, num_layers_dec=2,
                 dim_feedforward=256, dropout=0.1,
                 max_len_enc=600, max_len_dec=400):
        super().__init__()
        self.d_model = d_model
        self.future_len = future_len
        self.out_dim = out_dim

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
        self.dropout = nn.Dropout(dropout)

        # ---- NEW: state head (encoder pooling) ----
        self.pool_score = nn.Linear(d_model, 1)
        self.state_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

        # ---- NEW(v5.4): reversal classifier head (encoder pooling) ----
        self.rev_head = nn.Sequential(
            nn.Linear(d_model, 64),
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
        z_veh = self.state_head(ctx_enc)            # (B,2)
        rev_logit = self.rev_head(ctx_enc).squeeze(-1)  # (B,)

        # Decoder input
        pos_tgt = self.dec_pos_emb[:, :T_out, :].expand(B, T_out, -1)
        ctx2 = torch.cat([ctx, z_veh], dim=1)  # (B, context_dim+2)
        ctx_emb = self.ctx_proj(ctx2).unsqueeze(1).expand(B, T_out, -1)

        curve_feat = curve_norm.unsqueeze(-1)   # (B,T_out,1)
        curve_emb = self.curve_proj(curve_feat) # (B,T_out,d_model)

        tgt = pos_tgt + ctx_emb + curve_emb
        out = self.decoder(tgt, memory)
        y_hat_norm = self.out_proj(out)

        return y_hat_norm, z_veh, rev_logit




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

def evaluate_and_plot(model: nn.Module, test_loader: DataLoader,
                      y_mean: np.ndarray, y_std: np.ndarray,
                      fig_dir: Path, curve_thr: float = None, fs: int = 200, n_examples: int = 8):
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

            y_hat_norm, z_veh, rev_logit = model(src, ctx, curve_norm)

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
            curve_score_all.append(batch.get("curve_score", torch.full((src.shape[0],), float('nan'))).cpu().numpy())
            is_curve_all.append(batch.get("is_curve", torch.full((src.shape[0],), -1, dtype=torch.long)).cpu().numpy())

    pred_norm = np.concatenate(preds, axis=0)
    true_norm = np.concatenate(trues, axis=0)
    zveh_all = np.concatenate(zveh_all, axis=0)          # (N,2)
    zphys_all = np.concatenate(zphys_all, axis=0)        # (N,2)
    zmask_all = np.concatenate(zmask_all, axis=0).reshape(-1)  # (N,)

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
    save_json(fig_dir / "test_metrics.json", metrics)
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

    rev_metrics = {
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
    }
    save_json(fig_dir / "test_metrics_by_reversal.json", rev_metrics)
    print("🔁 Reversal 指标:", rev_metrics)


    # ---- state dump (event-level) ----
    df_state = pd.DataFrame({
        "A_veh": zveh_all[:, 0],
        "C_veh": zveh_all[:, 1],
        "A_teacher": zphys_all[:, 0],
        "C_teacher": zphys_all[:, 1],
        "teacher_mask": zmask_all,
        "is_curve": np.concatenate(is_curve_all, axis=0).astype(np.int64) if len(is_curve_all)>0 else -1,
        "curve_score": np.concatenate(curve_score_all, axis=0).astype(np.float32) if len(curve_score_all)>0 else np.nan,
        "rev_gt": np.concatenate(rev_gt_all, axis=0).astype(np.int64) if len(rev_gt_all)>0 else -1,
        "rev_gt_weak": np.concatenate(rev_gt_weak_all, axis=0).astype(np.int64) if len(rev_gt_weak_all)>0 else -1,
        "rev_gt_strong": np.concatenate(rev_gt_strong_all, axis=0).astype(np.int64) if len(rev_gt_strong_all)>0 else -1,
        "rev_prob": np.concatenate(rev_prob_all, axis=0).astype(np.float32) if len(rev_prob_all)>0 else np.nan,
        "idx": np.concatenate(idx_all, axis=0).astype(np.int64) if len(idx_all)>0 else -1,
    })

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

    _scatter(df_state["A_veh"].values, peak_abs_steer, "A_veh (student)", "peak|steer| (GT)", "state_vs_peak_steer_A.png")
    _scatter(df_state["C_veh"].values, peak_abs_steer, "C_veh (student)", "peak|steer| (GT)", "state_vs_peak_steer_C.png")
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
        ax1.set_ylabel("steer")
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

        style_map = load_driver_style_map(STYLE_CSV)
        X_pool, y_pool, curve_pool, ctx_pool, base_pool, feature_names, center_debug_rows = build_all_samples(
            style_map, event_center_mode=EVENT_CENTER_MODE
        )
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

        try:
            debug_csv = RUN_DIR / "event_center_debug.csv"
            debug_df = pd.DataFrame([{k: v for k, v in r.items() if k != "plot_traces"} for r in center_debug_rows])
            debug_df.to_csv(debug_csv, index=False, encoding="utf-8-sig")
            print("🧭 已保存事件中心调试表:", debug_csv)
            plot_event_center_debug(center_debug_rows, FIG_DIR / "event_center_debug", n_plot=EVENT_CENTER_DEBUG_PLOTS, seed=SEED)
            print("🧭 已保存事件中心调试图目录:", FIG_DIR / "event_center_debug")
        except Exception as e:
            print("⚠ 保存事件中心调试信息失败:", e)

        # ---- shuffle & split ----
        idx = np.arange(total)
        np.random.shuffle(idx)

        X_pool     = [X_pool[i] for i in idx]
        y_pool     = [y_pool[i] for i in idx]
        curve_pool = [curve_pool[i] for i in idx]
        ctx_pool   = [ctx_pool[i] for i in idx]
        base_pool  = [base_pool[i] for i in idx]
        center_debug_rows = [center_debug_rows[i] for i in idx]

        n_train = int(total * 0.8)

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
            curve_thr = auto_curve_threshold(curve_scores[:n_train])
            is_curve = (curve_scores > curve_thr).astype(np.int64)

            ratio_curve = float(np.mean(is_curve))
            print(f"🛣 road_type: 使用历史 3s 平均|curvature| 分割直/弯")
            print(f"   曲率列: {curve_feat_name}")
            print(f"   curve_thr = {curve_thr:.3e}  (train auto)")
            print(f"   curve_ratio(all) = {ratio_curve*100:.1f}%  |  straight_ratio = {(1-ratio_curve)*100:.1f}%")

        # ---- standardize encoder src features ----
        all_X_concat = np.concatenate(X_pool[:n_train], axis=0)
        feat_mean = all_X_concat.mean(axis=0)
        feat_std = all_X_concat.std(axis=0)
        feat_std[feat_std < 1e-6] = 1e-6
        for i in range(len(X_pool)):
            X_pool[i] = (X_pool[i] - feat_mean) / feat_std
    
        # ---- standardize outputs ----
        all_y_concat = np.concatenate([y.reshape(-1, 3) for y in y_pool[:n_train]], axis=0)
        y_mean = all_y_concat.mean(axis=0)
        y_std  = all_y_concat.std(axis=0)
        y_std[y_std < 1e-6] = 1e-6

        y_mean_t = torch.tensor(y_mean, device=DEVICE, dtype=torch.float32)
        y_std_t = torch.tensor(y_std, device=DEVICE, dtype=torch.float32)
    
        # ---- curve std ----
        all_curve_concat = np.concatenate(curve_pool[:n_train], axis=0)
        curve_mean = all_curve_concat.mean()
        curve_std = all_curve_concat.std()
        if curve_std < 1e-6:
            curve_std = 1e-6
    
        # ---- ctx std ----
        ctx_array = np.stack(ctx_pool[:n_train], axis=0)
        ctx_mean = ctx_array.mean(axis=0)
        ctx_std  = ctx_array.std(axis=0)
        ctx_std[ctx_std < 1e-6] = 1e-6
    
        # ---- teacher base feat z-score (train stats only) ----
        base_train = np.stack(base_pool[:n_train], axis=0)  # (Ntr,12)
        base_mu = np.nanmean(base_train, axis=0)
        base_sd = np.nanstd(base_train, axis=0)
        base_sd[base_sd < 1e-6] = 1e-6
    
        def zscore_base(x12):
            x = x12.copy()
            # NaN -> mean（等价于 z=0），避免污染
            nan_mask = ~np.isfinite(x)
            x[nan_mask] = np.take(base_mu, np.where(nan_mask)[0])
            return (x - base_mu) / base_sd
    
        base_z_all = np.stack([zscore_base(x) for x in base_pool], axis=0)  # (N,12)
        z_phys_raw = compute_teacher_state(base_z_all)  # (N,2)
    
        # 进一步把 A/C 再标准化（train stats）
        z_tr = z_phys_raw[:n_train]
        z_mu = np.mean(z_tr, axis=0)
        z_sd = np.std(z_tr, axis=0)
        z_sd[z_sd < 1e-6] = 1e-6
        z_phys = (z_phys_raw - z_mu) / z_sd  # (N,2)
    
        # dataset
        train_dataset = MultiTaskFutureWithCurveDataset(
            X_pool[:n_train], y_pool[:n_train], curve_pool[:n_train], ctx_pool[:n_train], z_phys[:n_train], rev_gt[:n_train], rev_gt_weak[:n_train], rev_gt_strong[:n_train],
            y_mean, y_std, curve_mean, curve_std, ctx_mean, ctx_std,
            curve_scores[:n_train], is_curve[:n_train]
        )
        test_dataset = MultiTaskFutureWithCurveDataset(
            X_pool[n_train:], y_pool[n_train:], curve_pool[n_train:], ctx_pool[n_train:], z_phys[n_train:], rev_gt[n_train:], rev_gt_weak[n_train:], rev_gt_strong[n_train:],
            y_mean, y_std, curve_mean, curve_std, ctx_mean, ctx_std,
            curve_scores[n_train:], is_curve[n_train:]
        )
    
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
        test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                  collate_fn=collate_fn, pin_memory=torch.cuda.is_available())

        # ---- class balancing for reversal aux loss ----
        try:
            pos_cnt = float(np.sum(rev_gt[:n_train] > 0.5))
            neg_cnt = float(n_train - pos_cnt)
            pw = neg_cnt / max(1.0, pos_cnt)
            rev_pos_weight = torch.tensor(pw, device=DEVICE)
            print(f"🔁 rev_head pos_weight={pw:.3f}  (pos={pos_cnt:.0f}, neg={neg_cnt:.0f})")
        except Exception:
            rev_pos_weight = torch.tensor(1.0, device=DEVICE)
    
        # model
        model = Past2FutureMultiTaskRoadPreview(
            input_dim=len(feature_names),
            context_dim=7,
            future_len=FUTURE_LEN,
            out_dim=3,
            d_model=D_MODEL,
            nhead=N_HEAD,
            num_layers_enc=NUM_LAYERS_ENC,
            num_layers_dec=NUM_LAYERS_DEC,
            dim_feedforward=FFN_DIM,
            dropout=DROPOUT,
            max_len_enc=WIN_LEN,
            max_len_dec=FUTURE_LEN
        ).to(DEVICE)
    
        optim = torch.optim.Adam(model.parameters(), lr=LR)
    
        print(f"训练集样本数: {len(train_dataset)} | 测试集样本数: {len(test_dataset)}")
        print(f"历史窗口: {WIN_SEC:.1f}s({WIN_LEN}) 未来窗口: {FUTURE_SEC:.1f}s({FUTURE_LEN})")
        print(f"Distill: lambda_state={LAMBDA_STATE} | lambda_rev={LAMBDA_REV} | REV_EPS={REV_EPS}\n")
    
        # ---- persist run config (for reproducibility) ----
        run_config = {
            "MODEL_VER": "v5_4_state_cond_multiscale_roadtype_eval_reversal_aux",
            "LAMBDA_REV": float(LAMBDA_REV),
            "REV_EPS": float(REV_EPS),
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
            "LAMBDA_STATE": LAMBDA_STATE,
            "EEG_HIST_SEC": EEG_HIST_SEC,
            "EVENT_CENTER_MODE": EVENT_CENTER_MODE,
            "EVENT_CENTER_SMOOTH_MS": EVENT_CENTER_SMOOTH_MS,
            "EVENT_CENTER_PROM_FRAC": EVENT_CENTER_PROM_FRAC,
            "EVENT_CENTER_MIN_VALID_FRAC": EVENT_CENTER_MIN_VALID_FRAC,
            "SEED": SEED,
            "N_TRAIN": int(len(train_dataset)),
            "N_TEST": int(len(test_dataset)),
        }
        save_json(RUN_DIR / "run_config.json", run_config)
    
        # ---- training history ----
        history = []
        history_csv = RUN_DIR / "loss_history.csv"
    
        best_val = np.inf
        start_all = time.time()
    
        for epoch in range(1, EPOCHS + 1):
            model.train()
            loss_sum, loss_task_sum, loss_state_sum, loss_rev_sum, n_batch = 0.0, 0.0, 0.0, 0.0, 0
    
            for batch in train_loader:
                src = batch["src"].to(DEVICE, non_blocking=True)
                y_true = batch["y_norm"].to(DEVICE, non_blocking=True)
                curve_norm = batch["curve_norm"].to(DEVICE, non_blocking=True)
                ctx = batch["ctx"].to(DEVICE, non_blocking=True)
                z_phys_b = batch["z_phys"].to(DEVICE, non_blocking=True)
                z_mask = batch["z_mask"].to(DEVICE, non_blocking=True)  # (B,1)
                rev_gt_b = batch["rev_gt"].to(DEVICE, non_blocking=True).squeeze(1)  # (B,)
                rev_gt_weak_b = batch.get("rev_gt_weak", batch["rev_gt"]).to(DEVICE, non_blocking=True).squeeze(1)
                rev_gt_strong_b = batch.get("rev_gt_strong", batch["rev_gt"]).to(DEVICE, non_blocking=True).squeeze(1)
    
                optim.zero_grad()
                y_hat, z_veh, rev_logit = model(src, ctx, curve_norm)

                # 主任务：加权多任务 MSE（steer 更高优先级）
                loss_steer = F.mse_loss(y_hat[:, :, 0], y_true[:, :, 0])
                loss_yaw   = F.mse_loss(y_hat[:, :, 1], y_true[:, :, 1])
                loss_ay    = F.mse_loss(y_hat[:, :, 2], y_true[:, :, 2])
                loss_task = (
                    W_TASK_STEER * loss_steer
                    + W_TASK_YAW * loss_yaw
                    + W_TASK_AY * loss_ay
                ) / (W_TASK_STEER + W_TASK_YAW + W_TASK_AY)
                loss_amp = compute_amplitude_loss(y_hat, y_true)

                # multi-scale (derivative) losses to encourage high-frequency details

                loss_d1 = F.mse_loss(_diff1(y_hat), _diff1(y_true))

                loss_d2 = F.mse_loss(_diff2(y_hat), _diff2(y_true))

                loss_task = loss_task + W_DIFF1 * loss_d1 + W_DIFF2 * loss_d2 + W_AMP * loss_amp

                # ---- reversal-aware losses on de-normalized steer ----
                y_hat_den = y_hat * y_std_t.view(1, 1, 3) + y_mean_t.view(1, 1, 3)
                y_true_den = y_true * y_std_t.view(1, 1, 3) + y_mean_t.view(1, 1, 3)
                steer_pred = y_hat_den[:, :, 0]
                steer_true = y_true_den[:, :, 0]

                alpha = REVSEQ_ALPHA_FRAC * y_std_t[0]
                p_rev_pred = _soft_reversal_prob(steer_pred, alpha)
                with torch.no_grad():
                    p_rev_true = _soft_reversal_prob(steer_true, alpha)
                loss_revseq = F.mse_loss(p_rev_pred, p_rev_true)

                steer_rate_pred = _diff1(steer_pred).abs()
                steer_rate_true = _diff1(steer_true).abs()
                temp = PEAK_TEMP_FRAC * (steer_rate_true.mean() + EPS)
                peak_pred = _soft_peak_time(steer_rate_pred, temp)
                with torch.no_grad():
                    peak_true = _soft_peak_time(steer_rate_true, temp)
                loss_peaktime = F.mse_loss(peak_pred, peak_true)

                # weighted steer MSE: emphasize reversals + high steer-rate regions
                with torch.no_grad():
                    rate_norm = steer_rate_true / (steer_rate_true.mean(dim=1, keepdim=True) + EPS)  # (B,T-1)
                    # 精简版训练：关闭 reversal 时序加权，只保留基于转角变化率的加权。
                    rev_seq = torch.zeros_like(rate_norm)  # (B,T-1)
                    w = 1.0 + W_STEER_RATE * rate_norm + W_STEER_REV * rev_seq
                    w = torch.clamp(w, max=STEER_WT_MAX)
                loss_steer_wt = (((steer_pred[:, 1:] - steer_true[:, 1:]) ** 2) * w).mean()

                loss_task = loss_task + W_REVSEQ * loss_revseq + W_PEAKTIME * loss_peaktime + W_STEER_WT * loss_steer_wt
                # state loss supports missing physio: if z_mask=0, ignore
                mse_state = ((z_veh - z_phys_b) ** 2).mean(dim=1, keepdim=True)  # (B,1)
                loss_state = (mse_state * z_mask).sum() / (z_mask.sum() + EPS)
    
                # reversal aux loss (binary): whether future steer changes sign
                # 精简版训练：关闭 reversal 分类辅助任务，避免与主回归目标竞争。
                loss_rev = torch.tensor(0.0, device=DEVICE)

                loss = loss_task + LAMBDA_STATE * loss_state + LAMBDA_REV * loss_rev
                loss.backward()
                optim.step()
    
                loss_sum += float(loss.item())
                loss_task_sum += float(loss_task.item())
                loss_state_sum += float(loss_state.item())
                loss_rev_sum += float(loss_rev.item())
                n_batch += 1
    
            train_loss = loss_sum / max(1, n_batch)
            train_loss_rev = loss_rev_sum / max(1, n_batch)
    
            # val
            model.eval()
            val_sum, val_n = 0.0, 0
            with torch.no_grad():
                for batch in test_loader:
                    src = batch["src"].to(DEVICE, non_blocking=True)
                    y_true = batch["y_norm"].to(DEVICE, non_blocking=True)
                    curve_norm = batch["curve_norm"].to(DEVICE, non_blocking=True)
                    ctx = batch["ctx"].to(DEVICE, non_blocking=True)
                    z_phys_b = batch["z_phys"].to(DEVICE, non_blocking=True)
                    z_mask = batch["z_mask"].to(DEVICE, non_blocking=True)
                    rev_gt_b = batch["rev_gt"].to(DEVICE, non_blocking=True).squeeze(1)
    
                    y_hat, z_veh, rev_logit = model(src, ctx, curve_norm)
                    # 主任务：加权多任务 MSE（steer 更高优先级）
                    loss_steer = F.mse_loss(y_hat[:, :, 0], y_true[:, :, 0])
                    loss_yaw   = F.mse_loss(y_hat[:, :, 1], y_true[:, :, 1])
                    loss_ay    = F.mse_loss(y_hat[:, :, 2], y_true[:, :, 2])
                    loss_task = (
                        W_TASK_STEER * loss_steer
                        + W_TASK_YAW * loss_yaw
                        + W_TASK_AY * loss_ay
                    ) / (W_TASK_STEER + W_TASK_YAW + W_TASK_AY)
                    loss_amp = compute_amplitude_loss(y_hat, y_true)
                    # multi-scale (derivative) losses to encourage high-frequency details
                    loss_d1 = F.mse_loss(_diff1(y_hat), _diff1(y_true))
                    loss_d2 = F.mse_loss(_diff2(y_hat), _diff2(y_true))
                    loss_task = loss_task + W_DIFF1 * loss_d1 + W_DIFF2 * loss_d2 + W_AMP * loss_amp

                    # ---- reversal-aware losses on de-normalized steer ----
                    y_hat_den = y_hat * y_std_t.view(1, 1, 3) + y_mean_t.view(1, 1, 3)
                    y_true_den = y_true * y_std_t.view(1, 1, 3) + y_mean_t.view(1, 1, 3)
                    steer_pred = y_hat_den[:, :, 0]
                    steer_true = y_true_den[:, :, 0]

                    alpha = REVSEQ_ALPHA_FRAC * y_std_t[0]
                    p_rev_pred = _soft_reversal_prob(steer_pred, alpha)
                    p_rev_true = _soft_reversal_prob(steer_true, alpha)
                    loss_revseq = F.mse_loss(p_rev_pred, p_rev_true)

                    steer_rate_pred = _diff1(steer_pred).abs()
                    steer_rate_true = _diff1(steer_true).abs()
                    temp = PEAK_TEMP_FRAC * (steer_rate_true.mean() + EPS)
                    peak_pred = _soft_peak_time(steer_rate_pred, temp)
                    peak_true = _soft_peak_time(steer_rate_true, temp)
                    loss_peaktime = F.mse_loss(peak_pred, peak_true)

                    # weighted steer MSE: emphasize reversals + high steer-rate regions
                    rate_norm = steer_rate_true / (steer_rate_true.mean(dim=1, keepdim=True) + EPS)  # (B,T-1)
                    rev_seq = p_rev_true  # (B,T-1)
                    w = 1.0 + W_STEER_RATE * rate_norm + W_STEER_REV * rev_seq
                    w = torch.clamp(w, max=STEER_WT_MAX)
                    loss_steer_wt = (((steer_pred[:, 1:] - steer_true[:, 1:]) ** 2) * w).mean()

                    loss_task = loss_task + W_REVSEQ * loss_revseq + W_PEAKTIME * loss_peaktime + W_STEER_WT * loss_steer_wt
                    mse_state = ((z_veh - z_phys_b) ** 2).mean(dim=1, keepdim=True)
                    loss_state = (mse_state * z_mask).sum() / (z_mask.sum() + EPS)
                    # reversal aux loss (binary): whether future steer changes sign
                    loss_rev = F.binary_cross_entropy_with_logits(rev_logit, rev_gt_b.float(), pos_weight=rev_pos_weight)
                    loss = loss_task + LAMBDA_STATE * loss_state + LAMBDA_REV * loss_rev
    
                    val_sum += float(loss.item())
                    val_n += 1
            val_loss = val_sum / max(1, val_n)
    
            print(f"[Epoch {epoch:02d}/{EPOCHS:02d}] "
                  f"Train={train_loss:.6f} (task={loss_task_sum/max(1,n_batch):.6f}, state={loss_state_sum/max(1,n_batch):.6f}) | "
                  f"Test={val_loss:.6f}")
    
            # ---- write history (CSV) ----
            task_avg = float(loss_task_sum / max(1, n_batch))
            state_avg = float(loss_state_sum / max(1, n_batch))
            history.append({
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "train_task": task_avg,
                "train_state": state_avg,
                "val_loss": float(val_loss),
            })
            try:
                pd.DataFrame(history).to_csv(str(history_csv), index=False, encoding="utf-8-sig")
            except Exception:
                pass
    
            if val_loss < best_val:
                best_val = val_loss
                best_path = CKPT_DIR / "best_model_v5_5_state_cond_multiscale_roadtype_eval.pth"
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
                "MODEL_VER": "v5_3_state_cond_multiscale_roadtype_eval",
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
                "LAMBDA_STATE": LAMBDA_STATE,
                "EEG_HIST_SEC": EEG_HIST_SEC,
            }
        }
        ckpt_path = CKPT_DIR / "model_rollpeak_transformer_v5_3_state_cond_multiscale_roadtype_eval.pth"
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
            best_path = CKPT_DIR / "best_model_v5_5_state_cond_multiscale_roadtype_eval.pth"
            if best_path.exists():
                model.load_state_dict(torch.load(str(best_path), map_location=DEVICE))
                print("✅ 已加载 best 权重用于评估画图:", best_path)
            evaluate_and_plot(model, test_loader, y_mean, y_std, FIG_DIR, curve_thr=curve_thr, fs=FS, n_examples=8)
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
