# -*- coding: utf-8 -*-
"""
v5.2: v5.1 + Driver State Conditioning + Multi-Scale Loss (Teacher: Physio+EEG, Student: Vehicle)
==============================================================================
- Baseline: Past2FutureMultiTaskRoadPreview (Non-AR Transformer Enc-Dec)
- Add: state_head on encoder memory -> z_veh (B,2)
- Training: L = L_task + lambda_state * MSE(z_veh, z_phys)
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


def make_run_dir(prefix="TRAIN_V5_2_STATECOND"):
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
EPOCHS = 150
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
W_DIFF1 = 0.30   # first-derivative loss weight
W_DIFF2 = 0.10   # second-derivative loss weight

# Distillation
LAMBDA_STATE = 0.10   # 可从 0.03~0.30 调
EEG_HIST_SEC = 2      # 你现在提取 EEG 事件特征用的 hist2s 文件名后缀
EPS = 1e-6

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

    hr  = float(np.nanmean(seg[col_hr].to_numpy(dtype=np.float64)))
    ton = float(np.nanmean(seg[col_t].to_numpy(dtype=np.float64)))
    pha = float(np.nanmean(seg[col_p].to_numpy(dtype=np.float64)))
    emg = float(np.nanmean(seg[col_emg].to_numpy(dtype=np.float64)))

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
# Build samples (vehicle + teacher base feats)
# =========================
def build_samples_for_vehicle(vehicle_file, style_map):
    df_v, df_e = load_vehicle_and_events(vehicle_file)
    if df_v is None:
        return [], [], [], [], [], None

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

    df_e = df_e[df_e["event_level"].isin(STRONG_LABELS)]
    if len(df_e) == 0:
        return [], [], [], [], [], None

    for ev_idx, ev in df_e.iterrows():
        t0 = float(ev["start_s"]); t1 = float(ev["end_s"])
        i0 = int(t0 * FS); i1 = int(t1 * FS)
        if i0 < 0 or i1 > N or (i1 - i0) < 10:
            continue

        roll_seg = X_all[i0:i1, roll_idx]
        if len(roll_seg) == 0:
            continue
        peak_rel = int(np.argmax(roll_seg))
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

        y_steer = X_all[peak_idx + 1: peak_idx + 1 + FUTURE_LEN, steer_idx]
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

    return X_list, y_list, curve_list, ctx_list, base_feat_list, feature_cols


def build_all_samples(style_map):
    pattern = os.path.join(ROOT, "*", "vehicle", "*_vehicle_aligned_cleaned.csv")
    vehicle_files = sorted(glob(pattern))

    X_pool, y_pool, curve_pool, ctx_pool, base_pool = [], [], [], [], []
    feature_names = None

    print("🔍 遍历车辆文件构造事件样本 + teacher base feats ...")
    total = 0
    for vf in vehicle_files:
        X_list, y_list, curve_list, ctx_list, base_list, feat_cols = build_samples_for_vehicle(vf, style_map)
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
        total += len(X_list)

    print(f"✅ 共收集到 {total} 个事件样本\n")
    return X_pool, y_pool, curve_pool, ctx_pool, base_pool, feature_names


# =========================
# Dataset
# =========================
class MultiTaskFutureWithCurveDataset(Dataset):
    def __init__(self, X_list, y_list, curve_list, ctx_list, z_phys_list,
                 y_mean, y_std, curve_mean, curve_std, ctx_mean, ctx_std):
        self.X = X_list
        self.y = y_list
        self.curve = curve_list
        self.ctx = ctx_list
        self.z_phys = z_phys_list  # (N,2) or NaN (masked)

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

        # Decoder input
        pos_tgt = self.dec_pos_emb[:, :T_out, :].expand(B, T_out, -1)
        ctx2 = torch.cat([ctx, z_veh], dim=1)  # (B, context_dim+2)
        ctx_emb = self.ctx_proj(ctx2).unsqueeze(1).expand(B, T_out, -1)

        curve_feat = curve_norm.unsqueeze(-1)   # (B,T_out,1)
        curve_emb = self.curve_proj(curve_feat) # (B,T_out,d_model)

        tgt = pos_tgt + ctx_emb + curve_emb
        out = self.decoder(tgt, memory)
        y_hat_norm = self.out_proj(out)

        return y_hat_norm, z_veh




# =========================
# Multi-scale loss helpers
# =========================

def _diff1(x: torch.Tensor) -> torch.Tensor:
    # x: (B,T,C)
    return x[:, 1:, :] - x[:, :-1, :]


def _diff2(x: torch.Tensor) -> torch.Tensor:
    return _diff1(_diff1(x))


def _denorm_y(y_norm_np: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray) -> np.ndarray:
    # y_norm_np: (N,T,3)
    return y_norm_np * y_std.reshape(1, 1, 3) + y_mean.reshape(1, 1, 3)


def evaluate_and_plot(model: nn.Module, test_loader: DataLoader,
                      y_mean: np.ndarray, y_std: np.ndarray,
                      fig_dir: Path, fs: int = 200, n_examples: int = 8):
    """Export:
      - figures/pred_vs_gt_example_*.png
      - figures/test_metrics.json
      - figures/test_state_dump.csv (A/C from veh & teacher + mask)
      - figures/state_vs_peak_*.png (quick relationship views)
    """
    model.eval()

    preds, trues = [], []
    zveh_all, zphys_all, zmask_all = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            src = batch["src"].to(DEVICE, non_blocking=True)
            y_true_norm = batch["y_norm"].to(DEVICE, non_blocking=True)
            curve_norm = batch["curve_norm"].to(DEVICE, non_blocking=True)
            ctx = batch["ctx"].to(DEVICE, non_blocking=True)
            z_phys = batch["z_phys"].to(DEVICE, non_blocking=True)
            z_mask = batch["z_mask"].to(DEVICE, non_blocking=True)  # (B,1)

            y_hat_norm, z_veh = model(src, ctx, curve_norm)

            preds.append(y_hat_norm.cpu().numpy())
            trues.append(y_true_norm.cpu().numpy())
            zveh_all.append(z_veh.cpu().numpy())
            zphys_all.append(z_phys.cpu().numpy())
            zmask_all.append(z_mask.cpu().numpy())

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

    # ---- state dump (event-level) ----
    df_state = pd.DataFrame({
        "A_veh": zveh_all[:, 0],
        "C_veh": zveh_all[:, 1],
        "A_teacher": zphys_all[:, 0],
        "C_teacher": zphys_all[:, 1],
        "teacher_mask": zmask_all,
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
    RUN_DIR = make_run_dir(prefix="TRAIN_V5_2_STATECOND")
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
        X_pool, y_pool, curve_pool, ctx_pool, base_pool, feature_names = build_all_samples(style_map)
        total = len(X_pool)
        if total == 0:
            print("❌ 没有有效事件样本")
            return

        # ---- shuffle & split ----
        idx = np.arange(total)
        np.random.shuffle(idx)

        X_pool     = [X_pool[i] for i in idx]
        y_pool     = [y_pool[i] for i in idx]
        curve_pool = [curve_pool[i] for i in idx]
        ctx_pool   = [ctx_pool[i] for i in idx]
        base_pool  = [base_pool[i] for i in idx]

        n_train = int(total * 0.8)

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
            X_pool[:n_train], y_pool[:n_train], curve_pool[:n_train], ctx_pool[:n_train], z_phys[:n_train],
            y_mean, y_std, curve_mean, curve_std, ctx_mean, ctx_std
        )
        test_dataset = MultiTaskFutureWithCurveDataset(
            X_pool[n_train:], y_pool[n_train:], curve_pool[n_train:], ctx_pool[n_train:], z_phys[n_train:],
            y_mean, y_std, curve_mean, curve_std, ctx_mean, ctx_std
        )
    
        def collate_fn(batch):
            src = torch.stack([torch.from_numpy(b["src"]).float() for b in batch], dim=0)
            y_norm = torch.stack([torch.from_numpy(b["y_norm"]).float() for b in batch], dim=0)
            curve_norm = torch.stack([torch.from_numpy(b["curve_norm"]).float() for b in batch], dim=0)
            ctx = torch.stack([torch.from_numpy(b["ctx"]).float() for b in batch], dim=0)
            z_phys = torch.stack([torch.from_numpy(b["z_phys"]).float() for b in batch], dim=0)
            z_mask = torch.stack([torch.from_numpy(b["z_mask"]).float() for b in batch], dim=0)  # (B,1)
            return {"src": src, "y_norm": y_norm, "curve_norm": curve_norm, "ctx": ctx, "z_phys": z_phys, "z_mask": z_mask}
    
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  collate_fn=collate_fn, pin_memory=torch.cuda.is_available())
        test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                  collate_fn=collate_fn, pin_memory=torch.cuda.is_available())
    
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
        print(f"Distill: lambda_state={LAMBDA_STATE}\n")
    
        # ---- persist run config (for reproducibility) ----
        run_config = {
            "MODEL_VER": "v5_2_state_cond_multiscale",
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
            "LAMBDA_STATE": LAMBDA_STATE,
            "EEG_HIST_SEC": EEG_HIST_SEC,
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
            loss_sum, loss_task_sum, loss_state_sum, n_batch = 0.0, 0.0, 0.0, 0
    
            for batch in train_loader:
                src = batch["src"].to(DEVICE, non_blocking=True)
                y_true = batch["y_norm"].to(DEVICE, non_blocking=True)
                curve_norm = batch["curve_norm"].to(DEVICE, non_blocking=True)
                ctx = batch["ctx"].to(DEVICE, non_blocking=True)
                z_phys_b = batch["z_phys"].to(DEVICE, non_blocking=True)
                z_mask = batch["z_mask"].to(DEVICE, non_blocking=True)  # (B,1)
    
                optim.zero_grad()
                y_hat, z_veh = model(src, ctx, curve_norm)
    
                loss_task = F.mse_loss(y_hat, y_true)
    
                # multi-scale (derivative) losses to encourage high-frequency details
    
                loss_d1 = F.mse_loss(_diff1(y_hat), _diff1(y_true))
    
                loss_d2 = F.mse_loss(_diff2(y_hat), _diff2(y_true))
    
                loss_task = loss_task + W_DIFF1 * loss_d1 + W_DIFF2 * loss_d2
                # state loss supports missing physio: if z_mask=0, ignore
                mse_state = ((z_veh - z_phys_b) ** 2).mean(dim=1, keepdim=True)  # (B,1)
                loss_state = (mse_state * z_mask).sum() / (z_mask.sum() + EPS)
    
                loss = loss_task + LAMBDA_STATE * loss_state
                loss.backward()
                optim.step()
    
                loss_sum += float(loss.item())
                loss_task_sum += float(loss_task.item())
                loss_state_sum += float(loss_state.item())
                n_batch += 1
    
            train_loss = loss_sum / max(1, n_batch)
    
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
    
                    y_hat, z_veh = model(src, ctx, curve_norm)
                    loss_task = F.mse_loss(y_hat, y_true)
                    # multi-scale (derivative) losses to encourage high-frequency details
                    loss_d1 = F.mse_loss(_diff1(y_hat), _diff1(y_true))
                    loss_d2 = F.mse_loss(_diff2(y_hat), _diff2(y_true))
                    loss_task = loss_task + W_DIFF1 * loss_d1 + W_DIFF2 * loss_d2
                    mse_state = ((z_veh - z_phys_b) ** 2).mean(dim=1, keepdim=True)
                    loss_state = (mse_state * z_mask).sum() / (z_mask.sum() + EPS)
                    loss = loss_task + LAMBDA_STATE * loss_state
    
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
                best_path = CKPT_DIR / "best_model_v5_2_state_cond_multiscale.pth"
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
                "MODEL_VER": "v5_2_state_cond_multiscale",
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
                "LAMBDA_STATE": LAMBDA_STATE,
                "EEG_HIST_SEC": EEG_HIST_SEC,
            }
        }
        ckpt_path = CKPT_DIR / "model_rollpeak_transformer_v5_2_state_cond_multiscale.pth"
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
            best_path = CKPT_DIR / "best_model_v5_2_state_cond_multiscale.pth"
            if best_path.exists():
                model.load_state_dict(torch.load(str(best_path), map_location=DEVICE))
                print("✅ 已加载 best 权重用于评估画图:", best_path)
            evaluate_and_plot(model, test_loader, y_mean, y_std, FIG_DIR, fs=FS, n_examples=8)
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
