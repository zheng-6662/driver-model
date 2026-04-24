# -*- coding: utf-8 -*-
"""
future_steer_event_rollpeak_transformer_v5_0_longwin_multitask_roadpreview_style.py
====================================================================================

v5.0a: 长窗口 + 多任务 + Context + Road Curvature Preview(速度投影预瞄) + Driver Style Embedding
-----------------------------------------------------------------------------------
在 v4.6 基础上加入【驾驶风格】信息：

- 对每个被试 subject_id，有一个 driver_style_id (0,1,2 ...)
- 在 Decoder context 中加入该风格 id：
    ctx = [steer_anchor, steer_rate, ay, yawrate, style_id]  → 5 维
- 模型里 context_dim = 5，通过线性层投影到 d_model

这样模型可以同时利用：
- 历史 3s 车辆状态 (Encoder)
- anchor 时刻车辆状态 (ctx)
- 未来 2s 道路曲率 (road curvature preview)
- 驾驶员长期风格 (driver_style)

其它结构与 v4.6 相同。
"""

import os
import time
from glob import glob
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")  # 想弹窗看图就改成 "TkAgg"
import matplotlib.pyplot as plt


# =========================
# 配置
# =========================

ROOT = r"F:\数据集处理\data_process\datasetprocess\多模态数据\被试数据集合"

# 🔴 这里改成你【驾驶风格聚类结果】所在的 CSV 路径
# 要求：至少包含一列 subject（被试 ID，如 byx/cwh/...）
# 以及一列风格标签列（例如 cluster_main_k2 / style_main / style_3style）
STYLE_CSV = r"F:\数据集处理\data_process\datasetprocess\多模态数据\driver_style_cluster_result.xlsx"

FS = 200
WIN_SEC = 3.0          # 历史窗口: 3s
FUTURE_SEC = 2.0       # 预测窗口: 2s
WIN_LEN = int(WIN_SEC * FS)       # 600
FUTURE_LEN = int(FUTURE_SEC * FS) # 400

BATCH_SIZE = 64
EPOCHS = 40
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LTR_COEFF = 0.11243
STRONG_LABELS = ["medium_active", "strong_active", "extreme_active"]

# Transformer 参数
D_MODEL = 128
N_HEAD = 2
NUM_LAYERS_ENC = 2
NUM_LAYERS_DEC = 2
FFN_DIM = 256
DROPOUT = 0.1

SEED = 2025
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =========================
# 工具函数
# =========================

def find_col(cols, candidates):
    """在多个候选列名中找第一个存在的"""
    for c in candidates:
        if c in cols:
            return c
    return None

def make_strictly_increasing(x, eps=1e-6):
    """Ensure x is strictly increasing for robust interpolation on distance axis.
    If the vehicle is stopped (v≈0), cumulative distance may contain plateaus;
    this function adds a tiny epsilon to keep it strictly increasing.
    """
    x = np.asarray(x, dtype=np.float64).copy()
    for i in range(1, len(x)):
        if x[i] <= x[i - 1]:
            x[i] = x[i - 1] + eps
    return x



def load_vehicle_and_events(vehicle_file):
    """根据车辆文件推断事件文件路径, 并读取"""
    event_file = vehicle_file.replace("\\vehicle\\", "\\event\\") \
        .replace("_vehicle_aligned_cleaned.csv",
                 "_vehicle_aligned_cleaned_events_v312.csv")
    if not os.path.exists(event_file):
        print(f"⚠ 事件文件不存在: {event_file}")
        return None, None
    df_v = pd.read_csv(vehicle_file)
    df_e = pd.read_csv(event_file)
    return df_v, df_e


def load_driver_style_map(style_csv):
    """
    读取驾驶风格聚类结果，生成 {subject_id -> style_id(int)} 映射。

    要求 CSV 至少包含:
        - 一列 subject (被试 ID，与 ROOT 下文件夹名称一致：byx/cwh/...)
        - 一列风格列名，比如 'cluster_main_k2' 或 'style_main' 等

    若风格列是字符串，会自动 factorize 成 0,1,2,...
    """
    if not os.path.exists(style_csv):
        print(f"⚠ 未找到驾驶风格结果 CSV: {style_csv}")
        print("  → 将所有被试的 style_id 默认为 0")
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

    # 候选的风格列名（你可以按需要在这里添加）
    style_col = None
    for c in ["cluster_main_k2", "style_main", "style_3style",
              "cluster", "style_id", "cluster_id"]:
        if c in cols:
            style_col = c
            break
    if style_col is None:
        raise ValueError(
            f"在 {style_csv} 中找不到风格列，请确认是否包含 'cluster_main_k2' 或 'style_main' 等列。"
        )

    style_vals = df[style_col].values
    # 若不是纯数字，做 factorize
    if not np.issubdtype(style_vals.dtype, np.number):
        cats, idx = np.unique(style_vals, return_inverse=True)
        style_ids = idx
        print("🔧 驾驶风格列为字符串，已自动 factorize：")
        for i, cat in enumerate(cats):
            print(f"  style_id={i} ⇔ 原始标签='{cat}'")
    else:
        # 数值型，转为 int
        style_ids = style_vals.astype(int)

    subj_vals = df[subj_col].astype(str).values
    style_map = {s: int(k) for s, k in zip(subj_vals, style_ids)}

    print("\n✅ 已加载驾驶风格映射 subject → style_id:")
    for s, k in style_map.items():
        print(f"  {s:10s} → style_id={k}")
    print()
    return style_map


def get_subject_id_from_path(vehicle_file):
    """
    根据 ROOT/<subject>/vehicle/*.csv 的路径结构解析出 subject_id
    """
    norm = os.path.normpath(vehicle_file)
    parts = norm.split(os.sep)
    # .../<ROOT>/<subject>/vehicle/xxx.csv → subject 在倒数第3个
    if len(parts) >= 3:
        return parts[-3]
    else:
        return "unknown"


# =========================
# 构造事件样本 (长窗口 + 多任务 + 道路预览 + 驾驶风格)
# =========================

def build_samples_for_vehicle(vehicle_file, style_map):
    """
    针对单个车辆文件, 构造所有有效事件样本:

    返回:
      X_list:       [(WIN_LEN, D), ...]  历史特征, 未标准化
      y_list:       [(FUTURE_LEN, 3), ...]  未来 [steer, yawrate, ay], 未标准化
      curve_list:   [(FUTURE_LEN,), ...]    未来道路曲率, 未标准化
      ctx_list:     [ (5,), ... ] context = [steer_anchor, steer_rate, ay, yawrate, style_id]
      feature_cols: 特征列名顺序
    """
    df_v, df_e = load_vehicle_and_events(vehicle_file)
    if df_v is None:
        return [], [], [], [], None

    cols = df_v.columns.tolist()

    # 自动识别关键列
    col_roll     = find_col(cols, ["zx|roll", "roll", "Roll"])
    col_steer    = find_col(cols, ["zx|SteeringWheel", "SteeringWheel", "steer"])
    # vyaw 直接当 yawrate 用
    col_yawrate  = find_col(cols, ["vyaw", "zx|vyaw", "YawRate", "zx|YawRate", "yaw_rate"])
    col_v        = find_col(cols, ["zx|vx", "Vx", "vx", "Speed", "speed"])
    col_z        = find_col(cols, ["zx|z", "z", "Z"])
    col_ay       = find_col(cols, ["zx|ay", "ay", "Ay", "lat_acc"])
    col_ax       = find_col(cols, ["zx|ax", "ax", "Ax", "Long_acc"])
    col_lane     = find_col(cols, ["lateraldistance", "lateralDistance", "lateraldistance_start"])
    col_curve    = find_col(cols, ["zx1|lanecurvatureXY", "laneCurvature", "lanecurvature_start"])  # 道路曲率
    col_yaw      = find_col(cols, ["zx|yaw", "yaw", "Yaw"])

    if col_roll is None or col_steer is None:
        print(f"⚠ {vehicle_file} 缺少 roll 或 steering 列, 跳过")
        return [], [], [], [], None
    if col_ay is None or col_yawrate is None:
        print(f"⚠ {vehicle_file} 缺少 ay 或 vyaw(yawrate) 列, 跳过")
        return [], [], [], [], None
    if col_curve is None:
        print(f"⚠ {vehicle_file} 缺少道路曲率列 laneCurvature, 跳过 (v5.0 需要道路预览)")
        return [], [], [], [], None

    # 历史特征列
    base_cols = [c for c in [
        col_roll, col_yawrate, col_ay, col_ax, col_v,
        col_z, col_lane, col_curve, col_yaw, col_steer
    ] if c is not None]

    df_feat = df_v[base_cols].copy()

    # 速度统一为 m/s
    if col_v is not None:
        df_feat[col_v] = df_feat[col_v] / 3.6

    # LTR_est
    if col_ay is not None:
        df_feat["LTR_est"] = df_v[col_ay] * LTR_COEFF

    # steer_rate
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
    curve_idx      = feature_cols.index(col_curve)  # 历史中曲率索引

    # ====== 距离轴 s(t)：用于道路曲率的"速度投影预瞄"（方案A，保持预测时间轴不变） ======
    # 说明：道路曲率本质上定义在弧长 s 上。这里保持模型预测仍以时间轴 (2s/400点) 输出，
    # 但曲率输入用 anchor 速度 v0 将未来每个时间步映射到前方距离位置，并在 (s, curve) 上插值获得。
    v_idx = feature_cols.index(col_v) if (col_v is not None and col_v in feature_cols) else None
    if v_idx is not None:
        v_arr = X_all[:, v_idx].astype(np.float32)
        v_arr = np.nan_to_num(v_arr, nan=0.0)
        v_arr = np.clip(v_arr, 0.0, None)

        s_axis = np.zeros(N, dtype=np.float64)
        # s[k] = sum_{0..k-1} v[j]*dt  (用前一时刻速度积分，避免未来信息泄漏到当前累计距离)
        s_axis[1:] = np.cumsum(v_arr[:-1].astype(np.float64) * dt)
        s_axis = make_strictly_increasing(s_axis)

        curve_arr = X_all[:, curve_idx].astype(np.float32)
        curve_arr = np.nan_to_num(curve_arr, nan=0.0)
    else:
        v_arr = None
        s_axis = None
        curve_arr = None



    X_list, y_list, curve_list, ctx_list = [], [], [], []

    # 当前文件的 subject_id
    subject_id = get_subject_id_from_path(vehicle_file)
    style_id = style_map.get(subject_id, 0)  # 若不存在，默认 style_id = 0

    # 只保留 medium / strong / extreme 事件
    df_e = df_e[df_e["event_level"].isin(STRONG_LABELS)]
    if len(df_e) == 0:
        print(f"  ⚠ {vehicle_file} 中无 strong 事件, 跳过")
        return [], [], [], [], None

    for _, ev in df_e.iterrows():
        t0 = float(ev["start_s"])
        t1 = float(ev["end_s"])
        i0 = int(t0 * FS)
        i1 = int(t1 * FS)

        if i0 < 0 or i1 > N or (i1 - i0) < 10:
            continue

        # 在事件区间内找到 roll 最大值作为 anchor
        roll_seg = X_all[i0:i1, roll_idx]
        if len(roll_seg) == 0:
            continue
        peak_rel = int(np.argmax(roll_seg))
        peak_idx = i0 + peak_rel

        # 长窗口: 3s 历史 + 2s 未来
        if peak_idx - WIN_LEN < 0 or peak_idx + FUTURE_LEN >= N:
            continue

        x_win = X_all[peak_idx - WIN_LEN: peak_idx]  # (600, D)

        # 未来多任务输出: steer, yawrate, ay
        y_steer = X_all[peak_idx + 1: peak_idx + 1 + FUTURE_LEN, steer_idx]
        y_yaw   = X_all[peak_idx + 1: peak_idx + 1 + FUTURE_LEN, yawrate_idx]
        y_ay    = X_all[peak_idx + 1: peak_idx + 1 + FUTURE_LEN, ay_idx]
        y_seq = np.stack([y_steer, y_yaw, y_ay], axis=-1)  # (400,3)

        # 未来道路曲率预览（方案A：速度投影预瞄，仍保持 2s/400点 的时间轴长度不变）
        # 用 anchor 速度 v0 将未来每个时间步 t_i 映射到距离 s_i = s0 + v0*t_i，
        # 再在 (s_axis, curve_arr) 上插值得到曲率序列。若缺少速度列，则回退为按时间切片。
        if v_arr is not None and s_axis is not None and curve_arr is not None:
            v0 = float(v_arr[peak_idx])
            s0 = float(s_axis[peak_idx])

            t_grid = (np.arange(1, FUTURE_LEN + 1, dtype=np.float64) * dt)
            s_query = s0 + v0 * t_grid
            s_query = np.clip(s_query, s_axis[0], s_axis[-1])

            curve_future = np.interp(s_query, s_axis, curve_arr).astype(np.float32)
        else:
            curve_future = X_all[peak_idx + 1: peak_idx + 1 + FUTURE_LEN, curve_idx].astype(np.float32)

        steer_anchor = X_all[peak_idx, steer_idx]

        # context: [steer_anchor, steer_rate, ay, yawrate, style_id]
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

    return X_list, y_list, curve_list, ctx_list, feature_cols


def build_all_samples(style_map):
    """遍历 ROOT 下所有被试, 汇总全部事件样本"""
    pattern = os.path.join(ROOT, "*", "vehicle", "*_vehicle_aligned_cleaned.csv")
    vehicle_files = sorted(glob(pattern))

    X_pool, y_pool, curve_pool, ctx_pool = [], [], [], []
    feature_names = None

    print("🔍 正在遍历车辆文件构造事件样本...")
    total = 0
    for vf in vehicle_files:
        print(f"  ▶ {vf}")
        X_list, y_list, curve_list, ctx_list, feat_cols = build_samples_for_vehicle(vf, style_map)
        if feat_cols is None or len(X_list) == 0:
            continue

        if feature_names is None:
            feature_names = feat_cols
        else:
            if feat_cols != feature_names:
                print("  ⚠ 特征列顺序与前一个文件不一致, 需人工检查。跳过该文件。")
                continue

        X_pool.extend(X_list)
        y_pool.extend(y_list)
        curve_pool.extend(curve_list)
        ctx_pool.extend(ctx_list)
        total += len(X_list)

    print(f"\n✅ 共收集到 {total} 个事件样本\n")
    return X_pool, y_pool, curve_pool, ctx_pool, feature_names


# =========================
# 数据集: 多任务 + 曲率预览 + 风格
# =========================

class MultiTaskFutureWithCurveDataset(Dataset):
    def __init__(self, X_list, y_list, curve_list, ctx_list,
                 y_mean, y_std,
                 curve_mean, curve_std,
                 ctx_mean, ctx_std):
        """
        X_list:     标准化后的 (WIN_LEN, D)
        y_list:     原始未来轨迹 (FUTURE_LEN, 3)
        curve_list: 原始未来曲率   (FUTURE_LEN,)
        ctx_list:   原始 context   (N,5) = [steer_anchor, steer_rate, ay, yawrate, style_id]

        y_mean/std:     (3,)
        curve_mean/std: 标量
        ctx_mean/std:   (5,)
        """
        self.X = X_list
        self.y = y_list
        self.curve = curve_list
        self.ctx = ctx_list

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

        y_norm = (y - self.y_mean) / self.y_std          # (T_out,3)
        curve_norm = (curve_raw - self.curve_mean) / self.curve_std  # (T_out,)
        ctx_norm = (ctx_raw - self.ctx_mean) / self.ctx_std          # (5,)

        return {
            "src": x.astype(np.float32),
            "y_norm": y_norm.astype(np.float32),
            "curve_norm": curve_norm.astype(np.float32),
            "ctx": ctx_norm.astype(np.float32)
        }


# =========================
# 模型: Past2Future + Context + RoadPreview + MultiTask + Style
# =========================

class Past2FutureMultiTaskRoadPreview(nn.Module):
    """
    v5.0: Non-AR + Context + Road Curvature Preview + Multi-task + DriverStyle
    - 输入:
        src:   (B, WIN_LEN, D)
        ctx:   (B, 5)  ← [steer_anchor, steer_rate, ay, yawrate, style_id]
        curve: (B, FUTURE_LEN)  ← 未来 2s 道路曲率 (已标准化)
    - 输出:
        y_hat_norm: (B, FUTURE_LEN, 3)
    """
    def __init__(self,
                 input_dim,
                 context_dim,
                 future_len,
                 out_dim=3,
                 d_model=128,
                 nhead=2,
                 num_layers_enc=2,
                 num_layers_dec=2,
                 dim_feedforward=256,
                 dropout=0.1,
                 max_len_enc=600,
                 max_len_dec=400):
        super().__init__()

        self.d_model = d_model
        self.future_len = future_len
        self.out_dim = out_dim

        # ----- Encoder -----
        self.enc_input_proj = nn.Linear(input_dim, d_model)
        self.enc_pos_emb = nn.Parameter(torch.zeros(1, max_len_enc, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers_enc)

        # ----- Decoder -----
        self.dec_pos_emb = nn.Parameter(torch.zeros(1, max_len_dec, d_model))
        self.ctx_proj    = nn.Linear(context_dim, d_model)   # context_dim=5
        self.curve_proj  = nn.Linear(1, d_model)             # 曲率预览投影

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers_dec)

        self.out_proj = nn.Linear(d_model, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, ctx, curve_norm):
        """
        src:        (B, T_in, D)
        ctx:        (B, 5)
        curve_norm: (B, T_out)
        """
        B, T_in, _ = src.shape
        T_out = self.future_len

        # ----- Encoder -----
        h_src = self.enc_input_proj(src)
        h_src = h_src + self.enc_pos_emb[:, :T_in, :]
        memory = self.encoder(self.dropout(h_src))  # (B,T_in,d_model)

        # ----- Decoder 输入 -----
        pos_tgt = self.dec_pos_emb[:, :T_out, :].expand(B, T_out, -1)
        ctx_emb = self.ctx_proj(ctx).unsqueeze(1).expand(B, T_out, -1)

        curve_feat = curve_norm.unsqueeze(-1)            # (B,T_out,1)
        curve_emb = self.curve_proj(curve_feat)          # (B,T_out,d_model)

        tgt = pos_tgt + ctx_emb + curve_emb              # (B,T_out,d_model)

        out = self.decoder(tgt, memory)                  # (B,T_out,d_model)
        y_hat_norm = self.out_proj(out)                  # (B,T_out,3)
        return y_hat_norm


# =========================
# 训练 & 评估
# =========================

def main():
    print("设备:", DEVICE)

    # -------- 加载驾驶风格映射 --------
    style_map = load_driver_style_map(STYLE_CSV)

    # -------- 构造样本 --------
    X_pool, y_pool, curve_pool, ctx_pool, feature_names = build_all_samples(style_map)
    total = len(X_pool)
    if total == 0:
        print("❌ 没有有效事件样本, 请检查路径或事件文件")
        return

    # -------- Encoder 特征标准化 --------
    all_X_concat = np.concatenate(X_pool, axis=0)
    feat_mean = all_X_concat.mean(axis=0)
    feat_std = all_X_concat.std(axis=0)
    feat_std[feat_std < 1e-6] = 1e-6

    print("🔧 Encoder 特征标准化参数:")
    for name, m, s in zip(feature_names, feat_mean, feat_std):
        print(f"  {name:20s} mean={m:8.4f}, std={s:8.4f}")

    for i in range(len(X_pool)):
        X_pool[i] = (X_pool[i] - feat_mean) / feat_std

    # -------- 输出 (steer, yawrate, ay) 标准化 --------
    all_y_concat = np.concatenate([y.reshape(-1, 3) for y in y_pool], axis=0)
    y_mean = all_y_concat.mean(axis=0)
    y_std  = all_y_concat.std(axis=0)
    y_std[y_std < 1e-6] = 1e-6

    print("\n🔧 Output(steer,yawrate,ay) 标准化参数:")
    print("  mean:", y_mean)
    print("  std :", y_std)

    # -------- 曲率标准化 --------
    all_curve_concat = np.concatenate(curve_pool, axis=0)
    curve_mean = all_curve_concat.mean()
    curve_std  = all_curve_concat.std()
    if curve_std < 1e-6:
        curve_std = 1e-6

    print("\n🔧 Road curvature 标准化参数:")
    print("  mean:", curve_mean)
    print("  std :", curve_std)

    # -------- Context 标准化 (5维: steer_anchor, steer_rate, ay, yawrate, style_id)--------
    ctx_array = np.stack(ctx_pool, axis=0)
    ctx_mean = ctx_array.mean(axis=0)
    ctx_std  = ctx_array.std(axis=0)
    ctx_std[ctx_std < 1e-6] = 1e-6

    print("\n🔧 Context 标准化参数 (steer_anchor, steer_rate, ay, yawrate, style_id):")
    print("  mean:", ctx_mean)
    print("  std :", ctx_std, "\n")

    # -------- 打乱 & 划分训练/测试 --------
    idx = np.arange(total)
    np.random.shuffle(idx)
    X_pool     = [X_pool[i] for i in idx]
    y_pool     = [y_pool[i] for i in idx]
    curve_pool = [curve_pool[i] for i in idx]
    ctx_pool   = [ctx_pool[i] for i in idx]

    n_train = int(total * 0.8)
    train_dataset = MultiTaskFutureWithCurveDataset(
        X_pool[:n_train], y_pool[:n_train], curve_pool[:n_train], ctx_pool[:n_train],
        y_mean, y_std, curve_mean, curve_std, ctx_mean, ctx_std
    )
    test_dataset = MultiTaskFutureWithCurveDataset(
        X_pool[n_train:], y_pool[n_train:], curve_pool[n_train:], ctx_pool[n_train:],
        y_mean, y_std, curve_mean, curve_std, ctx_mean, ctx_std
    )

    print(f"训练集样本数: {len(train_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")
    print("输入特征顺序:", feature_names)
    print(f"历史窗口: {WIN_SEC:.1f}s ({WIN_LEN} 点), 未来窗口: {FUTURE_SEC:.1f}s ({FUTURE_LEN} 点)\n")

    # -------- 模型 & 优化器 --------
    input_dim = len(feature_names)
    context_dim = 5   # [steer_anchor, steer_rate, ay, yawrate, style_id]
    out_dim = 3

    model = Past2FutureMultiTaskRoadPreview(
        input_dim=input_dim,
        context_dim=context_dim,
        future_len=FUTURE_LEN,
        out_dim=out_dim,
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

    # -------- DataLoader --------
    def collate_fn(batch):
        srcs = [torch.from_numpy(b["src"]).float() for b in batch]
        ys   = [torch.from_numpy(b["y_norm"]).float() for b in batch]
        curves = [torch.from_numpy(b["curve_norm"]).float() for b in batch]
        ctxs = [torch.from_numpy(b["ctx"]).float() for b in batch]

        src = torch.stack(srcs, dim=0)
        y_norm = torch.stack(ys, dim=0)
        curve_norm = torch.stack(curves, dim=0)
        ctx = torch.stack(ctxs, dim=0)
        return {"src": src, "y_norm": y_norm, "curve_norm": curve_norm, "ctx": ctx}

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )

    # =========================
    # 训练循环
    # =========================
    print("🚀 开始训练 v5.0 (长窗口 + 多任务 + Context + RoadPreview + Style)...")
    best_val = np.inf
    start_all = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum, n_batch = 0.0, 0

        for batch in train_loader:
            src = batch["src"].to(DEVICE, non_blocking=True)
            y_norm_true = batch["y_norm"].to(DEVICE, non_blocking=True)
            curve_norm = batch["curve_norm"].to(DEVICE, non_blocking=True)
            ctx = batch["ctx"].to(DEVICE, non_blocking=True)

            optim.zero_grad()
            y_hat_norm = model(src, ctx, curve_norm)
            loss = F.mse_loss(y_hat_norm, y_norm_true)
            loss.backward()
            optim.step()

            loss_sum += loss.item()
            n_batch += 1

        train_loss = loss_sum / max(1, n_batch)

        # 验证
        model.eval()
        val_loss, n_val = 0.0, 0
        with torch.no_grad():
            for batch in test_loader:
                src = batch["src"].to(DEVICE, non_blocking=True)
                y_norm_true = batch["y_norm"].to(DEVICE, non_blocking=True)
                curve_norm = batch["curve_norm"].to(DEVICE, non_blocking=True)
                ctx = batch["ctx"].to(DEVICE, non_blocking=True)
                y_hat_norm = model(src, ctx, curve_norm)
                loss = F.mse_loss(y_hat_norm, y_norm_true)
                val_loss += loss.item()
                n_val += 1
        val_loss /= max(1, n_val)

        print(f"[Epoch {epoch:02d}/{EPOCHS:02d}] "
              f"Train MSE(norm)={train_loss:.6f} | Test MSE(norm)={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "best_model_v5_0_longwin_multitask_roadpreview_style.pth")
            print("  🌟 New best, 已保存 best_model_v5_0_longwin_multitask_roadpreview_style.pth\n")

    print(f"\n⌛ 总训练耗时: {(time.time() - start_all)/60:.2f} min\n")

    # =========================
    # 测试集整体指标 (以 steering 通道为主)
    # =========================
    print("📈 正在评估测试集 steering RMSE/MAE (原始单位)...")
    model.load_state_dict(torch.load("best_model_v5_0_longwin_multitask_roadpreview_style.pth",
                                     map_location=DEVICE))
    model.eval()

    all_preds, all_trues = [], []

    with torch.no_grad():
        for batch in test_loader:
            src = batch["src"].to(DEVICE, non_blocking=True)
            y_norm_true = batch["y_norm"].to(DEVICE, non_blocking=True)
            curve_norm = batch["curve_norm"].to(DEVICE, non_blocking=True)
            ctx = batch["ctx"].to(DEVICE, non_blocking=True)

            y_hat_norm = model(src, ctx, curve_norm)
            y_hat = y_hat_norm.cpu().numpy() * y_std + y_mean
            y_true = y_norm_true.cpu().numpy() * y_std + y_mean

            # 保存完整 3 维
            all_preds.append(y_hat)  # (B,400,3)
            all_trues.append(y_true)  # (B,400,3)

    all_preds = np.concatenate(all_preds, axis=0)
    all_trues = np.concatenate(all_trues, axis=0)

    # overall metrics over all 3 channels
    mse_all = np.mean((all_preds - all_trues) ** 2)
    rmse_all = np.sqrt(mse_all)
    mae_all = np.mean(np.abs(all_preds - all_trues))

    # per-channel metrics
    ch_names = ["steer", "yawrate", "ay"]
    mse_ch = np.mean((all_preds - all_trues) ** 2, axis=(0, 1))  # (3,)
    rmse_ch = np.sqrt(mse_ch)
    mae_ch = np.mean(np.abs(all_preds - all_trues), axis=(0, 1))

    print("===== 测试集指标 (未来 2s，多任务 3通道，原始单位) =====")
    print(f"Overall: MSE={mse_all:.4f} | RMSE={rmse_all:.4f} | MAE={mae_all:.4f}")
    for n, m, r, a in zip(ch_names, mse_ch, rmse_ch, mae_ch):
        print(f"  {n:7s}: MSE={m:.4f} | RMSE={r:.4f} | MAE={a:.4f}")
    print()

    # =========================
    # 随机画一条示例 steering 曲线
    # =========================
    idx = np.random.randint(0, all_preds.shape[0])
    t = np.linspace(0, FUTURE_SEC, FUTURE_LEN, endpoint=False)

    plt.figure(figsize=(11, 4))
    plt.plot(t, all_trues[idx, :, 0], label="GT Steering", linewidth=1.2)
    plt.plot(t, all_preds[idx, :, 0], "--", label="Pred Steering (v5.0 road-preview+style)", linewidth=1.2)
    plt.xlabel("Time (s)")
    plt.ylabel("Steering angle")
    plt.title("Future 2s Steering Prediction (v5.0 LongWin + MultiTask + RoadPreview + Style)")
    plt.legend()
    plt.tight_layout()
    out_fig = "steer_future2s_v5_0_example.png"
    plt.savefig(out_fig, dpi=150)
    plt.close()
    print(f"📊 已保存示例图: {out_fig}\n")
    # =========================
    # 额外绘制 yawrate / ay 预测曲线
    # =========================
    print("📊 正在绘制 yawrate & ay 预测对比图 ...")

    # 随机取一个样本
    idx2 = np.random.randint(0, all_preds.shape[0])
    t = np.linspace(0, FUTURE_SEC, FUTURE_LEN, endpoint=False)

    # ---- yawrate ----
    plt.figure(figsize=(11,4))
    plt.plot(t, all_trues[idx2, :, 1], label="GT YawRate", linewidth=1.2)
    plt.plot(t, all_preds[idx2, :, 1], "--", label="Pred YawRate", linewidth=1.2)
    plt.xlabel("Time (s)")
    plt.ylabel("YawRate")
    plt.title("Future 2s YawRate Prediction (v5.0)")
    plt.legend()
    plt.tight_layout()
    out_fig_yaw = "yawrate_future2s_v5_0_example.png"
    plt.savefig(out_fig_yaw, dpi=150)
    plt.close()
    print(f"📈 已保存：{out_fig_yaw}")

    # ---- ay ----
    plt.figure(figsize=(11,4))
    plt.plot(t, all_trues[idx2, :, 2], label="GT Ay", linewidth=1.2)
    plt.plot(t, all_preds[idx2, :, 2], "--", label="Pred Ay", linewidth=1.2)
    plt.xlabel("Time (s)")
    plt.ylabel("Lateral Acc (Ay)")
    plt.title("Future 2s Lateral Acceleration (Ay) Prediction (v5.0)")
    plt.legend()
    plt.tight_layout()
    out_fig_ay = "ay_future2s_v5_0_example.png"
    plt.savefig(out_fig_ay, dpi=150)
    plt.close()
    print(f"📈 已保存：{out_fig_ay}\n")

    # =========================
    # 保存 checkpoint
    # =========================
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
        "config": {
            "FS": FS,
            "WIN_SEC": WIN_SEC,
            "FUTURE_SEC": FUTURE_SEC,
            "WIN_LEN": WIN_LEN,
            "FUTURE_LEN": FUTURE_LEN,
            "D_MODEL": D_MODEL,
            "N_HEAD": N_HEAD,
            "NUM_LAYERS_ENC": NUM_LAYERS_ENC,
            "NUM_LAYERS_DEC": NUM_LAYERS_DEC,
            "FFN_DIM": FFN_DIM,
            "DROPOUT": DROPOUT,
            "MODEL_VER": "v5_0_longwin_multitask_roadpreview_style"
        }
    }
    torch.save(ckpt, "model_rollpeak_transformer_v5_0_longwin_multitask_roadpreview_style.pth")
    print("💾 已保存模型 checkpoint: model_rollpeak_transformer_v5_0_longwin_multitask_roadpreview_style.pth\n")


if __name__ == "__main__":
    main()

# =========================================================
# INPUT FEATURE PRUNING (AUTO-APPLIED)
# Keep only: vx, ay, yawrate, steer, steer_rate, lane_curvature
# This override ensures minimal, high-correlation inputs.
# =========================================================
def _override_feature_cols(df, col_v, col_ay, col_yaw_rate, col_steer, col_steer_rate, col_curve):
    feature_cols = []
    if col_v in df.columns:
        feature_cols.append(col_v)
    if col_ay in df.columns:
        feature_cols.append(col_ay)
    if col_yaw_rate in df.columns:
        feature_cols.append(col_yaw_rate)
    if col_steer in df.columns:
        feature_cols.append(col_steer)
    if col_steer_rate in df.columns:
        feature_cols.append(col_steer_rate)
    if col_curve in df.columns:
        feature_cols.append(col_curve)
    return feature_cols
