# -*- coding: utf-8 -*-
"""
future_steer_event_rollpeak_transformer_v2.py
=============================================

V2（Transformer 强化版）：
- 模型：小型 Transformer Encoder + Attention Pooling + MLP
- anchor_s = roll_peak（与 baseline 一致）
- 输入序列：anchor 前 2s（400 × D）
- 输出：anchor 后 0.5s 时刻的方向盘角（标量回归）
- 事件：仅使用 medium_active / strong_active / extreme_active
- 特征：roll / yawrate / ay / ax / v(m/s) / lateral_distance /
        lanecurvatureXY / yaw / steering / LTR_est （共 10 维，若缺失则自动降维）
- 进行了 feature-wise 标准化（mean/std），适合 Transformer 训练
- 自动保存 checkpoint：model_rollpeak_transformer_v2.pth
  内容包括：
    - "state_dict"      : 模型权重
    - "feature_names"   : 特征名称顺序
    - "feat_mean"       : 标准化用均值（np.array，shape=[D]）
    - "feat_std"        : 标准化用标准差（np.array，shape=[D]）

说明：
- v1 = LSTM + 无归一化 的 “论文 baseline”
- v2 = Transformer + 归一化 的 “强化版本”，你可以在论文里对比展示。
"""

import os
from glob import glob

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# =============== 配置 ===============

ROOT = r"F:\数据集处理\data_process\datasetprocess\多模态数据\被试数据集合"

FS = 200
WIN_SEC = 2.0         # 输入窗口长度（秒）
PRED_SEC = 0.5        # 预测未来时间（秒）

BATCH_SIZE = 256
EPOCHS = 40
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LTR_COEFF = 0.11243
STRONG_LABELS = ["medium_active", "strong_active", "extreme_active"]

# Transformer 结构参数（小模型，防止过拟合）
D_MODEL = 128         # Transformer 内部特征维度
N_HEAD = 2            # Multi-head Attention 头数
NUM_LAYERS = 2        # Encoder 层数
FFN_DIM = 256         # 前馈网络维度
DROPOUT = 0.1

# 固定随机种子，方便复现
SEED = 2025
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =============== 辅助函数 ===============

def find_col(cols, candidates):
    """在列名列表中找第一个匹配的候选名，没有则返回 None"""
    for c in candidates:
        if c in cols:
            return c
    return None


def load_vehicle_and_events(vehicle_file):
    """给定 vehicle 文件路径，自动推断 event 文件路径并读取"""
    event_file = vehicle_file.replace("\\vehicle\\", "\\event\\") \
                             .replace("_vehicle_aligned_cleaned.csv",
                                      "_vehicle_aligned_cleaned_events_v312.csv")

    if not os.path.exists(event_file):
        print(f"⚠ 事件文件不存在：{event_file}")
        return None, None

    df_v = pd.read_csv(vehicle_file)
    df_e = pd.read_csv(event_file)
    return df_v, df_e


# =============== 构造事件样本（原始分布） ===============

def build_event_samples(df_v, df_e):
    """
    从单个文件构造事件样本：
    - 筛选 medium/strong/extreme 事件
    - 对每个事件：
        * 根据 start_s/end_s 截取片段
        * 在片段内找 roll 最大值作为 anchor
        * 取 anchor 前 WIN_SEC 秒作为输入窗口
        * 取 anchor 后 PRED_SEC 秒的 steering 作为目标
    返回：
        X_list: List[np.ndarray], 每个 shape=(win_len, D)
        y_list: List[float]
        feature_cols: 特征列名顺序
    """
    cols = df_v.columns.tolist()

    col_roll  = find_col(cols, ["zx|roll"])
    col_steer = find_col(cols, ["zx|SteeringWheel"])
    if col_roll is None or col_steer is None:
        print("⚠ 缺少 roll 或 steering，跳过该文件")
        return [], [], None

    # 其他特征列
    col_yawrate = find_col(cols, ["zx|vyaw"])
    col_ay      = find_col(cols, ["zx|ay"])
    col_ax      = find_col(cols, ["zx|ax"])
    col_v       = find_col(cols, ["zx1|v_km/h"])
    col_lane    = find_col(cols, ["zx1|lateraldistance"])
    col_curve   = find_col(cols, ["zx1|lanecurvatureXY"])
    col_yaw     = find_col(cols, ["zx|yaw"])

    feature_cols = [c for c in [
        col_roll, col_yawrate, col_ay, col_ax, col_v,
        col_lane, col_curve, col_yaw, col_steer
    ] if c is not None]

    df_feat = df_v[feature_cols].copy()

    # km/h → m/s
    if col_v:
        df_feat[col_v] = df_feat[col_v] / 3.6

    # LTR_est
    if col_ay:
        df_feat["LTR_est"] = df_feat[col_ay] * LTR_COEFF
        feature_cols.append("LTR_est")

    # 转为 numpy
    X_all = df_feat.to_numpy(dtype=np.float32)
    N = X_all.shape[0]

    steer_idx = feature_cols.index(col_steer)
    roll_idx = feature_cols.index(col_roll)

    win_len = int(WIN_SEC * FS)
    shift   = int(PRED_SEC * FS)

    X_list, y_list = [], []

    # 仅保留 medium / strong / extreme
    df_e = df_e[df_e["event_level"].isin(STRONG_LABELS)]
    if len(df_e) == 0:
        return [], [], None

    for _, ev in df_e.iterrows():
        t0 = float(ev["start_s"])
        t1 = float(ev["end_s"])
        i0 = int(round(t0 * FS))
        i1 = int(round(t1 * FS))
        if i0 < 0 or i1 >= N or i0 >= i1:
            continue

        # roll 最大处作为 anchor
        roll_seg = X_all[i0:i1, roll_idx]
        if roll_seg.size == 0:
            continue
        peak_rel = np.argmax(roll_seg)
        peak_idx = i0 + peak_rel

        # 确保窗口和预测点都在范围内
        if peak_idx - win_len < 0 or peak_idx + shift >= N:
            continue

        s = peak_idx - win_len
        e = peak_idx

        X_list.append(X_all[s:e, :])                 # (win_len, D)
        y_list.append(X_all[peak_idx + shift, steer_idx])  # 标量

    return X_list, y_list, feature_cols


# =============== 数据集 ===============

class SteerEventDataset(Dataset):
    def __init__(self, X_list, y_list):
        """
        X_list: List[np.ndarray]，每个 (T, D)
        y_list: List[float]
        """
        self.X = X_list
        self.y = np.array(y_list, dtype=np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]  # (T, D), np.float32
        y = self.y[idx]
        return {
            "x": x,
            "y": y
        }


# =============== 模型结构：小型 Transformer ===============

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=2,
                 num_layers=2, dim_feedforward=256, dropout=0.1,
                 max_len=400):
        """
        input_dim: 输入特征维度（如 10）
        d_model:   Transformer 内部维度
        nhead:     Multi-head Attention 头数
        num_layers:Encoder 层数
        max_len:   最大序列长度（这里固定为 2s*200Hz=400）
        """
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.max_len = max_len

        # 1) 线性投影：input_dim -> d_model
        self.input_proj = nn.Linear(input_dim, d_model)

        # 2) 可学习位置编码（比 sin/cos 简单）
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))

        # 3) Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # (B, T, E)
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 4) Attention Pooling：对时间维做加权平均
        self.attn = nn.Linear(d_model, 1)

        # 5) MLP 回归头
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (B, T, input_dim)
        """
        B, T, D = x.shape
        # 投影
        h = self.input_proj(x)  # (B, T, d_model)

        # 位置编码截取相同长度
        if T > self.max_len:
            raise ValueError(f"序列长度 T={T} 超过 max_len={self.max_len}")
        pos = self.pos_emb[:, :T, :]  # (1, T, d_model)

        h = h + pos
        h = self.dropout(h)

        # Transformer 编码
        h = self.encoder(h)  # (B, T, d_model)

        # Attention pooling
        scores = self.attn(h)             # (B, T, 1)
        alpha = torch.softmax(scores, dim=1)
        ctx = (alpha * h).sum(dim=1)      # (B, d_model)

        # 回归输出
        out = self.head(ctx).squeeze(-1)  # (B,)
        return out


# =============== 主程序 ===============

def main():

    all_vehicle_files = sorted(
        glob(os.path.join(ROOT, "*", "vehicle", "*_vehicle_aligned_cleaned.csv"))
    )

    X_pool, y_pool = [], []
    feature_names = None

    print("📂 正在收集 strong/medium/extreme 事件（原始分布）...")

    for vf in all_vehicle_files:
        df_v, df_e = load_vehicle_and_events(vf)
        if df_v is None or df_e is None:
            continue

        X_list, y_list, feats = build_event_samples(df_v, df_e)
        if feats is None:
            continue

        X_pool.extend(X_list)
        y_pool.extend(y_list)
        feature_names = feats  # 所有文件结构应一致

    total = len(y_pool)
    print("\n==============================")
    print(f"原始事件样本总数：{total}")
    print("==============================\n")

    if total == 0:
        print("❌ 没有有效事件样本，检查事件标签或路径配置")
        return

    # -------- 特征标准化（非常重要，适合 Transformer） --------
    # 将所有窗口在时间维拼起来： (total*T, D)
    all_X_concat = np.concatenate(X_pool, axis=0)   # (total*T, D)
    feat_mean = all_X_concat.mean(axis=0)           # (D,)
    feat_std = all_X_concat.std(axis=0)
    feat_std[feat_std < 1e-6] = 1e-6                # 防止除零

    print("🔧 特征标准化：")
    for name, m, s in zip(feature_names, feat_mean, feat_std):
        print(f"  {name:20s} mean={m:8.4f}, std={s:8.4f}")

    # 对每个窗口做 (x-mean)/std
    for i in range(len(X_pool)):
        X_pool[i] = (X_pool[i] - feat_mean) / feat_std

    # 打乱
    idx = np.arange(total)
    np.random.shuffle(idx)
    X_pool = [X_pool[i] for i in idx]
    y_pool = [y_pool[i] for i in idx]

    # 80/20 划分
    n_train = int(total * 0.8)
    train_dataset = SteerEventDataset(X_pool[:n_train], y_pool[:n_train])
    test_dataset  = SteerEventDataset(X_pool[n_train:], y_pool[n_train:])

    print(f"\n训练集样本数：{len(train_dataset)}")
    print(f"测试集样本数：{len(test_dataset)}")
    print("输入特征顺序：", feature_names)
    print(f"每个样本时间长度：{int(WIN_SEC*FS)} 点\n")

    input_dim = len(feature_names)
    model = TimeSeriesTransformer(
        input_dim=input_dim,
        d_model=D_MODEL,
        nhead=N_HEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=FFN_DIM,
        dropout=DROPOUT,
        max_len=int(WIN_SEC * FS)
    ).to(DEVICE)

    optim = torch.optim.Adam(model.parameters(), lr=LR)

    def collate_fn(batch):
        """
        默认 collate 会把 np.array 变成 float64 Tensor，
        这里手动转成 float32，避免类型不一致。
        """
        xs = [torch.from_numpy(b["x"]).float() for b in batch]  # (T, D)
        ys = [b["y"] for b in batch]
        x = torch.stack(xs, dim=0)  # (B, T, D)
        y = torch.tensor(ys, dtype=torch.float32)
        return {"x": x, "y": y}

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    # ======= 训练 =======
    for epoch in range(1, EPOCHS + 1):

        # ---- Train ----
        model.train()
        tr_loss = 0.0

        for batch in train_loader:
            x = batch["x"].to(DEVICE)  # (B, T, D)
            y = batch["y"].to(DEVICE)  # (B,)

            optim.zero_grad()
            y_hat = model(x)

            loss = F.mse_loss(y_hat, y)
            loss.backward()
            optim.step()

            tr_loss += loss.item() * x.size(0)

        tr_loss /= len(train_dataset)

        # ---- Test ----
        model.eval()
        ts_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                x = batch["x"].to(DEVICE)
                y = batch["y"].to(DEVICE)
                y_hat = model(x)

                loss = F.mse_loss(y_hat, y)
                ts_loss += loss.item() * x.size(0)

        ts_loss /= len(test_dataset)

        print(f"[Epoch {epoch:02d}] Train MSE={tr_loss:.6f} | Test MSE={ts_loss:.6f}")

    # ======= 保存 checkpoint =======
    ckpt = {
        "state_dict": model.state_dict(),
        "feature_names": feature_names,
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "config": {
            "FS": FS,
            "WIN_SEC": WIN_SEC,
            "PRED_SEC": PRED_SEC,
            "D_MODEL": D_MODEL,
            "N_HEAD": N_HEAD,
            "NUM_LAYERS": NUM_LAYERS,
            "FFN_DIM": FFN_DIM,
            "DROPOUT": DROPOUT
        }
    }
    torch.save(ckpt, "model_rollpeak_transformer_v2.pth")
    print("\n🎉 已保存 Transformer 模型 checkpoint：model_rollpeak_transformer_v2.pth")


if __name__ == "__main__":
    main()
