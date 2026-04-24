# -*- coding: utf-8 -*-
"""
driver_style_gru_ae_seq_v1_gpu.py
=================================
基于【强事件序列 + GRU AutoEncoder】的驾驶风格聚类（舍弃 weak 事件）
GPU 版：自动优先使用 CUDA（如 RTX2060），否则回退 CPU

数据结构假定：
F:\数据集处理\data_process\datasetprocess\多模态数据\被试数据集合\<subject>\
    ├─ vehicle\*vehicle_aligned_cleaned.csv
    └─ event\*vehicle_aligned_cleaned_events_v312.csv
"""

import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 支持中文
plt.rcParams["axes.unicode_minus"] = False

# ============================
# 0. 配置
# ============================
SUBJECT_ROOT = r"F:\数据集处理\data_process\datasetprocess\多模态数据\被试数据集合"
OUTPUT_ROOT = r"F:\driver_style_subject_level_results\GRUAE_seq_window2s_3s"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

LOG_PATH = os.path.join(OUTPUT_ROOT, "log.txt")
LOG = open(LOG_PATH, "w", encoding="utf-8")


def log(msg: str):
    print(msg)
    LOG.write(msg + "\n")


# 采样率与窗口
FS = 200.0
WIN_BEFORE = 2.0  # s
WIN_AFTER = 3.0   # s
SEQ_LEN = int((WIN_BEFORE + WIN_AFTER) * FS)  # 1000

# 使用的车辆特征列名（来自你给的列）
VEH_COLS = [
    "zx|SteeringWheel",
    "zx|vyaw",
    "zx|ax",
    "zx|ay",
    "zx|roll",
    "zx1|v_km/h",
]

# 只保留这些事件等级（自动舍弃 weak）
VALID_EVENT_LEVELS = {"medium_active", "strong_active", "extreme_active"}

RANDOM_SEED = 2025
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ============================
# 1. 工具函数
# ============================

def ensure_time_column(df: pd.DataFrame) -> np.ndarray:
    """
    返回统一的时间轴 t_s（单位：秒）
    优先使用已有的 t_s；否则尝试 StorageTime；否则用索引 / FS 近似
    """
    if "t_s" in df.columns:
        return df["t_s"].astype(float).values

    if "StorageTime" in df.columns:
        t_raw = pd.to_datetime(df["StorageTime"], errors="coerce")
        if not t_raw.isna().all():
            t0 = t_raw.iloc[0]
            return (t_raw - t0).dt.total_seconds().values

    # 回退：假设已经是匀速 200Hz 采样
    log("⚠ 未找到 t_s / StorageTime，使用索引/FS 近似时间")
    n = len(df)
    return np.arange(n) / FS


def extract_event_sequence(
    veh_df: pd.DataFrame,
    t_s: np.ndarray,
    anchor_s: float,
    seq_len: int = SEQ_LEN,
    fs: float = FS,
) -> np.ndarray:
    """
    从车辆数据中截取以 anchor_s 为中心（前2s+后3s）的序列，长度固定为 seq_len
    特征：Steering, SteeringRate(自动计算), vyaw, ax, ay, roll, v_km/h
    返回形状：(seq_len, 7)
    """
    # 找到 anchor_s 对应的最近索引
    idx_anchor = int(round(anchor_s * fs))
    half_before = int(WIN_BEFORE * fs)
    half_after = int(WIN_AFTER * fs)

    start_idx = idx_anchor - half_before
    end_idx = idx_anchor + half_after  # 不含 end_idx

    n = len(veh_df)
    if end_idx <= 0 or start_idx >= n:
        # 完全越界，返回 None
        return None

    # 边界裁剪
    start_idx = max(0, start_idx)
    end_idx = min(n, end_idx)

    # 取原始特征
    for col in VEH_COLS:
        if col not in veh_df.columns:
            raise ValueError(f"车辆文件缺少列：{col}")

    seg = veh_df.loc[start_idx:end_idx - 1, VEH_COLS].values  # (L, 6)

    # 计算方向盘角速度（简易差分 * fs）
    steer = seg[:, 0]
    steer_rate = np.gradient(steer, edge_order=1) * fs  # (L,)
    steer_rate = steer_rate.reshape(-1, 1)

    # 组合特征：[steer, steer_rate, vyaw, ax, ay, roll, v_km/h]
    feat = np.concatenate([seg[:, [0]], steer_rate, seg[:, 1:]], axis=1)  # (L, 7)

    L = feat.shape[0]
    if L == seq_len:
        return feat

    if L > seq_len:
        # 直接截取中间部分
        extra = L - seq_len
        cut_start = extra // 2
        cut_end = cut_start + seq_len
        return feat[cut_start:cut_end, :]

    # L < seq_len：用边界值重复填充
    pad_before = (seq_len - L) // 2
    pad_after = seq_len - L - pad_before
    feat_padded = np.vstack([
        np.repeat(feat[0:1, :], pad_before, axis=0),
        feat,
        np.repeat(feat[-1:, :], pad_after, axis=0),
    ])
    return feat_padded


# ============================
# 2. 扫描所有被试 & 事件，构造序列数据集
# ============================

all_seqs = []        # (N_events, SEQ_LEN, feat_dim)
all_subject_ids = [] # 与上对应
all_event_ids = []   # 方便追踪：subject#file#index

subject_dirs = []
for p in glob.glob(os.path.join(SUBJECT_ROOT, "*")):
    if not os.path.isdir(p):
        continue
    name = os.path.basename(p)
    # 只保留全部是字母的目录作为被试（过滤 “驾驶风格聚类结果_xx” 这种）
    if name.isalpha():
        subject_dirs.append(p)

subject_dirs = sorted(subject_dirs)
log(f"🔍 发现被试目录：{len(subject_dirs)} 个 -> {', '.join(os.path.basename(p) for p in subject_dirs)}")

for sub_dir in subject_dirs:
    sid = os.path.basename(sub_dir)
    veh_dir = os.path.join(sub_dir, "vehicle")
    evt_dir = os.path.join(sub_dir, "event")
    if not (os.path.isdir(veh_dir) and os.path.isdir(evt_dir)):
        log(f"⚠ 跳过 {sid}：缺少 vehicle 或 event 目录")
        continue

    event_files = sorted(glob.glob(os.path.join(evt_dir, "*_vehicle_aligned_cleaned_events_v312.csv")))
    if not event_files:
        log(f"⚠ 被试 {sid} 未找到 v312 事件文件，跳过")
        continue

    log(f"\n===== 被试 {sid} =====")
    log(f"  事件文件数：{len(event_files)}")

    for evt_path in event_files:
        evt_name = os.path.basename(evt_path)
        base = evt_name.replace("_vehicle_aligned_cleaned_events_v312.csv", "")
        veh_path = os.path.join(veh_dir, base + "_vehicle_aligned_cleaned.csv")

        if not os.path.exists(veh_path):
            log(f"  ⚠ 未找到匹配的车辆文件：{veh_path}")
            continue

        log(f"  ▶ 处理记录：{base}")

        # 读取车辆数据
        veh_df = pd.read_csv(veh_path)
        t_s = ensure_time_column(veh_df)

        # 读取事件数据
        evt_df = pd.read_csv(evt_path)

        # 过滤事件：只保留 medium/strong/extreme，且 keep_for_training=1（如果有该列）
        if "event_level" in evt_df.columns:
            evt_df = evt_df[evt_df["event_level"].isin(list(VALID_EVENT_LEVELS))]
        if "keep_for_training" in evt_df.columns:
            evt_df = evt_df[evt_df["keep_for_training"].astype(int) == 1]

        if len(evt_df) == 0:
            log("    ⚠ 无符合条件的强事件，跳过本记录")
            continue

        log(f"    ✔ 强事件数：{len(evt_df)}")

        # 遍历每个事件
        for idx, row in evt_df.iterrows():
            start_s = float(row["start_s"])
            end_s = float(row["end_s"])
            # anchor ≈ 事件中心时刻，用来近似 roll_peak 发生时间
            anchor_s = 0.5 * (start_s + end_s)

            seq = extract_event_sequence(veh_df, t_s, anchor_s)
            if seq is None:
                continue

            all_seqs.append(seq)
            all_subject_ids.append(sid)
            all_event_ids.append(f"{sid}|{base}|event_idx{idx}")

log(f"\n📦 总共构造事件序列：{len(all_seqs)} 条")

if len(all_seqs) == 0:
    log("❌ 没有任何事件序列，检查路径/列名配置！")
    LOG.close()
    raise SystemExit

all_seqs = np.stack(all_seqs, axis=0)  # (N_events, T, F)
N_events, T, F_dim = all_seqs.shape
log(f"事件序列形状：{all_seqs.shape} = (N_events, SEQ_LEN, feat_dim)")

# ============================
# 3. 特征标准化（逐特征）
# ============================
flat = all_seqs.reshape(-1, F_dim)  # (N_events*T, F_dim)
scaler = StandardScaler()
flat_std = scaler.fit_transform(flat)
all_seqs_std = flat_std.reshape(N_events, T, F_dim)

# ============================
# 4. 构建 GRU-AE 模型（GPU 优先）
# ============================

class EventSeqDataset(Dataset):
    def __init__(self, X):
        # X: (N, T, F)
        self.X = torch.from_numpy(X).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx]


class GRUEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=16, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        # x: (B, T, F)
        _, h_last = self.gru(x)  # h_last: (num_layers, B, hidden_dim)
        h_last = h_last[-1]      # (B, hidden_dim)
        z = self.fc_mu(h_last)   # (B, latent_dim)
        return z


class GRUDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim=64, output_dim=F_dim, num_layers=2, seq_len=SEQ_LEN):
        super().__init__()
        self.seq_len = seq_len
        self.fc_init = nn.Linear(latent_dim, hidden_dim)
        self.gru = nn.GRU(
            output_dim,  # 我们用输出维度作为每步输入的维度
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        # z: (B, latent_dim)
        B = z.size(0)
        h0 = self.fc_init(z).unsqueeze(0).repeat(self.gru.num_layers, 1, 1)  # (num_layers, B, hidden_dim)
        # 解码时使用零输入序列
        dec_in = torch.zeros(B, self.seq_len, F_dim, device=z.device)
        out, _ = self.gru(dec_in, h0)
        x_hat = self.fc_out(out)  # (B, T, F)
        return x_hat


class GRUAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=16, num_layers=2, seq_len=SEQ_LEN):
        super().__init__()
        self.encoder = GRUEncoder(input_dim, hidden_dim, latent_dim, num_layers)
        self.decoder = GRUDecoder(latent_dim, hidden_dim, input_dim, num_layers, seq_len)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"\n🖥 使用设备：{device}")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

dataset = EventSeqDataset(all_seqs_std)
# 简单划分 80% 训练 / 20% 验证
indices = np.arange(N_events)
np.random.shuffle(indices)
split = int(0.8 * N_events)
train_idx, val_idx = indices[:split], indices[split:]

train_ds = torch.utils.data.Subset(dataset, train_idx)
val_ds = torch.utils.data.Subset(dataset, val_idx)

pin_memory = (device.type == "cuda")

train_loader = DataLoader(
    train_ds,
    batch_size=64,
    shuffle=True,
    drop_last=False,
    pin_memory=pin_memory,
    num_workers=0,
)
val_loader = DataLoader(
    val_ds,
    batch_size=64,
    shuffle=False,
    drop_last=False,
    pin_memory=pin_memory,
    num_workers=0,
)

model = GRUAutoEncoder(
    input_dim=F_dim,
    hidden_dim=64,
    latent_dim=16,
    num_layers=2,
    seq_len=SEQ_LEN
).to(device)

criterion = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
torch.cuda.empty_cache()

# ============================
# 5. 训练 GRU-AE
# ============================
EPOCHS = 40
log(f"\n⏳ 开始训练 GRU-AE，共 {EPOCHS} 轮…")

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        batch = batch.to(device, non_blocking=True).float()
        optimizer.zero_grad()
        x_hat, z = model(batch)
        loss = criterion(x_hat, batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch.size(0)
    train_loss /= len(train_ds)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device, non_blocking=True).float()
            x_hat, z = model(batch)
            loss = criterion(x_hat, batch)
            val_loss += loss.item() * batch.size(0)
    val_loss /= len(val_ds)

    log(f"[Epoch {epoch:03d}] Train MSE={train_loss:.6f} | Val MSE={val_loss:.6f}")

# 保存模型
torch.save(model.state_dict(), os.path.join(OUTPUT_ROOT, "gru_ae_driver_style.pth"))
log("💾 已保存 GRU-AE 模型权重")

# ============================
# 6. 提取所有事件的 latent 表达
# ============================
model.eval()
all_latent = []

with torch.no_grad():
    for i in range(0, N_events, 256):
        batch_np = all_seqs_std[i:i+256]
        batch = torch.from_numpy(batch_np).to(device, non_blocking=True).float()
        z = model.encoder(batch)
        all_latent.append(z.cpu().numpy())

all_latent = np.concatenate(all_latent, axis=0)  # (N_events, latent_dim)
latent_dim = all_latent.shape[1]
log(f"\n🧬 事件 latent 形状：{all_latent.shape}")
# ===== 保存事件原始序列（1000×7） =====
save_seqs_path = os.path.join(OUTPUT_ROOT, "all_seqs.npy")
np.save(save_seqs_path, all_seqs)
log(f"💾 已保存事件序列 all_seqs.npy，路径：{save_seqs_path}")


# 保存事件级 latent
events_latent_df = pd.DataFrame(all_latent, columns=[f"z{i+1}" for i in range(latent_dim)])
events_latent_df["subject"] = all_subject_ids
events_latent_df["event_id"] = all_event_ids
events_latent_df.to_csv(os.path.join(OUTPUT_ROOT, "events_latent.csv"), index=False, encoding="utf-8-sig")

# ============================
# 7. 被试级聚合（mean + std）
# ============================
rows = []
for sid in sorted(set(all_subject_ids)):
    idx = [i for i, s in enumerate(all_subject_ids) if s == sid]
    z_sub = all_latent[idx, :]  # (n_evt, latent_dim)
    row = {"subject": sid}
    for j in range(latent_dim):
        row[f"z{j+1}_mean"] = z_sub[:, j].mean()
        row[f"z{j+1}_std"] = z_sub[:, j].std()
    rows.append(row)

df_subject = pd.DataFrame(rows)
df_subject = df_subject.fillna(0.0)
df_subject.to_csv(os.path.join(OUTPUT_ROOT, "driver_style_subject_latent.csv"), index=False, encoding="utf-8-sig")
log(f"\n👥 被试级特征表已保存，shape={df_subject.shape}")

# ============================
# 8. 第一层聚类：K=2（主风格：Steady vs Active）
# ============================

log("\n==============================")
log("🔵 第一层聚类：K=2（主风格：Steady vs Active）")
log("==============================")

feat_cols = [c for c in df_subject.columns if c != "subject"]
X = df_subject[feat_cols].values
X_std = StandardScaler().fit_transform(X)

# K=2 主聚类
kmeans2 = KMeans(n_clusters=2, random_state=RANDOM_SEED, n_init=20)
labels2 = kmeans2.fit_predict(X_std)
df_subject["cluster_main_k2"] = labels2

# 主聚类中心
centers2 = pd.DataFrame(kmeans2.cluster_centers_, columns=feat_cols)
centers2.to_csv(os.path.join(OUTPUT_ROOT, "cluster_centers_main_k2.csv"),
                index=False, encoding="utf-8-sig")
log("📌 已保存主聚类(K=2)中心：cluster_centers_main_k2.csv")

# ===== 自动判定哪一类是 Steady / Active =====
# 用中心向量的 L2 范数衡量“整体激烈程度”，越大越激进
main_scores = [(i, np.linalg.norm(centers2.iloc[i].values)) for i in range(2)]
main_scores = sorted(main_scores, key=lambda x: x[1])  # 从稳到激进

idx_steady_main = main_scores[0][0]
idx_active_main = main_scores[1][0]

cluster_main_to_name = {
    idx_steady_main: "Steady_main",   # 总体更稳
    idx_active_main: "Active",        # 总体更激进
}

df_subject["style_main"] = df_subject["cluster_main_k2"].map(cluster_main_to_name)

log(f"👉 主风格映射：{cluster_main_to_name}")
log("   Steady_main = 总体较稳一类")
log("   Active      = 总体较激进一类")


# ============================
# 9. 第二层：只在 Steady_main 里再聚类成 Steady / Conservative
# ============================

log("\n==============================")
log("🟦 第二层聚类：只在 Steady_main 内部再做 K=2（Steady / Conservative）")
log("==============================")

mask_steady = (df_subject["style_main"] == "Steady_main")
if mask_steady.sum() < 2:
    log("⚠ Steady_main 内被试数 < 2，无法再细分，全部当作 Steady 处理")
    df_subject["style_3style"] = df_subject["style_main"].replace({"Steady_main": "Steady"})
else:
    # 取 Steady_main 内部的样本
    X_steady = X_std[mask_steady]
    subjects_steady = df_subject.loc[mask_steady, "subject"].values

    # 在 Steady_main 内再做 K=2
    kmeans_sub = KMeans(n_clusters=2, random_state=RANDOM_SEED, n_init=20)
    sub_labels = kmeans_sub.fit_predict(X_steady)

    # 子聚类中心
    centers_sub = pd.DataFrame(kmeans_sub.cluster_centers_, columns=feat_cols)
    centers_sub.to_csv(os.path.join(OUTPUT_ROOT, "cluster_centers_steady_sub_k2.csv"),
                       index=False, encoding="utf-8-sig")
    log("📌 已保存 Steady_main 内部子聚类中心：cluster_centers_steady_sub_k2.csv")

    # 判定哪一类更稳 / 哪一类偏稳（保守）
    sub_scores = [(i, np.linalg.norm(centers_sub.iloc[i].values)) for i in range(2)]
    sub_scores = sorted(sub_scores, key=lambda x: x[1])  # 从稳到不那么稳

    idx_steady = sub_scores[0][0]        # 最稳
    idx_conservative = sub_scores[1][0]  # 稍微“更有动作”的那群

    # 先全部默认成 Active，再覆盖 Steady_main 内部
    df_subject["style_3style"] = "Active"
    # 在 Steady_main 里面，根据子聚类再拆成 Steady / Conservative
    for sid, lbl in zip(subjects_steady, sub_labels):
        if lbl == idx_steady:
            df_subject.loc[df_subject["subject"] == sid, "style_3style"] = "Steady"
        else:
            df_subject.loc[df_subject["subject"] == sid, "style_3style"] = "Conservative"

    log(f"👉 Steady_main 内部细分：子簇 {idx_steady} = Steady，子簇 {idx_conservative} = Conservative")

# 保存三风格标签
df_subject.to_csv(os.path.join(OUTPUT_ROOT, "driver_style_labels_3style.csv"),
                  index=False, encoding="utf-8-sig")
log("🎉 已保存三风格结果：driver_style_labels_3style.csv")
log("   三类风格：Steady / Conservative / Active")


# ============================
# 10. UMAP 可视化（三风格）
# ============================

log("\n==============================")
log("🎨 UMAP 可视化（三风格：Steady / Conservative / Active）")
log("==============================")

try:
    from umap import UMAP

    umap_model = UMAP(
        n_components=2,
        n_neighbors=5,
        min_dist=0.1,
        random_state=RANDOM_SEED,
    )
    X_umap = umap_model.fit_transform(X_std)

    plt.figure(figsize=(7, 6))

    color_map = {
        "Steady": "#1f77b4",        # 蓝
        "Conservative": "#ff7f0e",  # 橙
        "Active": "#2ca02c",        # 绿（原来的激进类）
    }

    for style, color in color_map.items():
        mask = (df_subject["style_3style"] == style)
        if mask.sum() == 0:
            continue
        plt.scatter(
            X_umap[mask, 0],
            X_umap[mask, 1],
            s=80,
            alpha=0.85,
            label=style,
            color=color
        )

    # 标注被试 ID
    for i, sid in enumerate(df_subject["subject"]):
        plt.text(X_umap[i, 0], X_umap[i, 1], sid, fontsize=8)

    plt.title("驾驶风格聚类（三风格：Steady / Conservative / Active）", fontsize=14)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_ROOT, "umap_driver_style_3style.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    log(f"📊 UMAP(三风格) 图已保存：{save_path}")

except Exception as e:
    log(f"❌ UMAP 可视化失败：{e}")

log("\n🎉 完成：两层聚类 + 三风格结果（Steady / Conservative / Active）")
LOG.close()

