# -*- coding: utf-8 -*-
"""
visualize_steer_true_pred_v2.py
================================

功能：
- 重新按和训练时一样的方式构建事件样本
- 加载 Transformer v2 的 checkpoint
- 对「测试集」所有样本做预测
- 画出：
    1）真实值 vs 预测值 的散点图
    2）y = x 参考直线
- 并打印 RMSE / MAE 方便你写论文

注意：
- 需要 future_steer_event_rollpeak_transformer_v2.py
- 需要 model_rollpeak_transformer_v2.pth
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib
matplotlib.use("Agg")  # 强制使用非交互后端，PyCharm不会崩

import os
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# ====== 从 v2 文件中导入需要的内容 ======
from future_steer_event_rollpeak_transformer_v2 import (
    ROOT, FS, WIN_SEC, STRONG_LABELS,
    find_col, load_vehicle_and_events, build_event_samples,
    SteerEventDataset, TimeSeriesTransformer,
    D_MODEL, N_HEAD, NUM_LAYERS, FFN_DIM, DROPOUT
)

# 和训练时保持一致的随机种子
SEED = 2025
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def rebuild_dataset_and_predict():
    """
    1）重新构建所有事件样本（和训练时一样）
    2）加载 checkpoint
    3）对测试集样本做预测
    4）返回 y_true_test, y_pred_test
    """

    # ---------- 1. 按训练方式构造样本 ----------
    all_vehicle_files = sorted(
        glob(os.path.join(ROOT, "*", "vehicle", "*_vehicle_aligned_cleaned.csv"))
    )

    X_pool, y_pool = [], []
    feature_names = None

    print("📂 正在重新构建事件样本（和训练时方式一致）...")

    for vf in all_vehicle_files:
        df_v, df_e = load_vehicle_and_events(vf)
        if df_v is None or df_e is None:
            continue

        X_list, y_list, feats = build_event_samples(df_v, df_e)
        if feats is None:
            continue

        X_pool.extend(X_list)
        y_pool.extend(y_list)
        feature_names = feats

    total = len(y_pool)
    print(f"✅ 共得到事件样本：{total} 个")
    if total == 0:
        raise RuntimeError("没有事件样本，检查 ROOT 和数据路径")

    # ---------- 2. 加载 checkpoint ----------
    print("📦 正在加载模型 checkpoint：model_rollpeak_transformer_v2.pth")
    ckpt = torch.load("model_rollpeak_transformer_v2.pth",
                      map_location=DEVICE,
                      weights_only=False)

    ckpt_feature_names = ckpt["feature_names"]
    feat_mean = ckpt["feat_mean"]
    feat_std = ckpt["feat_std"]
    config = ckpt["config"]

    # 检查特征顺序是否一致
    assert feature_names == ckpt_feature_names, \
        f"当前构造的特征顺序与 checkpoint 不一致：\n现在: {feature_names}\nckpt: {ckpt_feature_names}"

    feat_mean = np.asarray(feat_mean)
    feat_std = np.asarray(feat_std)

    # 标准化
    print("🔧 对所有样本做标准化 (x-mean)/std ...")
    for i in range(len(X_pool)):
        X_pool[i] = (X_pool[i] - feat_mean) / feat_std

    # ---------- 3. 按训练时的方式打乱 + 划分 train/test ----------
    print("🔀 按训练时相同的随机种子打乱并划分 80/20 ...")
    idx = np.arange(total)
    np.random.seed(SEED)          # 确保和训练时一致
    np.random.shuffle(idx)

    X_pool = [X_pool[i] for i in idx]
    y_pool = [y_pool[i] for i in idx]

    n_train = int(total * 0.8)
    X_train = X_pool[:n_train]
    y_train = y_pool[:n_train]
    X_test  = X_pool[n_train:]
    y_test  = y_pool[n_train:]

    print(f"训练集样本数：{len(X_train)}")
    print(f"测试集样本数：{len(X_test)}")

    # ---------- 4. 构造模型并加载权重 ----------
    input_dim = len(feature_names)
    model = TimeSeriesTransformer(
        input_dim=input_dim,
        d_model=config["D_MODEL"],
        nhead=config["N_HEAD"],
        num_layers=config["NUM_LAYERS"],
        dim_feedforward=config["FFN_DIM"],
        dropout=config["DROPOUT"],
        max_len=int(config["WIN_SEC"] * config["FS"])
    ).to(DEVICE)

    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # ---------- 5. 对测试集做预测 ----------
    print("🧮 正在对测试集做预测 ...")
    X_test_arr = np.stack(X_test, axis=0)  # (N_test, T, D)
    y_test_arr = np.array(y_test, dtype=np.float32)

    # 分批推理，避免一次性放入 GPU 过大
    batch_size = 256
    y_pred_list = []

    with torch.no_grad():
        for start in range(0, len(X_test_arr), batch_size):
            end = start + batch_size
            x_batch = torch.tensor(
                X_test_arr[start:end], dtype=torch.float32, device=DEVICE
            )  # (B, T, D)

            pred_batch = model(x_batch)  # (B,)
            y_pred_list.append(pred_batch.cpu().numpy())

    y_pred_arr = np.concatenate(y_pred_list, axis=0)  # (N_test,)

    return y_test_arr, y_pred_arr


def plot_true_vs_pred_degree(y_true_rad, y_pred_rad,
                             out_file="true_vs_pred_scatter_degree.png"):
    """
    可视化：真实 vs 预测 steering（单位：度）
    原始模型输出和标签为 rad（弧度），此处转换为 degree。
    """

    # ---- rad → degree ----
    y_true = y_true_rad * 180 / np.pi
    y_pred = y_pred_rad * 180 / np.pi

    # ---- 评估指标（在 degree 空间下更直观） ----
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    print("\n===== 评估指标（测试集，单位：度）=====")
    print(f"MSE  = {mse:.4f}")
    print(f"RMSE = {rmse:.4f}  (度)")
    print(f"MAE  = {mae:.4f}  (度)")

    # ---- 绘图范围 ----
    y_min = min(y_true.min(), y_pred.min())
    y_max = max(y_true.max(), y_pred.max())
    margin = 0.1 * (y_max - y_min + 1e-6)
    y_min -= margin
    y_max += margin

    # ---- 绘图 ----
    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, s=10, alpha=0.4, label="Samples")

    # y=x 参考线
    plt.plot([y_min, y_max], [y_min, y_max],
             'r--', label="y = x")

    plt.xlabel("True Steering (degree)", fontsize=13)
    plt.ylabel("Predicted Steering (degree)", fontsize=13)
    plt.title("True vs Predicted Steering (Test Set, Degree)", fontsize=15)
    plt.xlim(y_min, y_max)
    plt.ylim(y_min, y_max)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    print(f"📈 已保存角度可视化散点图：{out_file}")



if __name__ == "__main__":
    y_true_test, y_pred_test = rebuild_dataset_and_predict()
    plot_true_vs_pred_degree(
        y_true_test,
        y_pred_test,
        out_file="true_vs_pred_scatter_degree_v2.png"
    )

