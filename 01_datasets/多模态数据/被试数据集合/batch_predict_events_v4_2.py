# # -*- coding: utf-8 -*-
# """
# test_single_event_v4_2_residual.py
# ==================================
#
# 功能：
# - 加载训练好的 residual Seq2Seq transformer（v4.2）
# - 输入：指定车辆文件 + 指定事件编号
# - 输出：该事件 anchor 之后 2s 的方向盘预测 vs 真实图
# """
#
# import matplotlib
# matplotlib.use("TkAgg")
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#
# import torch
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import os
#
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# print("DEVICE =", DEVICE)
#
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# # ============================
# #  1. Seq2Seq Residual 模型结构
# # ============================
# class Seq2SeqTransformer(nn.Module):
#     def __init__(self,
#                  input_dim,
#                  output_len,
#                  d_model=128,
#                  nhead=2,
#                  num_layers_enc=2,
#                  num_layers_dec=2,
#                  dim_feedforward=256,
#                  dropout=0.1,
#                  max_len_enc=400,
#                  max_len_dec=400):
#         super().__init__()
#
#         self.d_model = d_model
#         self.output_len = output_len
#
#         # Encoder
#         self.enc_input_proj = nn.Linear(input_dim, d_model)
#         self.enc_pos_emb = nn.Parameter(torch.zeros(1, max_len_enc, d_model))
#
#         enc_layer = nn.TransformerEncoderLayer(
#             d_model=d_model,
#             nhead=nhead,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout,
#             batch_first=True
#         )
#         self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers_enc)
#
#         # Decoder
#         self.dec_input_proj = nn.Linear(1, d_model)
#         self.dec_pos_emb   = nn.Parameter(torch.zeros(1, max_len_dec, d_model))
#
#         dec_layer = nn.TransformerDecoderLayer(
#             d_model=d_model,
#             nhead=nhead,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout,
#             batch_first=True
#         )
#         self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers_dec)
#
#         self.out_proj = nn.Linear(d_model, 1)
#         self.dropout  = nn.Dropout(dropout)
#
#     def _generate_mask(self, sz, device):
#         mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
#         return mask.masked_fill(mask == 1, float('-inf'))
#
#     def forward(self, src, tgt_in):
#         B, T_in, _ = src.shape
#         B2, T_out  = tgt_in.shape
#         assert B == B2
#
#         # Encoder
#         h_src = self.enc_input_proj(src)
#         h_src = h_src + self.enc_pos_emb[:, :T_in, :]
#         memory = self.encoder(self.dropout(h_src))
#
#         # Decoder
#         tgt_feat = tgt_in.unsqueeze(-1)
#         h_tgt = self.dec_input_proj(tgt_feat)
#         h_tgt = h_tgt + self.dec_pos_emb[:, :T_out, :]
#         tgt_mask = self._generate_mask(T_out, src.device)
#
#         out = self.decoder(h_tgt, memory, tgt_mask=tgt_mask)
#         y_hat = self.out_proj(out).squeeze(-1)
#
#         return y_hat
#
#
# # ============================
# #  2. 提取事件样本（Residual 版本）
# # ============================
# def extract_event_sample(vehicle_file, event_file, event_idx,
#                          feature_names, feat_mean, feat_std,
#                          res_mean, res_std,
#                          FS=200, WIN_SEC=2.0, FUTURE_SEC=2.0):
#
#     df_v = pd.read_csv(vehicle_file)
#     df_e = pd.read_csv(event_file)
#
#     df_e = df_e[df_e["event_level"].isin(["medium_active","strong_active","extreme_active"])]
#     df_e = df_e.reset_index(drop=True)
#     print("筛选后事件列表：")
#     print(df_e[["event_index", "event_level"]])
#
#     if event_idx >= len(df_e):
#         raise ValueError(f"事件编号 {event_idx} 超出范围，共 {len(df_e)} 个事件")
#
#     ev = df_e.iloc[event_idx]
#     print(f"▶ 当前事件编号：{event_idx}, 事件强度：{ev['event_level']}")
#
#     t0, t1 = float(ev["start_s"]), float(ev["end_s"])
#     i0 = int(t0 * FS)
#     i1 = int(t1 * FS)
#
#     # ------- 构造特征 -------
#     df_feat = df_v.copy()
#
#     # LTR
#     if "zx|ay" in df_v.columns and "LTR_est" not in df_feat.columns:
#         df_feat["LTR_est"] = df_v["zx|ay"] * 0.11243
#
#     # steer_rate
#     if "zx|SteeringWheel" in df_v.columns:
#         steer = df_v["zx|SteeringWheel"].values
#         dt = 1.0 / FS
#         df_feat["steer_rate"] = np.gradient(steer, dt)
#
#     df_feat = df_feat[feature_names]
#     X_all = df_feat.values.astype(np.float32)
#
#     roll_idx  = feature_names.index("zx|roll")
#     steer_idx = feature_names.index("zx|SteeringWheel")
#
#     # ------- 找 anchor -------
#     roll_seg = X_all[i0:i1, roll_idx]
#     peak_idx = i0 + int(np.argmax(roll_seg))
#
#     WIN_LEN = int(WIN_SEC * FS)
#     FUTURE_LEN = int(FUTURE_SEC * FS)
#
#     if peak_idx - WIN_LEN < 0 or peak_idx + FUTURE_LEN >= len(df_v):
#         raise RuntimeError("事件边界不够，无法截取窗口")
#
#     x_win = X_all[peak_idx - WIN_LEN: peak_idx]
#     anchor_steer = X_all[peak_idx, steer_idx]
#     y_true = X_all[peak_idx + 1 : peak_idx + 1 + FUTURE_LEN, steer_idx]
#
#     # residual
#     residual = y_true - anchor_steer
#     res_norm = (residual - res_mean) / res_std
#
#     # decoder 输入
#     tgt_in = np.zeros_like(res_norm)
#     tgt_in[1:] = res_norm[:-1]
#
#     # 标准化 src
#     x_norm = (x_win - feat_mean) / feat_std
#
#     return x_norm, tgt_in, y_true, anchor_steer
#
#
# # ============================
# #  3. 可视化
# # ============================
# def predict_and_plot(model, x_norm, tgt_in, y_true, anchor, FUTURE_SEC=2.0):
#
#     model.eval()
#
#     x_tensor = torch.from_numpy(x_norm).unsqueeze(0).float().to(DEVICE)
#     t_in     = torch.from_numpy(tgt_in).unsqueeze(0).float().to(DEVICE)
#
#     with torch.no_grad():
#         res_hat_norm = model(x_tensor, t_in)
#         res_hat = res_hat_norm.cpu().numpy()[0] * res_std + res_mean
#         y_hat = res_hat + anchor
#
#     t = np.linspace(0, FUTURE_SEC, len(y_hat), endpoint=False)
#
#     plt.figure(figsize=(11,4))
#     plt.plot(t, y_true, label="GT Steering", linewidth=1.2)
#     plt.plot(t, y_hat, "--", label="Pred Steering", linewidth=1.2)
#     plt.xlabel("Time (s)")
#     plt.ylabel("Steering angle")
#     plt.title("Future 2s Steering Prediction (v4.2 Residual)")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
#
#
# # ============================
# #  4. 主函数（你只改这里）
# # ============================
# if __name__ == "__main__":
#
#     # ① 使用 v4.2 residual 模型
#     ckpt_path = r"F:\数据集处理\data_process\datasetprocess\多模态数据\model_rollpeak_transformer_v4_2_seq2seq_residual.pth"
#
#     # ② 指定车辆文件 & 事件文件
#     vehicle_file = r"F:\数据集处理\data_process\datasetprocess\多模态数据\被试数据集合\byx\vehicle\Entity_Recording_2025_09_28_17_05_51_vehicle_aligned_cleaned.csv"
#     event_file   = r"F:\数据集处理\data_process\datasetprocess\多模态数据\被试数据集合\byx\event\Entity_Recording_2025_09_28_17_05_51_multimodal_event_features_v1.csv"
#
#     # ③ 指定你想看的事件编号
#     event_idx =10# 👈 修改这里即可测试不同事件
#
#     # ======================
#     print("\n加载模型...")
#     import torch.serialization
#
#     # 允许 numpy 的 reconstruct（你的 checkpoint 里确实包含 numpy 对象）
#     torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
#
#     ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
#
#     feature_names = ckpt["feature_names"]
#     feat_mean = ckpt["feat_mean"]
#     feat_std  = ckpt["feat_std"]
#     res_mean  = ckpt["res_mean"]
#     res_std   = ckpt["res_std"]
#     config    = ckpt["config"]
#
#     # 构建模型
#     model = Seq2SeqTransformer(
#         input_dim=len(feature_names),
#         output_len=config["FUTURE_LEN"],
#         d_model=config["D_MODEL"],
#         nhead=config["N_HEAD"],
#         num_layers_enc=config["NUM_LAYERS_ENC"],
#         num_layers_dec=config["NUM_LAYERS_DEC"],
#         dim_feedforward=config["FFN_DIM"],
#         dropout=config["DROPOUT"],
#         max_len_enc=config["WIN_LEN"],
#         max_len_dec=config["FUTURE_LEN"]
#     ).to(DEVICE)
#
#     model.load_state_dict(ckpt["state_dict"])
#
#     print("提取事件样本...")
#     x_norm, tgt_in, y_true, anchor = extract_event_sample(
#         vehicle_file, event_file, event_idx,
#         feature_names, feat_mean, feat_std,
#         res_mean, res_std,
#         FS=config["FS"],
#         WIN_SEC=config["WIN_SEC"],
#         FUTURE_SEC=config["FUTURE_SEC"]
#     )
#
#     print("预测并绘图...")
#     predict_and_plot(model, x_norm, tgt_in, y_true, anchor, FUTURE_SEC=config["FUTURE_SEC"])
#
# -*- coding: utf-8 -*-
"""
test_autoregressive_v4_2_residual.py
====================================

严格自回归测试（NO teacher forcing）
- 每一步 residual_norm_hat[t] 作为下一步 decoder 输入
- 能真实反映模型的未来轨迹预测能力
"""

import matplotlib
matplotlib.use("TkAgg")
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE =", DEVICE)

import torch.nn as nn
import torch.nn.functional as F


# ===============================================
#  1. 模型结构（和你原来完全一致，不改）
# ===============================================
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 input_dim,
                 output_len,
                 d_model=128,
                 nhead=2,
                 num_layers_enc=2,
                 num_layers_dec=2,
                 dim_feedforward=256,
                 dropout=0.1,
                 max_len_enc=400,
                 max_len_dec=400):
        super().__init__()

        self.d_model = d_model
        self.output_len = output_len

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
        self.dec_input_proj = nn.Linear(1, d_model)
        self.dec_pos_emb   = nn.Parameter(torch.zeros(1, max_len_dec, d_model))

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers_dec)

        self.out_proj = nn.Linear(d_model, 1)
        self.dropout  = nn.Dropout(dropout)

    def _generate_mask(self, sz, device):
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))

    def forward(self, src, tgt_in):
        """这里仍然是 teacher forcing 的 forward（不修改）
        自回归版本我们在外部逐步手动生成 tgt_in
        """
        B, T_in, _ = src.shape
        B2, T_out  = tgt_in.shape
        assert B == B2

        # ----- Encoder -----
        h_src = self.enc_input_proj(src)
        h_src = h_src + self.enc_pos_emb[:, :T_in, :]
        memory = self.encoder(self.dropout(h_src))

        # ----- Decoder -----
        tgt_feat = tgt_in.unsqueeze(-1)
        h_tgt = self.dec_input_proj(tgt_feat)
        h_tgt = h_tgt + self.dec_pos_emb[:, :T_out, :]

        tgt_mask = self._generate_mask(T_out, src.device)
        out = self.decoder(h_tgt, memory, tgt_mask=tgt_mask)

        y_hat = self.out_proj(out).squeeze(-1)  # (B, T_out)
        return y_hat


# ===============================================
#  2. 自回归预测函数（核心）
# ===============================================
def autoregressive_predict(model, src_norm, res_mean, res_std, anchor, FUTURE_LEN):

    model.eval()

    # (1, 400, D)
    src_tensor = torch.from_numpy(src_norm).unsqueeze(0).float().to(DEVICE)

    # 用来保存每个时刻的 residual_norm_hat
    res_hat_norm = np.zeros(FUTURE_LEN, dtype=np.float32)

    # 初始化 decoder 输入：形状必须为 (1, 1, 1)
    # 第一个 token = 0
    tgt_in = torch.zeros(1, 1, 1, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():

        # -----------------
        #    Encoder
        # -----------------
        h_src = model.enc_input_proj(src_tensor)
        h_src = h_src + model.enc_pos_emb[:, :src_tensor.size(1), :]
        memory = model.encoder(model.dropout(h_src))  # (1, 400, 128)

        # -----------------
        # Autoregressive loop
        # -----------------
        for t in range(FUTURE_LEN):

            T_current = tgt_in.size(1)

            # Decoder 输入处理
            h_tgt = model.dec_input_proj(tgt_in)                    # (1, T_current, 128)
            h_tgt = h_tgt + model.dec_pos_emb[:, :T_current, :]    # pos emb

            tgt_mask = model._generate_mask(T_current, DEVICE)     # (T_current, T_current)

            out = model.decoder(h_tgt, memory, tgt_mask=tgt_mask)  # (1, T_current, 128)

            # 取最后一个时间点的预测
            y_hat_step = model.out_proj(out[:, -1:, :])  # (1, 1, 1)

            # 保存 residual_norm_hat
            res_hat_norm[t] = y_hat_step.item()

            # 将预测值 append 到 decoder 输入 (保持三维!)
            tgt_in = torch.cat([tgt_in, y_hat_step], dim=1)  # 变成 (1, T_current+1, 1)

    # ----------- 反标准化 -----------
    res_hat = res_hat_norm * res_std + res_mean
    y_hat = res_hat + anchor

    return y_hat, res_hat_norm


# ===============================================
#  3. 事件提取（和你原来完全一致）
# ===============================================
def extract_event_sample(vehicle_file, event_file, event_idx,
                         feature_names, feat_mean, feat_std,
                         res_mean, res_std,
                         FS=200, WIN_SEC=2.0, FUTURE_SEC=2.0):

    df_v = pd.read_csv(vehicle_file)
    df_e = pd.read_csv(event_file)

    df_e = df_e[df_e["event_level"].isin(["medium_active","strong_active","extreme_active"])]
    df_e = df_e.reset_index(drop=True)

    ev = df_e.iloc[event_idx]
    t0, t1 = float(ev["start_s"]), float(ev["end_s"])
    i0, i1 = int(t0*FS), int(t1*FS)

    df_feat = df_v.copy()

    if "zx|ay" in df_v.columns and "LTR_est" not in df_feat.columns:
        df_feat["LTR_est"] = df_v["zx|ay"] * 0.11243

    if "zx|SteeringWheel" in df_v.columns:
        steer = df_v["zx|SteeringWheel"].values
        dt = 1.0 / FS
        df_feat["steer_rate"] = np.gradient(steer, dt)

    df_feat = df_feat[feature_names]
    X_all = df_feat.values.astype(np.float32)

    roll_idx  = feature_names.index("zx|roll")
    steer_idx = feature_names.index("zx|SteeringWheel")

    roll_seg = X_all[i0:i1, roll_idx]
    peak_idx = i0 + int(np.argmax(roll_seg))

    WIN_LEN = int(WIN_SEC * FS)
    FUTURE_LEN = int(FUTURE_SEC * FS)

    x_win = X_all[peak_idx - WIN_LEN : peak_idx]
    anchor_steer = X_all[peak_idx, steer_idx]
    y_true = X_all[peak_idx+1 : peak_idx+1+FUTURE_LEN, steer_idx]

    # 标准化 encoder 输入
    x_norm = (x_win - feat_mean) / feat_std

    return x_norm, y_true, anchor_steer


# ===============================================
#  4. 可视化
# ===============================================
def plot_result(y_true, y_hat, FUTURE_SEC=2.0):
    t = np.linspace(0, FUTURE_SEC, len(y_hat), endpoint=False)
    plt.figure(figsize=(11,4))
    plt.plot(t, y_true, label="GT Steering")
    plt.plot(t, y_hat, "--", label="Pred Steering (Autoregressive)")
    plt.xlabel("Time (s)")
    plt.ylabel("Steering angle")
    plt.title("Strict Autoregressive Future 2s Steering Prediction")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ===============================================
#  5. 主程序
# ===============================================
if __name__ == "__main__":

    ckpt_path = r"F:\数据集处理\data_process\datasetprocess\多模态数据\model_rollpeak_transformer_v4_2_seq2seq_residual.pth"

    vehicle_file = r"F:\数据集处理\data_process\datasetprocess\多模态数据\被试数据集合\byx\vehicle\Entity_Recording_2025_09_28_17_05_51_vehicle_aligned_cleaned.csv"
    event_file   = r"F:\数据集处理\data_process\datasetprocess\多模态数据\被试数据集合\byx\event\Entity_Recording_2025_09_28_17_05_51_multimodal_event_features_v1.csv"

    event_idx = 8

    print("加载模型...")

    import torch.serialization
    torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    feature_names = ckpt["feature_names"]
    feat_mean = ckpt["feat_mean"]
    feat_std  = ckpt["feat_std"]
    res_mean  = ckpt["res_mean"]
    res_std   = ckpt["res_std"]
    config    = ckpt["config"]

    model = Seq2SeqTransformer(
        input_dim=len(feature_names),
        output_len=config["FUTURE_LEN"],
        d_model=config["D_MODEL"],
        nhead=config["N_HEAD"],
        num_layers_enc=config["NUM_LAYERS_ENC"],
        num_layers_dec=config["NUM_LAYERS_DEC"],
        dim_feedforward=config["FFN_DIM"],
        dropout=config["DROPOUT"],
        max_len_enc=config["WIN_LEN"],
        max_len_dec=config["FUTURE_LEN"]
    ).to(DEVICE)

    model.load_state_dict(ckpt["state_dict"])

    print("提取事件样本...")
    x_norm, y_true, anchor = extract_event_sample(
        vehicle_file, event_file, event_idx,
        feature_names, feat_mean, feat_std,
        res_mean, res_std,
        FS=config["FS"],
        WIN_SEC=config["WIN_SEC"],
        FUTURE_SEC=config["FUTURE_SEC"]
    )

    print("严格自回归预测...")
    y_hat, _ = autoregressive_predict(
        model, x_norm, res_mean, res_std,
        anchor, FUTURE_LEN=config["FUTURE_LEN"]
    )

    plot_result(y_true, y_hat, config["FUTURE_SEC"])
