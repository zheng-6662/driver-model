# # -*- coding: utf-8 -*-
# """
# corr_analysis_v5_1_FULLSET.py
# =========================================================
# v5.1 预测模型 · 全集相关性分析（单文件）
#
# ✔ 分析对象：
#   - 所有 zx|* 与 zx1|* 数值型车辆数据字段（不做先验删减）
#   - 额外派生：steer_rate
#
# ✔ 输入（历史）：
#   - anchor 前 3s（600 点）
#
# ✔ 输出（未来，path-based）：
#   - 30 m，ds=0.5 m → 60 点
#   - 多任务：[steer, yawrate, ay]
#   - 每个通道 3 个指标：RMS / MaxAbs / MaxSlope
#   → 共 9 个目标指标
#
# ✔ 分析：
#   - X-X 冗余（Pearson / Spearman）
#   - X-Y（Pearson / Spearman / Mutual Information）
#   - 特征投票（跨 9 个指标）
#
# ⚠ 本脚本只做“分析”，不决定 Encoder 最终输入
# """
#
# import os
# from glob import glob
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.feature_selection import mutual_info_regression
#
# # ======================================================
# # 基本配置（与你 v5.1 一致）
# # ======================================================
# ROOT = r"F:\数据集处理\data_process\datasetprocess\多模态数据\被试数据集合"
#
# FS = 200
# WIN_SEC = 3.0
# WIN_LEN = int(FS * WIN_SEC)
#
# FUTURE_PATH = 30.0
# DS = 0.5
# FUTURE_LEN = int(FUTURE_PATH / DS)
#
# STRONG_LABELS = [ "strong_active", "extreme_active"]
#
# OUT_DIR = "corr_outputs_v5_1_FULLSET"
# os.makedirs(OUT_DIR, exist_ok=True)
#
# TOPK = 10
# SEED = 2025
# np.random.seed(SEED)
#
# # ======================================================
# # 工具函数
# # ======================================================
# def corr_pearson(a, b):
#     a = np.asarray(a); b = np.asarray(b)
#     m = np.isfinite(a) & np.isfinite(b)
#     if m.sum() < 3: return np.nan
#     return np.corrcoef(a[m], b[m])[0, 1]
#
# def corr_spearman(a, b):
#     a = pd.Series(a).rank().to_numpy()
#     b = pd.Series(b).rank().to_numpy()
#     return corr_pearson(a, b)
#
# def stdz(x, eps=1e-6):
#     return (x - np.nanmean(x)) / (np.nanstd(x) + eps)
#
# # ======================================================
# # 核心：提取事件样本（全集特征）
# # ======================================================
# def extract_events(vehicle_csv):
#     event_csv = vehicle_csv.replace(
#         "_vehicle_aligned_cleaned.csv",
#         "_vehicle_aligned_cleaned_events_v312.csv"
#     ).replace("\\vehicle\\", "\\event\\")
#
#     if not os.path.exists(event_csv):
#         return [], [], None
#
#     df_v = pd.read_csv(vehicle_csv)
#     df_e = pd.read_csv(event_csv)
#
#     # ---------- 全集特征自动发现 ----------
#     feature_cols = [
#         c for c in df_v.columns
#         if (c.startswith("zx|") or c.startswith("zx1|"))
#         and np.issubdtype(df_v[c].dtype, np.number)
#     ]
#
#     if "zx|SteeringWheel" not in df_v.columns:
#         raise RuntimeError("Missing zx|SteeringWheel")
#
#     # 派生 steer_rate
#     df_feat = df_v[feature_cols].copy()
#     steer = df_v["zx|SteeringWheel"].to_numpy()
#     df_feat["steer_rate"] = np.gradient(steer, 1 / FS)
#     feature_cols.append("steer_rate")
#
#     X_all = df_feat.to_numpy(dtype=np.float32)
#     N = len(df_feat)
#
#     # 位置用于 path
#     if not all(c in df_v.columns for c in ["zx|x", "zx|y"]):
#         return [], [], None
#
#     x = df_v["zx|x"].to_numpy()
#     y = df_v["zx|y"].to_numpy()
#     dist = np.concatenate([[0], np.cumsum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))])
#
#     roll_col = "zx|roll"
#     steer_col = "zx|SteeringWheel"
#     yawrate_col = "zx|vyaw"
#     ay_col = "zx|ay"
#
#     if not all(c in df_feat.columns for c in [roll_col, steer_col, yawrate_col, ay_col]):
#         return [], [], None
#
#     idx_roll = feature_cols.index(roll_col)
#     idx_steer = feature_cols.index(steer_col)
#     idx_yawrate = feature_cols.index(yawrate_col)
#     idx_ay = feature_cols.index(ay_col)
#
#     X_list, Y_list = [], []
#
#     df_e = df_e[df_e["event_level"].isin(STRONG_LABELS)]
#
#     for _, ev in df_e.iterrows():
#         i0 = int(ev["start_s"] * FS)
#         i1 = int(ev["end_s"] * FS)
#
#         if i0 < 0 or i0 >= N:
#             continue
#
#         search_end = min(i1 + FS, N)
#         peak = i0 + np.argmax(X_all[i0:search_end, idx_roll])
#
#         if peak - WIN_LEN < 0:
#             continue
#
#         # ---------- 历史 ----------
#         X_hist = X_all[peak - WIN_LEN: peak]
#
#         # ---------- 未来 path ----------
#         d0 = dist[peak]
#         idxs = []
#         for k in range(1, FUTURE_LEN + 1):
#             s = d0 + k * DS
#             j = np.searchsorted(dist, s)
#             if j >= N:
#                 break
#             idxs.append(j)
#
#         if len(idxs) != FUTURE_LEN:
#             continue
#
#         y_future = np.stack([
#             X_all[idxs, idx_steer],
#             X_all[idxs, idx_yawrate],
#             X_all[idxs, idx_ay]
#         ], axis=1)
#
#         X_list.append(X_hist)
#         Y_list.append(y_future)
#
#     return X_list, Y_list, feature_cols
#
# # ======================================================
# # 目标指标（path-based）
# # ======================================================
# def path_metrics(Y):
#     out = {}
#     names = ["steer", "yawrate", "ay"]
#     for i, n in enumerate(names):
#         v = Y[:, i]
#         out[f"{n}_rms"] = np.sqrt(np.mean(v**2))
#         out[f"{n}_maxabs"] = np.max(np.abs(v))
#         out[f"{n}_maxslope"] = np.max(np.abs(np.diff(v) / DS))
#     return out
#
# # ======================================================
# # 主流程
# # ======================================================
# def main():
#     X_all, Y_all = [], []
#     feature_cols = None
#
#     for vf in glob(os.path.join(ROOT, "*", "vehicle", "*_vehicle_aligned_cleaned.csv")):
#         Xs, Ys, feats = extract_events(vf)
#         if feats is None or len(Xs) == 0:
#             continue
#         if feature_cols is None:
#             feature_cols = feats
#         X_all += Xs
#         Y_all += Ys
#
#     K = len(Y_all)
#     print(f"Total events: {K}, total features: {len(feature_cols)}")
#
#     # ---------- X_event ----------
#     X_event = []
#     for X in X_all:
#         X_event.append(np.concatenate([
#             X.mean(axis=0),
#             X.std(axis=0),
#             X[-1]
#         ]))
#     X_event = np.asarray(X_event)
#
#     X_names = (
#         [f"{c}_mean" for c in feature_cols] +
#         [f"{c}_std" for c in feature_cols] +
#         [f"{c}_last" for c in feature_cols]
#     )
#
#     # ---------- Y_metrics ----------
#     Ym = pd.DataFrame([path_metrics(y) for y in Y_all])
#     Ym.to_csv(os.path.join(OUT_DIR, "Y_path_metrics.csv"), index=False)
#
#     # ---------- X-Y ----------
#     rows = []
#     for tgt in Ym.columns:
#         y = stdz(Ym[tgt].to_numpy())
#         Xz = np.stack([stdz(X_event[:, i]) for i in range(X_event.shape[1])], axis=1)
#
#         mi = mutual_info_regression(Xz, y, random_state=SEED)
#
#         for i, fn in enumerate(X_names):
#             rows.append({
#                 "target": tgt,
#                 "feature": fn,
#                 "pearson": corr_pearson(X_event[:, i], y),
#                 "spearman": corr_spearman(X_event[:, i], y),
#                 "mutual_info": mi[i]
#             })
#
#     df = pd.DataFrame(rows)
#     df.to_csv(os.path.join(OUT_DIR, "XY_fullset_stats.csv"), index=False)
#
#     # ---------- 投票 ----------
#     votes = {f: 0 for f in X_names}
#     for tgt in Ym.columns:
#         d = df[df.target == tgt].copy()
#         d["score"] = d["pearson"].abs()
#         top = d.sort_values("score", ascending=False).head(TOPK)
#         for f in top.feature:
#             votes[f] += 1
#
#     vote_df = pd.DataFrame({
#         "feature": list(votes.keys()),
#         "vote": list(votes.values())
#     }).sort_values("vote", ascending=False)
#
#     vote_df.to_csv(os.path.join(OUT_DIR, "feature_vote_FULLSET.csv"), index=False)
#
#     print("DONE. Results in:", OUT_DIR)
#
# if __name__ == "__main__":
#     main()
#
#
# -*- coding: utf-8 -*-
"""
plot_feature_vote_bar.py
========================
把 feature_vote_FULLSET.csv
→ 汇总到“物理变量层级”
→ 画成柱状图（显而易见版）
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

# -----------------------------
# 1. 读取投票结果
# -----------------------------
df = pd.read_csv("feature_vote_FULLSET.csv")

# df.columns = ["feature", "vote"]

# -----------------------------
# 2. 提取“物理变量名”
#    zx|roll_std  -> zx|roll
#    steer_rate   -> steer_rate
# -----------------------------
def get_base_feature(name):
    if name.endswith("_mean") or name.endswith("_std") or name.endswith("_last"):
        return name.rsplit("_", 1)[0]
    return name

df["base_feature"] = df["feature"].apply(get_base_feature)

# -----------------------------
# 3. 按物理变量汇总票数
# -----------------------------
vote_sum = (
    df.groupby("base_feature")["vote"]
    .sum()
    .sort_values(ascending=False)
)

# -----------------------------
# 4. 画柱状图
# -----------------------------
plt.figure(figsize=(10, 5))
vote_sum.plot(kind="bar")

plt.ylabel("Vote count (Top-K frequency)")
plt.xlabel("Physical variable")
plt.title("Feature importance by voting (physical-variable level)")
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()




# # -*- coding: utf-8 -*-
# """
# plot_feature_target_heatmap.py
# ==============================
# 从 XY_fullset_stats.csv
# → 汇总到物理变量层级
# → 画 特征 × 目标 的相关性热力图
# """
#
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
#
# # -----------------------------
# # 1. 读取相关性结果
# # -----------------------------
# df = pd.read_csv("XY_fullset_stats.csv")
#
# # 我们用 |Pearson|，最直观
# df["abs_corr"] = df["pearson"].abs()
#
# # -----------------------------
# # 2. 提取物理变量名
# # -----------------------------
# def get_base_feature(name):
#     if name.endswith("_mean") or name.endswith("_std") or name.endswith("_last"):
#         return name.rsplit("_", 1)[0]
#     return name
#
# df["base_feature"] = df["feature"].apply(get_base_feature)
#
# # -----------------------------
# # 3. 按“物理变量 × 目标指标”汇总
# #    取 max(|corr|) —— 表示“最强作用”
# # -----------------------------
# heatmap_df = (
#     df.groupby(["base_feature", "target"])["abs_corr"]
#       .max()
#       .unstack(fill_value=0)
# )
#
# # -----------------------------
# # 4. 按整体重要性排序（让图更好看）
# # -----------------------------
# heatmap_df["__sum__"] = heatmap_df.sum(axis=1)
# heatmap_df = heatmap_df.sort_values("__sum__", ascending=False)
# heatmap_df = heatmap_df.drop(columns="__sum__")
#
# # -----------------------------
# # 5. 画热力图
# # -----------------------------
# plt.figure(figsize=(12, max(4, 0.35 * len(heatmap_df))))
# plt.imshow(heatmap_df.values, aspect="auto")
#
# plt.colorbar(label="|Pearson correlation|")
#
# plt.yticks(range(len(heatmap_df.index)), heatmap_df.index)
# plt.xticks(range(len(heatmap_df.columns)), heatmap_df.columns, rotation=45, ha="right")
#
# plt.title("Feature–Target Correlation Heatmap (Physical-variable level)")
# plt.tight_layout()
# plt.savefig("feature_target_heatmap.png", dpi=300)
# plt.close()
#
# print("✅ Saved figure: feature_target_heatmap.png")



