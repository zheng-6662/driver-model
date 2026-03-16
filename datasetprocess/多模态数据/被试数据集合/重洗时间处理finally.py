# # -*- coding: utf-8 -*-
# """
# 事件检测 V3.5（你的项目最终优化版）
# - 自动 StorageTime → t_s
# - 自动识别关键列名（roll/steer/yaw_rate/vx/z/ay）
# - 自动计算 LTR_est（若不存在）
# - z 模式检测 offroad（适配下坡 + 斜坡 + 掉路肩）
# - 无绘图（极速）
# - 去前10s噪声
# - weak 噪声剔除
# - episode + primary 事件识别
# - 批处理所有被试
# """
#
# import os
# import numpy as np
# import pandas as pd
# from glob import glob
#
#
# # =======================
# # 车辆参数（你给我的）
# # =======================
# G = 9.81
# H_COG = 1.2
# TRACK = 2.176
# LTR_COEFF = 2 * H_COG / (TRACK * G)     # ≈0.11243
#
#
# # =======================
# # 事件参数
# # =======================
# FS = 200
# MIN_EP_GAP = 5.0
# Z_JUMP = 0.25
# SLOPE_JUMP = 0.15
#
#
# # =======================
# # 自动查列名
# # =======================
# def find_col(df, keys):
#     for c in df.columns:
#         cc = c.lower()
#         if any(k in cc for k in keys):
#             return c
#     return None
#
#
# # =======================
# # 自动生成 t_s
# # =======================
# def get_ts(df):
#     if "t_s" in df.columns:
#         return df["t_s"].astype(float).values
#
#     if "StorageTime" in df.columns:
#         t_raw = pd.to_datetime(df["StorageTime"], errors="coerce")
#         if not t_raw.isna().all():
#             return (t_raw - t_raw.iloc[0]).dt.total_seconds().values
#
#     print("⚠ 使用第一列作为时间")
#     return pd.to_numeric(df.iloc[:, 0], errors="coerce").values
#
#
# # =======================
# # 自动生成 LTR_est
# # =======================
# def get_ltr(df, roll):
#     # 1) 有 LTR_est → 用现成的
#     if "LTR_est" in df.columns:
#         return df["LTR_est"].astype(float).values
#
#     # 2) 有 ay → 用真实 LTR 计算公式
#     ay_col = find_col(df, ["ay"])
#     if ay_col:
#         ay = df[ay_col].astype(float).values
#         return LTR_COEFF * ay
#
#     # 3) Fallback → sin(roll)
#     print("⚠ 未找到 ay，LTR_est = sin(roll)")
#     return np.sin(roll)
#
#
# # =======================
# # weak 噪声判定
# # =======================
# def is_very_weak(roll_peak, ltr_peak, steer_range, steer_rate_peak,
#                  yaw_rate_peak, roll_rate_peak, duration):
#     return (
#         abs(roll_peak) < 0.05 and
#         abs(ltr_peak) < 0.20 and
#         steer_range < 0.3 and
#         steer_rate_peak < 5 and
#         abs(yaw_rate_peak) < 0.2 and
#         abs(roll_rate_peak) < 0.1 and
#         duration < 0.4
#     )
#
#
# # =======================
# # z 模式 offroad 检测（V3.4 核心）
# # =======================
# def detect_offroad(z_seg, dz_seg):
#     if np.nanmax(z_seg) - np.nanmin(z_seg) > Z_JUMP:
#         return True
#     if np.nanmax(np.abs(dz_seg)) > SLOPE_JUMP:
#         return True
#     return False
#
#
# # =======================
# # 基础事件触发：方向盘速率 > 4
# # =======================
# def detect_triggers(t, steer_rate):
#     idx = np.where(np.abs(steer_rate) > 4)[0]
#     events = []
#
#     for i in idx:
#         s = t[i] - 0.5
#         e = t[i] + 0.5
#         s = max(0, s)
#         if len(events) and s <= events[-1]["end_s"]:
#             continue
#         events.append({"start_s": s, "end_s": e})
#     return events
#
#
# # =======================
# # 事件分类
# # =======================
# def classify_level(steer_range, steer_rate_peak, roll_peak, ltr_peak):
#     if steer_rate_peak >= 25 or abs(ltr_peak) >= 0.75:
#         return "extreme_active"
#     if steer_rate_peak >= 12 or abs(ltr_peak) >= 0.60:
#         return "strong_active"
#     if steer_range >= 0.9 and steer_rate_peak >= 7:
#         return "medium_active"
#     return "weak"
#
#
# # =======================
# # 主函数：处理一个 CSV
# # =======================
# def process_vehicle(csv_path):
#     df = pd.read_csv(csv_path)
#
#     # 时间
#     t = get_ts(df)
#     dt = 1 / FS
#
#     # 基础列自动识别
#     roll_col = find_col(df, ["roll"])
#     steer_col = find_col(df, ["steering"])
#     yaw_col = find_col(df, ["vyaw", "yaw"])
#     vx_col = find_col(df, ["vx"])
#     z_col = find_col(df, ["|z"])
#
#     roll = df[roll_col].astype(float).values
#     steer = df[steer_col].astype(float).values if steer_col else np.zeros_like(roll)
#     yaw_rate = df[yaw_col].astype(float).values if yaw_col else np.zeros_like(roll)
#     vx = df[vx_col].astype(float).values if vx_col else np.zeros_like(roll)
#     z = df[z_col].astype(float).values if z_col else np.zeros_like(roll)
#
#     # LTR（自动）
#     ltr = get_ltr(df, roll)
#
#     # 衍生量
#     steer_rate = np.append(np.diff(steer) / dt, 0)
#     roll_rate = np.append(np.diff(roll) / dt, 0)
#     dz = np.append(np.diff(z), 0)
#
#     # 初步事件
#     events = detect_triggers(t, steer_rate)
#
#     out = []
#     for ev in events:
#         s, e = ev["start_s"], ev["end_s"]
#         if e <= 10:
#             continue
#
#         mask = (t >= s) & (t <= e)
#         if not np.any(mask):
#             continue
#
#         seg_roll = roll[mask]
#         seg_ltr = ltr[mask]
#         seg_steer = steer[mask]
#         seg_steer_rate = steer_rate[mask]
#         seg_yaw_rate = yaw_rate[mask]
#         seg_roll_rate = roll_rate[mask]
#         seg_vx = vx[mask]
#         seg_z = z[mask]
#         seg_dz = dz[mask]
#
#         roll_peak = np.nanmax(np.abs(seg_roll))
#         ltr_peak = np.nanmax(np.abs(seg_ltr))
#         steer_range = np.nanmax(seg_steer) - np.nanmin(seg_steer)
#         steer_rate_peak = np.nanmax(np.abs(seg_steer_rate))
#         yaw_rate_peak = np.nanmax(np.abs(seg_yaw_rate))
#         roll_rate_peak = np.nanmax(np.abs(seg_roll_rate))
#         z_range = np.nanmax(seg_z) - np.nanmin(seg_z)
#         dz_max = np.nanmax(np.abs(seg_dz))
#         vx_mean = np.nanmean(seg_vx)
#
#         duration = e - s
#
#         if is_very_weak(roll_peak, ltr_peak, steer_range, steer_rate_peak,
#                          yaw_rate_peak, roll_rate_peak, duration):
#             continue
#
#         level = classify_level(steer_range, steer_rate_peak, roll_peak, ltr_peak)
#         offflag = detect_offroad(seg_z, seg_dz)
#
#         out.append({
#             "start_s": s,
#             "end_s": e,
#             "duration": duration,
#             "roll_peak": roll_peak,
#             "LTR_peak": ltr_peak,
#             "steer_range": steer_range,
#             "steer_rate_peak": steer_rate_peak,
#             "yaw_rate_peak": yaw_rate_peak,
#             "roll_rate_peak": roll_rate_peak,
#             "vx_mean": vx_mean,
#             "z_range": z_range,
#             "dz_max": dz_max,
#             "event_level": level,
#             "offroad": offflag
#         })
#
#     if len(out) == 0:
#         return None
#
#     df_out = pd.DataFrame(out)
#
#     # episode + primary
#     df_out = df_out.sort_values("start_s").reset_index(drop=True)
#     df_out["episode_id"] = -1
#     df_out["phase_type"] = "secondary"
#
#     ep = 0
#     last_end = -999
#
#     for i, row in df_out.iterrows():
#         if row["start_s"] - last_end > MIN_EP_GAP:
#             ep += 1
#         df_out.at[i, "episode_id"] = ep
#         last_end = row["end_s"]
#
#     # primary = 每段 LTR 最大
#     for ep, sub in df_out.groupby("episode_id"):
#         idx = sub["LTR_peak"].idxmax()
#         df_out.at[idx, "phase_type"] = "primary"
#
#     return df_out
#
#
# # =======================
# # 批处理
# # =======================
# ROOT = r"F:\数据集处理\data_process\datasetprocess\数据预处理\车辆数据处理\正式车辆数据处理"
#
# print("\n=== V3.5 事件检测开始 ===\n")
#
# for subj in os.listdir(ROOT):
#     subj_dir = os.path.join(ROOT, subj)
#     if not os.path.isdir(subj_dir):
#         continue
#
#     print(f"\n--- 被试：{subj} ---")
#
#     files = glob(os.path.join(subj_dir, "*aligned_cleaned*.csv"))
#     for f in files:
#         df_ev = process_vehicle(f)
#         if df_ev is None:
#             print("  ⚠ 无事件")
#             continue
#
#         out_csv = f.replace(".csv", "_events_v35.csv")
#         df_ev.to_csv(out_csv, index=False, encoding="utf-8-sig")
#         print("  ✔ 输出事件：", out_csv)
#
# print("\n=== 全部完成 ===")

# -*- coding: utf-8 -*-
# """
# 事件检测 V3.6（最终完整版）
# ===========================
#
# 特点：
# - 自动 StorageTime → t_s
# - 自动识别 roll/steer/yaw_rate/vx/z 列
# - 自动计算 LTR_est（若不存在）
# - 弱噪声过滤 very_weak
# - episode + primary 系统
# - 高级 offroad 判定：加入 duration < 3.0s → correction，不算 offroad
# - 批处理所有被试
#
# 这是你目前项目的最终版本，可直接用于事件表生成。
# """
#
# import os
# import numpy as np
# import pandas as pd
# from glob import glob
#
#
# # =======================
# # 车辆物理参数
# # =======================
# G = 9.81
# H_COG = 1.2
# TRACK = 2.176
# LTR_COEFF = 2 * H_COG / (TRACK * G)     # ≈0.11243
#
#
# # =======================
# # 参数
# # =======================
# FS = 200
# MIN_EP_GAP = 5.0
# Z_JUMP = 0.25
# SLOPE_JUMP = 0.15
# OFFROAD_MIN_DURATION = 3.0    # ⭐ 你要求的修改：短于3秒不算offroad
#
#
# # =======================
# # 列名自动查找
# # =======================
# def find_col(df, keys):
#     for c in df.columns:
#         cc = c.lower()
#         if any(k in cc for k in keys):
#             return c
#     return None
#
#
# # =======================
# # 自动生成 t_s
# # =======================
# def get_ts(df):
#     if "t_s" in df.columns:
#         return df["t_s"].astype(float).values
#
#     if "StorageTime" in df.columns:
#         t_raw = pd.to_datetime(df["StorageTime"], errors="coerce")
#         if not t_raw.isna().all():
#             return (t_raw - t_raw.iloc[0]).dt.total_seconds().values
#
#     print("⚠ 使用第一列作为时间")
#     return pd.to_numeric(df.iloc[:, 0], errors="coerce").values
#
#
# # =======================
# # 自动生成 LTR
# # =======================
# def get_ltr(df, roll):
#     if "LTR_est" in df.columns:
#         return df["LTR_est"].astype(float).values
#
#     ay_col = find_col(df, ["ay"])
#     if ay_col:
#         ay = df[ay_col].astype(float).values
#         return LTR_COEFF * ay
#
#     print("⚠ 未找到 ay，LTR = sin(roll)")
#     return np.sin(roll)
#
#
# # =======================
# # very weak 判断
# # =======================
# def is_very_weak(roll_peak, ltr_peak, steer_range, steer_rate_peak,
#                  yaw_rate_peak, roll_rate_peak, duration):
#     return (
#         abs(roll_peak) < 0.05 and
#         abs(ltr_peak) < 0.20 and
#         steer_range < 0.3 and
#         steer_rate_peak < 5 and
#         abs(yaw_rate_peak) < 0.2 and
#         abs(roll_rate_peak) < 0.1 and
#         duration < 0.4
#     )
#
#
# # =======================
# # 新版 Offroad 检测（你要求的最终逻辑）
# # =======================
# def detect_offroad(z_seg, dz_seg, duration):
#     """
#     满足 z / dz 的特征 → base_flag = True
#     但：
#        若 duration < 3.0s → correction，不当作 offroad
#        若 duration >= 3.0s → 真实 offroad
#     """
#
#     z_jump = np.nanmax(z_seg) - np.nanmin(z_seg)
#     dz_max = np.nanmax(np.abs(dz_seg))
#
#     # 原始 offroad 形状判定
#     base_flag = (z_jump > Z_JUMP) or (dz_max > SLOPE_JUMP)
#     if not base_flag:
#         return False
#
#     # ⭐ 最关键修改：短暂（<3s）为 correction，不算 offroad
#     if duration < OFFROAD_MIN_DURATION:
#         return False
#
#     return True


# =======================
# 基础触发（steer_rate）
# # =======================
# def detect_triggers(t, steer_rate):
#     idx = np.where(np.abs(steer_rate) > 4)[0]
#     events = []
#
#     for i in idx:
#         s = t[i] - 0.5
#         e = t[i] + 0.5
#         s = max(0, s)
#         if len(events) and s <= events[-1]["end_s"]:
#             continue
#         events.append({"start_s": s, "end_s": e})
#     return events
#
#
# # =======================
# # 分类逻辑
# # =======================
# def classify_level(steer_range, steer_rate_peak, roll_peak, ltr_peak):
#     if steer_rate_peak >= 25 or abs(ltr_peak) >= 0.75:
#         return "extreme_active"
#     if steer_rate_peak >= 12 or abs(ltr_peak) >= 0.60:
#         return "strong_active"
#     if steer_range >= 0.9 and steer_rate_peak >= 7:
#         return "medium_active"
#     return "weak"
#
#
# # =======================
# # 主流程
# # =======================
# def process_vehicle(csv_path):
#     df = pd.read_csv(csv_path)
#
#     t = get_ts(df)
#     dt = 1 / FS
#
#     roll_col = find_col(df, ["roll"])
#     steer_col = find_col(df, ["steering"])
#     yaw_col = find_col(df, ["yaw", "vyaw"])
#     vx_col = find_col(df, ["vx"])
#     z_col = find_col(df, ["|z"])
#
#     roll = df[roll_col].astype(float).values
#     steer = df[steer_col].astype(float).values if steer_col else np.zeros_like(roll)
#     yaw_rate = df[yaw_col].astype(float).values if yaw_col else np.zeros_like(roll)
#     vx = df[vx_col].astype(float).values if vx_col else np.zeros_like(roll)
#     z = df[z_col].astype(float).values if z_col else np.zeros_like(roll)
#
#     ltr = get_ltr(df, roll)
#
#     steer_rate = np.append(np.diff(steer) / dt, 0)
#     roll_rate = np.append(np.diff(roll) / dt, 0)
#     dz = np.append(np.diff(z), 0)
#
#     events = detect_triggers(t, steer_rate)
#
#     out = []
#
#     for ev in events:
#         s, e = ev["start_s"], ev["end_s"]
#         if e <= 10:
#             continue
#
#         mask = (t >= s) & (t <= e)
#         if not np.any(mask):
#             continue
#
#         seg_roll = roll[mask]
#         seg_ltr = ltr[mask]
#         seg_steer = steer[mask]
#         seg_steer_rate = steer_rate[mask]
#         seg_yaw_rate = yaw_rate[mask]
#         seg_roll_rate = roll_rate[mask]
#         seg_vx = vx[mask]
#         seg_z = z[mask]
#         seg_dz = dz[mask]
#
#         roll_peak = np.nanmax(np.abs(seg_roll))
#         ltr_peak = np.nanmax(np.abs(seg_ltr))
#         steer_range = np.nanmax(seg_steer) - np.nanmin(seg_steer)
#         steer_rate_peak = np.nanmax(np.abs(seg_steer_rate))
#         yaw_rate_peak = np.nanmax(np.abs(seg_yaw_rate))
#         roll_rate_peak = np.nanmax(np.abs(seg_roll_rate))
#         z_range = np.nanmax(seg_z) - np.nanmin(seg_z)
#         dz_max = np.nanmax(np.abs(seg_dz))
#         vx_mean = np.nanmean(seg_vx)
#         duration = e - s
#
#         if is_very_weak(roll_peak, ltr_peak, steer_range,
#                         steer_rate_peak, yaw_rate_peak, roll_rate_peak, duration):
#             continue
#
#         level = classify_level(steer_range, steer_rate_peak, roll_peak, ltr_peak)
#         offflag = detect_offroad(seg_z, seg_dz, duration)
#
#         out.append({
#             "start_s": s,
#             "end_s": e,
#             "duration": duration,
#             "roll_peak": roll_peak,
#             "LTR_peak": ltr_peak,
#             "steer_range": steer_range,
#             "steer_rate_peak": steer_rate_peak,
#             "yaw_rate_peak": yaw_rate_peak,
#             "roll_rate_peak": roll_rate_peak,
#             "vx_mean": vx_mean,
#             "z_range": z_range,
#             "dz_max": dz_max,
#             "event_level": level,
#             "offroad": offflag
#         })
#
#     if len(out) == 0:
#         return None
#
#     df_out = pd.DataFrame(out)
#
#     df_out = df_out.sort_values("start_s").reset_index(drop=True)
#     df_out["episode_id"] = -1
#     df_out["phase_type"] = "secondary"
#
#     ep = 0
#     last_end = -999
#
#     for i, row in df_out.iterrows():
#         if row["start_s"] - last_end > MIN_EP_GAP:
#             ep += 1
#         df_out.at[i, "episode_id"] = ep
#         last_end = row["end_s"]
#
#     # primary = LTR 最大
#     for ep, sub in df_out.groupby("episode_id"):
#         idx = sub["LTR_peak"].idxmax()
#         df_out.at[idx, "phase_type"] = "primary"
#
#     return df_out
#
#
# # =======================
# # 批处理入口
# # =======================
# ROOT = r"F:\数据集处理\data_process\datasetprocess\数据预处理\车辆数据处理\正式车辆数据处理"
#
# print("\n=== V3.6 事件检测开始（最终版） ===\n")
#
# for subj in os.listdir(ROOT):
#     subj_dir = os.path.join(ROOT, subj)
#     if not os.path.isdir(subj_dir):
#         continue
#
#     print(f"\n--- 被试：{subj} ---")
#
#     files = glob(os.path.join(subj_dir, "*aligned_cleaned*.csv"))
#     for f in files:
#         df_ev = process_vehicle(f)
#         if df_ev is None:
#             print("  ⚠ 无事件")
#             continue
#
#         out_csv = f.replace(".csv", "_events_v36.csv")
#         df_ev.to_csv(out_csv, index=False, encoding="utf-8-sig")
#         print("  ✔ 输出事件：", out_csv)
#
# print("\n=== 全部完成（V3.6） ===")




# -*- coding: utf-8 -*-
# """
# 事件检测 V3.7（最终版）
# --------------------------------------------------
# 更新点：
# - steer_rate 触发 + 动态扩展事件边界
# - STOP_TH=1.5（更合理的动作结束判定）
# - MIN_DUR=0.15, MAX_DUR=3.0（保留短促纠偏，剔除过长稳态）
# - 自动识别 vx 列并处理 km/h → m/s
# - 无速度列时不会误杀扩展
# - 仍然支持 offroad 检测 / weak 噪声剔除 / episode / primary
# """
#
# import os
# import numpy as np
# import pandas as pd
# from glob import glob
#
#
# # =======================
# # 车辆参数
# # =======================
# G = 9.81
# H_COG = 1.2
# TRACK = 2.176
# LTR_COEFF = 2 * H_COG / (TRACK * G)     # ≈0.11243
#
#
# # =======================
# # 参数设置
# # =======================
# FS = 200                         # 采样频率 Hz
# MIN_EP_GAP = 5.0                 # episode 间隔 s
# Z_JUMP = 0.25                    # offroad 用
# SLOPE_JUMP = 0.15
#
# MIN_DUR = 0.15                   # 最短事件时长（秒）
# MAX_DUR = 3.0                    # 最长事件时长（秒）
#
# TRIGGER_TH = 4.0                 # 触发阈值：|steer_rate| > 4 rad/s
# STOP_TH = 1.5                    # 扩展停止阈值：|steer_rate| < 1.5 rad/s 认为动作基本结束
#
# VX_STOP = 3 / 3.6                # 低速保护：3 km/h → m/s
# STEER_STABLE = 0.005             # 方向盘稳定阈值（rad）
#
#
# # =======================
# # 工具函数：自动识别列名
# # =======================
# def find_col(df, keys):
#     for c in df.columns:
#         cc = c.lower()
#         if any(k in cc for k in keys):
#             return c
#     return None
#
#
# # =======================
# # 自动生成时间 t_s
# # =======================
# def get_ts(df):
#     if "t_s" in df.columns:
#         return df["t_s"].astype(float).values
#
#     if "StorageTime" in df.columns:
#         t_raw = pd.to_datetime(df["StorageTime"], errors="coerce")
#         if not t_raw.isna().all():
#             return (t_raw - t_raw.iloc[0]).dt.total_seconds().values
#
#     print("⚠ 使用第一列作为时间")
#     return pd.to_numeric(df.iloc[:, 0], errors="coerce").values
#
#
# # =======================
# # 自动生成 LTR_est
# # =======================
# def get_ltr(df, roll):
#     if "LTR_est" in df.columns:
#         return df["LTR_est"].astype(float).values
#
#     ay_col = find_col(df, ["ay"])
#     if ay_col:
#         ay = df[ay_col].astype(float).values
#         return LTR_COEFF * ay
#
#     print("⚠ 未找到 ay，LTR_est = sin(roll)")
#     return np.sin(roll)
#
#
# # =======================
# # very weak 噪声判定
# # =======================
# def is_very_weak(roll_peak, ltr_peak, steer_range, steer_rate_peak,
#                  yaw_rate_peak, roll_rate_peak, duration):
#     return (
#         abs(roll_peak) < 0.05 and
#         abs(ltr_peak) < 0.20 and
#         steer_range < 0.3 and
#         steer_rate_peak < 5 and
#         abs(yaw_rate_peak) < 0.2 and
#         abs(roll_rate_peak) < 0.1 and
#         duration < 0.4
#     )
#
#
# # =======================
# # offroad 判定（z + dz）
# # =======================
# def detect_offroad(z_seg, dz_seg):
#     if np.nanmax(z_seg) - np.nanmin(z_seg) > Z_JUMP:
#         return True
#     if np.nanmax(np.abs(dz_seg)) > SLOPE_JUMP:
#         return True
#     return False
#
#
# # =======================
# # 事件强度分类
# # =======================
# def classify_level(steer_range, steer_rate_peak, roll_peak, ltr_peak):
#     if steer_rate_peak >= 25 or abs(ltr_peak) >= 0.75:
#         return "extreme_active"
#     if steer_rate_peak >= 12 or abs(ltr_peak) >= 0.60:
#         return "strong_active"
#     if steer_range >= 0.9 and steer_rate_peak >= 7:
#         return "medium_active"
#     return "weak"
#
#
# # =======================
# # V3.7 核心：动态扩展事件边界
# # =======================
# def expand_event(i, steer_rate, steer, vx, t):
#     N = len(steer_rate)
#
#     # -------- 左扩展 --------
#     j = i
#     stable_counter = 0
#
#     while j > 0:
#         # steer_rate 足够小 → 认为动作已“开始前”
#         if abs(steer_rate[j]) < STOP_TH:
#             break
#         # 低速保护：vx 太小就不再扩展
#         if vx[j] < VX_STOP:
#             break
#
#         # 长时间保持极小方向盘变化 → 稳态，停止扩展
#         if abs(steer[j] - steer[j - 1]) < STEER_STABLE:
#             stable_counter += 1
#         else:
#             stable_counter = 0
#
#         if stable_counter > int(0.3 * FS):   # 连续 0.3 s 很稳定
#             break
#
#         j -= 1
#
#     start_idx = j
#
#     # -------- 右扩展 --------
#     k = i
#     stable_counter = 0
#
#     while k < N - 1:
#         if abs(steer_rate[k]) < STOP_TH:
#             break
#         if vx[k] < VX_STOP:
#             break
#
#         if abs(steer[k] - steer[k - 1]) < STEER_STABLE:
#             stable_counter += 1
#         else:
#             stable_counter = 0
#
#         if stable_counter > int(0.3 * FS):
#             break
#
#         k += 1
#
#     end_idx = k
#     # 确保最小扩展窗口长度不少于 0.2s（防止只扩到 1~2 个点）
#     min_pts = int(0.2 * FS)  # 0.2 秒对应的采样点数
#     if t[end_idx] - t[start_idx] < 0.2:
#         center = i
#         half = min_pts // 2
#         start_idx = max(0, center - half)
#         end_idx = min(N - 1, center + half)
#
#     return start_idx, end_idx
#
#
# # =======================
# # 主函数：处理一个原始车辆文件
# # =======================
# def process_vehicle(csv_path):
#     df = pd.read_csv(csv_path)
#
#     # 时间
#     t = get_ts(df)
#     dt = 1.0 / FS
#
#     # 自动识别列
#     roll_col = find_col(df, ["roll"])
#     steer_col = find_col(df, ["steering"])
#     yaw_col = find_col(df, ["vyaw", "yaw"])
#     vx_col = find_col(df, ["vx", "v_km/h", "v_km", "speed"])
#     z_col = find_col(df, ["|z"])
#
#     if roll_col is None or steer_col is None:
#         print("⚠ 缺少 roll 或 steering 列，跳过：", os.path.basename(csv_path))
#         return None
#
#     roll = df[roll_col].astype(float).values
#     steer = df[steer_col].astype(float).values
#     yaw_rate = df[yaw_col].astype(float).values if yaw_col else np.zeros_like(roll)
#
#     # 速度：尽量转为 m/s；没有的话用大值避免扩展被误杀
#     if vx_col:
#         vx_raw = df[vx_col].astype(float).values
#         if "km" in vx_col.lower():
#             vx = vx_raw / 3.6
#         else:
#             vx = vx_raw
#     else:
#         print("⚠ 未找到速度列 vx，假设全程高速（不启用低速限制）")
#         vx = np.full_like(roll, 100.0)
#
#     z = df[z_col].astype(float).values if z_col else np.zeros_like(roll)
#
#     # LTR
#     ltr = get_ltr(df, roll)
#
#     # 导数
#     steer_rate = np.append(np.diff(steer) / dt, 0.0)
#     roll_rate = np.append(np.diff(roll) / dt, 0.0)
#     dz = np.append(np.diff(z), 0.0)
#
#     # 触发点：|steer_rate| > TRIGGER_TH
#     trigger_idx = np.where(np.abs(steer_rate) > TRIGGER_TH)[0]
#
#     if len(trigger_idx) == 0:
#         return None
#
#     out = []
#     last_ev_end_t = -1e9   # 用于避免同一区间重复事件
#
#     for i in trigger_idx:
#         ti = t[i]
#         # 去掉起步 10s 内的噪声和慢速调整
#         if ti <= 10:
#             continue
#
#         # 如果触发点时间落在上一个事件内部，则跳过（避免重复）
#         if ti <= last_ev_end_t:
#             continue
#
#         # 动态扩展
#         s_idx, e_idx = expand_event(i, steer_rate, steer, vx, t)
#         s = t[s_idx]
#         e = t[e_idx]
#         duration = e - s
#
#         # 更新 last_ev_end_t
#         last_ev_end_t = e
#
#         # 时长过滤
#         if duration < MIN_DUR or duration > MAX_DUR:
#             continue
#
#         mask = (t >= s) & (t <= e)
#         if not np.any(mask):
#             continue
#
#         seg_roll = roll[mask]
#         seg_ltr = ltr[mask]
#         seg_steer = steer[mask]
#         seg_steer_rate = steer_rate[mask]
#         seg_yaw_rate = yaw_rate[mask]
#         seg_roll_rate = roll_rate[mask]
#         seg_vx = vx[mask]
#         seg_z = z[mask]
#         seg_dz = dz[mask]
#
#         roll_peak = np.nanmax(np.abs(seg_roll))
#         ltr_peak = np.nanmax(np.abs(seg_ltr))
#         steer_range = np.nanmax(seg_steer) - np.nanmin(seg_steer)
#         steer_rate_peak = np.nanmax(np.abs(seg_steer_rate))
#         yaw_rate_peak = np.nanmax(np.abs(seg_yaw_rate))
#         roll_rate_peak = np.nanmax(np.abs(seg_roll_rate))
#         z_range = np.nanmax(seg_z) - np.nanmin(seg_z)
#         dz_max = np.nanmax(np.abs(seg_dz))
#         vx_mean = np.nanmean(seg_vx)
#
#         if is_very_weak(roll_peak, ltr_peak, steer_range, steer_rate_peak,
#                         yaw_rate_peak, roll_rate_peak, duration):
#             continue
#
#         level = classify_level(steer_range, steer_rate_peak, roll_peak, ltr_peak)
#         offflag = detect_offroad(seg_z, seg_dz)
#
#         out.append({
#             "start_s": s,
#             "end_s": e,
#             "duration": duration,
#             "roll_peak": roll_peak,
#             "LTR_peak": ltr_peak,
#             "steer_range": steer_range,
#             "steer_rate_peak": steer_rate_peak,
#             "yaw_rate_peak": yaw_rate_peak,
#             "roll_rate_peak": roll_rate_peak,
#             "vx_mean": vx_mean,
#             "z_range": z_range,
#             "dz_max": dz_max,
#             "event_level": level,
#             "offroad": offflag
#         })
#
#     if len(out) == 0:
#         return None
#
#     df_out = pd.DataFrame(out)
#
#     # episode + primary
#     df_out = df_out.sort_values("start_s").reset_index(drop=True)
#     df_out["episode_id"] = -1
#     df_out["phase_type"] = "secondary"
#
#     ep = 0
#     last_end = -999
#
#     for i, row in df_out.iterrows():
#         if row["start_s"] - last_end > MIN_EP_GAP:
#             ep += 1
#         df_out.at[i, "episode_id"] = ep
#         last_end = row["end_s"]
#
#     # primary = 每个 episode 中 LTR_peak 最大的事件
#     for ep, sub in df_out.groupby("episode_id"):
#         idx = sub["LTR_peak"].idxmax()
#         df_out.at[idx, "phase_type"] = "primary"
#
#     return df_out
#
#
# # =======================
# # 批处理
# # =======================
# ROOT = r"F:\数据集处理\data_process\datasetprocess\数据预处理\车辆数据处理\正式车辆数据处理"
#
# print("\n=== V3.7 事件检测开始 ===\n")
#
# for subj in os.listdir(ROOT):
#     subj_dir = os.path.join(ROOT, subj)
#     if not os.path.isdir(subj_dir):
#         continue
#
#     print(f"\n--- 被试：{subj} ---")
#
#     # 只处理原始车辆数据
#     files = glob(os.path.join(subj_dir, "*vehicle_aligned_cleaned.csv"))
#
#     for f in files:
#         print("  ▶ 处理文件：", os.path.basename(f))
#
#         df_ev = process_vehicle(f)
#
#         if df_ev is None or df_ev.empty:
#             print("  ⚠ 无事件：", os.path.basename(f))
#             continue
#
#         out_csv = f.replace(".csv", "_events_v37.csv")
#         df_ev.to_csv(out_csv, index=False, encoding="utf-8-sig")
#         print("  ✔ 输出事件：", os.path.basename(out_csv))
#
# print("\n=== 全部完成 ===")



# -*- coding: utf-8 -*-
"""
事件检测 V3.8（稳定版）
--------------------------------------------------
修正内容：
- 不再使用 steer_rate<阈值 作为扩展结束条件（之前导致所有事件窗口只有 0.01s）
- 删除“强制补足 0.2s”逻辑（之前会让所有 duration=0.2s）
- 事件边界仅由：
    * 方向盘是否进入“稳定状态”（连续 0.3s 极小变化）
    * 最大时长限制 MAX_DUR
  来决定 → duration 变为真实可变时长
- 保留原有：
    * offroad 检测（z + dz）
    * very weak 噪声剔除
    * event_level 分类
    * episode + primary 标记
"""

import os
import numpy as np
import pandas as pd
from glob import glob


# =======================
# 车辆参数
# =======================
G = 9.81
H_COG = 1.2
TRACK = 2.176
LTR_COEFF = 2 * H_COG / (TRACK * G)     # ≈0.11243


# =======================
# 参数设置
# =======================
FS = 200                         # 采样频率 Hz
MIN_EP_GAP = 5.0                 # episode 间隔 s
Z_JUMP = 0.25                    # offroad 用
SLOPE_JUMP = 0.15

MIN_DUR = 0.15                   # 最短事件时长（秒）
MAX_DUR = 3.0                    # 最长事件时长（秒）

TRIGGER_TH = 4.0                 # 触发阈值：|steer_rate| > 4 rad/s
VX_STOP = 3 / 3.6                # 低速保护：3 km/h → m/s
STEER_STABLE = 0.005             # 方向盘稳定阈值（rad）≈ 0.3°
STABLE_TIME = 0.3                # 连续 0.3s 稳定视为动作结束
STABLE_SAMPLES = int(STABLE_TIME * FS)


# =======================
# 工具函数：自动识别列名
# =======================
def find_col(df, keys):
    for c in df.columns:
        cc = c.lower()
        if any(k in cc for k in keys):
            return c
    return None


# =======================
# 自动生成时间 t_s
# =======================
def get_ts(df):
    if "t_s" in df.columns:
        return df["t_s"].astype(float).values

    if "StorageTime" in df.columns:
        t_raw = pd.to_datetime(df["StorageTime"], errors="coerce")
        if not t_raw.isna().all():
            return (t_raw - t_raw.iloc[0]).dt.total_seconds().values

    print("⚠ 使用第一列作为时间")
    return pd.to_numeric(df.iloc[:, 0], errors="coerce").values


# =======================
# 自动生成 LTR_est
# =======================
def get_ltr(df, roll):
    if "LTR_est" in df.columns:
        return df["LTR_est"].astype(float).values

    ay_col = find_col(df, ["ay"])
    if ay_col:
        ay = df[ay_col].astype(float).values
        return LTR_COEFF * ay

    print("⚠ 未找到 ay，LTR_est = sin(roll)")
    return np.sin(roll)


# =======================
# very weak 噪声判定
# =======================
def is_very_weak(roll_peak, ltr_peak, steer_range, steer_rate_peak,
                 yaw_rate_peak, roll_rate_peak, duration):
    return (
        abs(roll_peak) < 0.05 and
        abs(ltr_peak) < 0.20 and
        steer_range < 0.3 and
        steer_rate_peak < 5 and
        abs(yaw_rate_peak) < 0.2 and
        abs(roll_rate_peak) < 0.1 and
        duration < 0.4
    )


# =======================
# offroad 判定（z + dz）
# =======================
def detect_offroad(z_seg, dz_seg):
    if np.nanmax(z_seg) - np.nanmin(z_seg) > Z_JUMP:
        return True
    if np.nanmax(np.abs(dz_seg)) > SLOPE_JUMP:
        return True
    return False


# =======================
# 事件强度分类
# =======================
def classify_level(steer_range, steer_rate_peak, roll_peak, ltr_peak):
    if steer_rate_peak >= 25 or abs(ltr_peak) >= 0.75:
        return "extreme_active"
    if steer_rate_peak >= 12 or abs(ltr_peak) >= 0.60:
        return "strong_active"
    if steer_range >= 0.9 and steer_rate_peak >= 7:
        return "medium_active"
    return "weak"


# =======================
# V3.8 核心：动态扩展事件边界（无 STOP_TH）
# =======================
def expand_event(i, steer_rate, steer, vx, t):
    """
    基于“方向盘进入稳定状态 + 最大时长限制”的扩展：
    - 不再用 steer_rate<阈值 作为结束条件
    - 只要方向盘仍在明显变化，就继续扩展
    - 当连续 STABLE_TIME 时间内 Δsteer 都很小 → 视为动作结束
    """
    N = len(steer_rate)
    center_t = t[i]

    # -------- 左扩展 --------
    j = i
    stable_counter = 0
    while j > 0:
        # 超过最大时长的一半就不再向左扩展（左右各最多 MAX_DUR/2）
        if center_t - t[j] > MAX_DUR / 2:
            break

        # 低速保护：极低速时不扩太多
        if vx[j] < VX_STOP:
            break

        # 判断这一小步方向盘是否足够“稳定”
        if abs(steer[j] - steer[j - 1]) < STEER_STABLE:
            stable_counter += 1
        else:
            stable_counter = 0

        # 若已经有 0.3s 连续稳定，说明动作在更靠前就结束了
        if stable_counter >= STABLE_SAMPLES:
            break

        j -= 1

    start_idx = j

    # -------- 右扩展 --------
    k = i
    stable_counter = 0
    while k < N - 1:
        if t[k] - center_t > MAX_DUR / 2:
            break

        if vx[k] < VX_STOP:
            break

        if abs(steer[k] - steer[k - 1]) < STEER_STABLE:
            stable_counter += 1
        else:
            stable_counter = 0

        if stable_counter >= STABLE_SAMPLES:
            break

        k += 1

    end_idx = k

    return start_idx, end_idx


# =======================
# 主函数：处理一个原始车辆文件
# =======================
def process_vehicle(csv_path):
    df = pd.read_csv(csv_path)

    # 时间
    t = get_ts(df)
    dt = 1.0 / FS

    # 自动识别列
    roll_col = find_col(df, ["roll"])
    steer_col = find_col(df, ["steering"])
    yaw_col = find_col(df, ["vyaw", "yaw"])
    vx_col = find_col(df, ["vx", "v_km/h", "v_km", "speed"])
    z_col = find_col(df, ["|z"])

    if roll_col is None or steer_col is None:
        print("⚠ 缺少 roll 或 steering 列，跳过：", os.path.basename(csv_path))
        return None

    roll = df[roll_col].astype(float).values
    steer = df[steer_col].astype(float).values
    yaw_rate = df[yaw_col].astype(float).values if yaw_col else np.zeros_like(roll)

    # 速度列：尽量转成 m/s；没有则假定高速避免误杀
    if vx_col:
        vx_raw = df[vx_col].astype(float).values
        if "km" in vx_col.lower():
            vx = vx_raw / 3.6
        else:
            vx = vx_raw
    else:
        print("⚠ 未找到速度列 vx，假设全程高速（不启用低速限制）")
        vx = np.full_like(roll, 100.0)

    z = df[z_col].astype(float).values if z_col else np.zeros_like(roll)

    # LTR
    ltr = get_ltr(df, roll)

    # 导数
    steer_rate = np.append(np.diff(steer) / dt, 0.0)
    roll_rate = np.append(np.diff(roll) / dt, 0.0)
    dz = np.append(np.diff(z), 0.0)

    # 触发点
    trigger_idx = np.where(np.abs(steer_rate) > TRIGGER_TH)[0]
    if len(trigger_idx) == 0:
        return None

    out = []
    last_ev_end_t = -1e9

    for i in trigger_idx:
        ti = t[i]

        # 去掉起步 10 s 以内
        if ti <= 10:
            continue

        # 避免同一个时间段内重复事件
        if ti <= last_ev_end_t:
            continue

        # 动态扩展事件边界
        s_idx, e_idx = expand_event(i, steer_rate, steer, vx, t)
        s = t[s_idx]
        e = t[e_idx]
        duration = e - s

        last_ev_end_t = e

        # 基于时长过滤
        if duration < MIN_DUR or duration > MAX_DUR:
            continue

        mask = (t >= s) & (t <= e)
        if not np.any(mask):
            continue

        seg_roll = roll[mask]
        seg_ltr = ltr[mask]
        seg_steer = steer[mask]
        seg_steer_rate = steer_rate[mask]
        seg_yaw_rate = yaw_rate[mask]
        seg_roll_rate = roll_rate[mask]
        seg_vx = vx[mask]
        seg_z = z[mask]
        seg_dz = dz[mask]

        roll_peak = np.nanmax(np.abs(seg_roll))
        ltr_peak = np.nanmax(np.abs(seg_ltr))
        steer_range = np.nanmax(seg_steer) - np.nanmin(seg_steer)
        steer_rate_peak = np.nanmax(np.abs(seg_steer_rate))
        yaw_rate_peak = np.nanmax(np.abs(seg_yaw_rate))
        roll_rate_peak = np.nanmax(np.abs(seg_roll_rate))
        z_range = np.nanmax(seg_z) - np.nanmin(seg_z)
        dz_max = np.nanmax(np.abs(seg_dz))
        vx_mean = np.nanmean(seg_vx)

        if is_very_weak(roll_peak, ltr_peak, steer_range, steer_rate_peak,
                        yaw_rate_peak, roll_rate_peak, duration):
            continue

        level = classify_level(steer_range, steer_rate_peak, roll_peak, ltr_peak)
        offflag = detect_offroad(seg_z, seg_dz)

        out.append({
            "start_s": s,
            "end_s": e,
            "duration": duration,
            "roll_peak": roll_peak,
            "LTR_peak": ltr_peak,
            "steer_range": steer_range,
            "steer_rate_peak": steer_rate_peak,
            "yaw_rate_peak": yaw_rate_peak,
            "roll_rate_peak": roll_rate_peak,
            "vx_mean": vx_mean,
            "z_range": z_range,
            "dz_max": dz_max,
            "event_level": level,
            "offroad": offflag
        })

    if len(out) == 0:
        return None

    df_out = pd.DataFrame(out)

    # episode + primary
    df_out = df_out.sort_values("start_s").reset_index(drop=True)
    df_out["episode_id"] = -1
    df_out["phase_type"] = "secondary"

    ep = 0
    last_end = -999

    for i, row in df_out.iterrows():
        if row["start_s"] - last_end > MIN_EP_GAP:
            ep += 1
        df_out.at[i, "episode_id"] = ep
        last_end = row["end_s"]

    # primary = 每个 episode 中 LTR_peak 最大
    for ep, sub in df_out.groupby("episode_id"):
        idx = sub["LTR_peak"].idxmax()
        df_out.at[idx, "phase_type"] = "primary"

    return df_out


# =======================
# 批处理
# =======================
ROOT = r"F:\数据集处理\data_process\datasetprocess\数据预处理\车辆数据处理\正式车辆数据处理"

print("\n=== V3.8 事件检测开始 ===\n")

for subj in os.listdir(ROOT):
    subj_dir = os.path.join(ROOT, subj)
    if not os.path.isdir(subj_dir):
        continue

    print(f"\n--- 被试：{subj} ---")

    files = glob(os.path.join(subj_dir, "*vehicle_aligned_cleaned.csv"))

    for f in files:
        print("  ▶ 处理文件：", os.path.basename(f))

        df_ev = process_vehicle(f)

        if df_ev is None or df_ev.empty:
            print("  ⚠ 无事件：", os.path.basename(f))
            continue

        out_csv = f.replace(".csv", "_events_v38.csv")
        df_ev.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print("  ✔ 输出事件：", os.path.basename(out_csv))

print("\n=== 全部完成 ===")
