# # -*- coding: utf-8 -*-
# """
# 事件检测 V3.10（终极版）
# --------------------------------------------------
# 核心特性：
# - steer_rate > 4 触发
# - 动态扩展事件边界（基于方向盘进入“稳定状态” + MAX_DUR 限制）
# - very weak 噪声剔除
# - event_level 分类（weak/medium_active/strong_active/extreme_active）
# - episode 划分 + primary（LTR_peak 最大为 primary）
#
# offroad & 训练相关（本版新增）：
# - 使用 lateraldistance 判定是否在道路外：
#     * 道路宽度 3.5 m → 半宽 1.75 m
#     * |lateraldistance| > 1.75 → offroad
# - 对每个事件计算：
#     * offroad_any：是否出现过路外
#     * offroad_ratio：路外点占比
#     * offroad_max_cont_time：最长连续路外时间（秒）
# - 根据 offroad 模式生成：
#     * keep_for_training：是否建议保留用于训练
#
# 保留用于训练（keep_for_training=True）的逻辑：
# - 如果完全在道路内（offroad_any=False） → 保留
# - 如果有越界，但：
#     * offroad_ratio < 0.30   （短暂偏离） 或
#     * offroad_max_cont_time < 0.80 s （最长连续路外 <0.8秒） 或
#     * duration < 1.2 s       （事件本身很短，多半是“出界+立刻纠偏”）
#   → 保留
# - 只有当：
#     * offroad_ratio >= 0.30 且
#     * offroad_max_cont_time >= 0.80 s 且
#     * duration >= 1.2 s
#   同时满足时，认为该事件是“长时间在路外稳定行驶”，不保留用于训练
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
#
# # 道路宽度 & lateraldistance 判定
# ROAD_WIDTH = 3.5
# ROAD_HALF_WIDTH = ROAD_WIDTH / 2.0   # = 1.75 m
#
# MIN_DUR = 0.15                   # 最短事件时长（秒）
# MAX_DUR = 3.0                    # 最长事件时长（秒）
#
# TRIGGER_TH = 4.0                 # 触发阈值：|steer_rate| > 4 rad/s
# VX_STOP = 3 / 3.6                # 低速保护：3 km/h → m/s
# STEER_STABLE = 0.005             # 方向盘稳定阈值（rad）≈ 0.3°
# STABLE_TIME = 0.3                # 连续 0.3s 稳定判定为动作结束
# STABLE_SAMPLES = int(STABLE_TIME * FS)
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
# # lateraldistance → offroad 统计
# # =======================
# def compute_offroad_metrics(seg_latdist):
#     """
#     根据 lateraldistance 计算：
#     - offroad_any: 是否出现路外
#     - offroad_ratio: 路外点占比
#     - offroad_max_cont_time: 最长连续路外持续时间（秒）
#     """
#     seg_latdist = np.asarray(seg_latdist, dtype=float)
#     n = len(seg_latdist)
#     if n == 0:
#         return False, 0.0, 0.0
#
#     off_mask = np.abs(seg_latdist) > ROAD_HALF_WIDTH  # True=在路外
#     if not np.any(off_mask):
#         return False, 0.0, 0.0
#
#     offroad_any = True
#     offroad_ratio = float(np.sum(off_mask)) / float(n)
#
#     # 计算最长连续 True 的 run-length
#     max_run = 0
#     cur = 0
#     for flag in off_mask:
#         if flag:
#             cur += 1
#             if cur > max_run:
#                 max_run = cur
#         else:
#             cur = 0
#
#     offroad_max_cont_time = max_run / FS
#     return offroad_any, offroad_ratio, offroad_max_cont_time
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
# # 动态扩展事件边界（与 V3.8 一致）
# # =======================
# def expand_event(i, steer, vx, t):
#     """
#     基于“方向盘进入稳定状态 + 最大时长限制”的扩展：
#     - 不使用 steer_rate<阈值 做结束条件
#     - 只要方向盘仍在明显变化，就继续扩展
#     - 当连续 STABLE_TIME 时间内 Δsteer 都很小 → 视为动作结束
#     """
#     N = len(steer)
#     center_t = t[i]
#
#     # -------- 左扩展 --------
#     j = i
#     stable_counter = 0
#     while j > 0:
#         if center_t - t[j] > MAX_DUR / 2:
#             break
#         if vx[j] < VX_STOP:
#             break
#
#         if abs(steer[j] - steer[j - 1]) < STEER_STABLE:
#             stable_counter += 1
#         else:
#             stable_counter = 0
#
#         if stable_counter >= STABLE_SAMPLES:
#             break
#
#         j -= 1
#
#     start_idx = j
#
#     # -------- 右扩展 --------
#     k = i
#     stable_counter = 0
#     while k < N - 1:
#         if t[k] - center_t > MAX_DUR / 2:
#             break
#         if vx[k] < VX_STOP:
#             break
#
#         if abs(steer[k] - steer[k - 1]) < STEER_STABLE:
#             stable_counter += 1
#         else:
#             stable_counter = 0
#
#         if stable_counter >= STABLE_SAMPLES:
#             break
#
#         k += 1
#
#     end_idx = k
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
#     latdist_col = find_col(df, ["lateraldistance"])
#
#     if roll_col is None or steer_col is None:
#         print("⚠ 缺少 roll 或 steering 列，跳过：", os.path.basename(csv_path))
#         return None
#
#     roll = df[roll_col].astype(float).values
#     steer = df[steer_col].astype(float).values
#     yaw_rate = df[yaw_col].astype(float).values if yaw_col else np.zeros_like(roll)
#
#     # 速度：尽量转为 m/s；没有就假设高速
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
#     # lateraldistance
#     if latdist_col:
#         latdist = df[latdist_col].astype(float).values
#     else:
#         print("⚠ 未找到 lateraldistance 列，所有事件视为在道路内（offroad_any=False）")
#         latdist = None
#
#     # LTR
#     ltr = get_ltr(df, roll)
#
#     # 导数
#     steer_rate = np.append(np.diff(steer) / dt, 0.0)
#     roll_rate = np.append(np.diff(roll) / dt, 0.0)
#     dz = np.append(np.diff(z), 0.0)
#
#     # 触发点
#     trigger_idx = np.where(np.abs(steer_rate) > TRIGGER_TH)[0]
#     if len(trigger_idx) == 0:
#         return None
#
#     out = []
#     last_ev_end_t = -1e9
#
#     for i in trigger_idx:
#         ti = t[i]
#         if ti <= 10:
#             continue
#
#         # 避免一个时间段重复事件
#         if ti <= last_ev_end_t:
#             continue
#
#         # 动态扩展
#         s_idx, e_idx = expand_event(i, steer, vx, t)
#         s = t[s_idx]
#         e = t[e_idx]
#         duration = e - s
#         last_ev_end_t = e
#
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
#         # lateraldistance 相关
#         if latdist is not None:
#             seg_latdist = latdist[mask]
#             off_any, off_ratio, off_max_cont = compute_offroad_metrics(seg_latdist)
#         else:
#             seg_latdist = None
#             off_any, off_ratio, off_max_cont = False, 0.0, 0.0
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
#
#         # 事件是否用于训练（核心逻辑）
#         if not off_any:
#             keep_for_training = True
#         else:
#             # 有越界的情况：
#             if (off_ratio >= 0.30 and
#                 off_max_cont >= 0.80 and
#                 duration >= 1.20):
#                 # 长时间在路外稳定行驶 → 不用于训练
#                 keep_for_training = False
#             else:
#                 # 短暂越界 / 立刻纠偏 → 有用
#                 keep_for_training = True
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
#
#             # offroad 相关字段（基于 lateraldistance）
#             "offroad_any": off_any,
#             "offroad_ratio": off_ratio,
#             "offroad_max_cont_time": off_max_cont,
#             "keep_for_training": keep_for_training
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
#     # 每个 episode 中 LTR_peak 最大的事件标为 primary
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
# print("\n=== V3.10 事件检测开始 ===\n")
#
# for subj in os.listdir(ROOT):
#     subj_dir = os.path.join(ROOT, subj)
#     if not os.path.isdir(subj_dir):
#         continue
#
#     print(f"\n--- 被试：{subj} ---")
#
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
#         out_csv = f.replace(".csv", "_events_v310.csv")
#         df_ev.to_csv(out_csv, index=False, encoding="utf-8-sig")
#         print("  ✔ 输出事件：", os.path.basename(out_csv))
#
# print("\n=== 全部完成 ===")

# -*- coding: utf-8 -*-
"""
事件检测 V3.12（修复 last_ev_end_t 覆盖 bug）
--------------------------------------------------
与 V3.11 相比，主要改动：
1) 只有当事件最终被保留时，才更新 last_ev_end_t
   - 解决了“一个被丢弃的长事件挡住后面真正需要的事件”的问题
   - 145s 那段 lateraldistance>1.75 的 offroad 事件会被正常检测到
2) 时长上限允许极小容差：duration > MAX_DUR + EPS 才剔除

其它逻辑保持与 V3.11 一致：
- steer_rate > 4 触发主动事件
- lateraldistance 越界触发 offroad 事件
- lane-departure 使用固定窗口 [t-0.5, t+1.0]
- very weak 噪声剔除
- offroad_ratio + offroad_max_cont_time + duration → keep_for_training
- episode 划分 + primary 选取
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

ROAD_WIDTH = 3.5
ROAD_HALF_WIDTH = ROAD_WIDTH / 2.0   # = 1.75 m

MIN_DUR = 0.15                   # 最短事件时长（秒）
MAX_DUR = 3.0                    # 最长事件时长（秒）
EPS_DUR = 0.02                   # 时长容差，允许到 3.02s 左右

TRIGGER_TH = 4.0                 # 主动触发阈值：|steer_rate| > 4 rad/s
VX_STOP = 3 / 3.6                # 低速保护：3 km/h → m/s
STEER_STABLE = 0.005             # 方向盘稳定阈值（rad）≈ 0.3°
STABLE_TIME = 0.3                # 连续 0.3s 稳定判定为动作结束
STABLE_SAMPLES = int(STABLE_TIME * FS)

LAT_WIN_PRE = 0.5                # lat 触发前窗口
LAT_WIN_POST = 1.0               # lat 触发后窗口


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
# lateraldistance → offroad 统计
# =======================
def compute_offroad_metrics(seg_latdist):
    seg_latdist = np.asarray(seg_latdist, dtype=float)
    n = len(seg_latdist)
    if n == 0:
        return False, 0.0, 0.0

    off_mask = np.abs(seg_latdist) > ROAD_HALF_WIDTH
    if not np.any(off_mask):
        return False, 0.0, 0.0

    offroad_any = True
    offroad_ratio = float(np.sum(off_mask)) / float(n)

    max_run = 0
    cur = 0
    for flag in off_mask:
        if flag:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0

    offroad_max_cont_time = max_run / FS
    return offroad_any, offroad_ratio, offroad_max_cont_time


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
# steer 触发事件扩展
# =======================
def expand_event_steer(i, steer, vx, t):
    N = len(steer)
    center_t = t[i]

    j = i
    stable_counter = 0
    while j > 0:
        if center_t - t[j] > MAX_DUR / 2:
            break
        if vx[j] < VX_STOP:
            break
        if abs(steer[j] - steer[j - 1]) < STEER_STABLE:
            stable_counter += 1
        else:
            stable_counter = 0
        if stable_counter >= STABLE_SAMPLES:
            break
        j -= 1
    start_idx = j

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
# lat 触发事件扩展（固定窗口）
# =======================
def expand_event_lat(i, t):
    N = len(t)
    center_t = t[i]

    start_t = max(t[0], center_t - LAT_WIN_PRE)
    end_t = min(t[-1], center_t + LAT_WIN_POST)
    if end_t - start_t > MAX_DUR + EPS_DUR:
        end_t = start_t + MAX_DUR

    start_idx = int(np.searchsorted(t, start_t, side="left"))
    end_idx = int(np.searchsorted(t, end_t, side="right")) - 1
    end_idx = max(start_idx, min(end_idx, N - 1))
    return start_idx, end_idx


# =======================
# 主函数：处理一个原始车辆文件
# =======================
def process_vehicle(csv_path):
    df = pd.read_csv(csv_path)

    t = get_ts(df)
    dt = 1.0 / FS

    roll_col = find_col(df, ["roll"])
    steer_col = find_col(df, ["steering"])
    yaw_col = find_col(df, ["vyaw", "yaw"])
    vx_col = find_col(df, ["vx", "v_km/h", "v_km", "speed"])
    z_col = find_col(df, ["|z"])
    latdist_col = find_col(df, ["lateraldistance", "latdist", "lane", "offset"])

    if roll_col is None or steer_col is None:
        print("⚠ 缺少 roll 或 steering 列，跳过：", os.path.basename(csv_path))
        return None

    roll = df[roll_col].astype(float).values
    steer = df[steer_col].astype(float).values
    yaw_rate = df[yaw_col].astype(float).values if yaw_col else np.zeros_like(roll)

    if vx_col:
        vx_raw = df[vx_col].astype(float).values
        if "km" in vx_col.lower():
            vx = vx_raw / 3.6
        else:
            vx = vx_raw
    else:
        print("⚠ 未找到速度列 vx，假设全程高速")
        vx = np.full_like(roll, 100.0)

    z = df[z_col].astype(float).values if z_col else np.zeros_like(roll)

    if latdist_col:
        latdist = df[latdist_col].astype(float).values
    else:
        print("⚠ 未找到 lateraldistance 列，本文件不做 offroad")
        latdist = None

    ltr = get_ltr(df, roll)

    steer_rate = np.append(np.diff(steer) / dt, 0.0)
    roll_rate = np.append(np.diff(roll) / dt, 0.0)
    dz = np.append(np.diff(z), 0.0)

    # --- 触发点 ---
    trigger_steer = np.where(np.abs(steer_rate) > TRIGGER_TH)[0]

    trigger_lat = []
    if latdist is not None:
        off_mask = np.abs(latdist) > ROAD_HALF_WIDTH
        prev_off = np.concatenate(([False], off_mask[:-1]))
        edge = np.where(off_mask & (~prev_off))[0]
        trigger_lat = edge

    if (trigger_steer.size == 0) and (len(trigger_lat) == 0):
        return None

    triggers = []
    for i in trigger_steer:
        triggers.append((t[i], i, "steer"))
    for i in trigger_lat:
        triggers.append((t[i], i, "lat"))
    triggers.sort(key=lambda x: x[0])

    out = []
    last_ev_end_t = -1e9

    for ti, idx, kind in triggers:
        if ti <= 10:
            continue
        if ti <= last_ev_end_t:
            continue

        # ① 扩展窗口
        if kind == "steer":
            s_idx, e_idx = expand_event_steer(idx, steer, vx, t)
        else:
            s_idx, e_idx = expand_event_lat(idx, t)

        s = t[s_idx]
        e = t[e_idx]
        duration = e - s

        # ② 时长过滤（注意加了 EPS_DUR）—— 不合格事件不会更新 last_ev_end_t
        if duration < MIN_DUR or duration > MAX_DUR + EPS_DUR:
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

        if latdist is not None:
            seg_latdist = latdist[mask]
            off_any, off_ratio, off_max_cont = compute_offroad_metrics(seg_latdist)
        else:
            seg_latdist = None
            off_any, off_ratio, off_max_cont = False, 0.0, 0.0

        roll_peak = np.nanmax(np.abs(seg_roll))
        ltr_peak = np.nanmax(np.abs(seg_ltr))
        steer_range = np.nanmax(seg_steer) - np.nanmin(seg_steer)
        steer_rate_peak = np.nanmax(np.abs(seg_steer_rate))
        yaw_rate_peak = np.nanmax(np.abs(seg_yaw_rate))
        roll_rate_peak = np.nanmax(np.abs(seg_roll_rate))
        z_range = np.nanmax(seg_z) - np.nanmin(seg_z)
        dz_max = np.nanmax(np.abs(seg_dz))
        vx_mean = np.nanmean(seg_vx)

        # ③ very weak 过滤——同样不更新 last_ev_end_t
        if is_very_weak(roll_peak, ltr_peak, steer_range, steer_rate_peak,
                        yaw_rate_peak, roll_rate_peak, duration):
            continue

        level = classify_level(steer_range, steer_rate_peak, roll_peak, ltr_peak)

        # ④ 是否用于训练
        if not off_any:
            keep_for_training = True
        else:
            if (off_ratio >= 0.30 and
                off_max_cont >= 0.80 and
                duration >= 1.20):
                keep_for_training = False
            else:
                keep_for_training = True

        # ✅ 到这里，事件被真正保留，才更新 last_ev_end_t
        last_ev_end_t = e

        out.append({
            "start_s": s,
            "end_s": e,
            "duration": duration,
            "trigger_type": kind,
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
            "offroad_any": off_any,
            "offroad_ratio": off_ratio,
            "offroad_max_cont_time": off_max_cont,
            "keep_for_training": keep_for_training
        })

    if len(out) == 0:
        return None

    df_out = pd.DataFrame(out)

    # --- episode + primary ---
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

    for ep, sub in df_out.groupby("episode_id"):
        idx = sub["LTR_peak"].idxmax()
        df_out.at[idx, "phase_type"] = "primary"

    return df_out


# =======================
# 批处理
# =======================
ROOT = r"F:\数据集处理\data_process\datasetprocess\数据预处理\车辆数据处理\正式车辆数据处理"

print("\n=== V3.12 事件检测开始 ===\n")

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

        out_csv = f.replace(".csv", "_events_v312.csv")
        df_ev.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print("  ✔ 输出事件：", os.path.basename(out_csv))

print("\n=== 全部完成 ===")

