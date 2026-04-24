# -*- coding: utf-8 -*-
"""
reclean_physio_selected_v1.py
-----------------------------------
功能：
  1. 自动扫描所有被试的 *_physio_filterstats.csv
  2. 根据严格规则判定是否需要对 EDA / RESP 重洗
  3. 仅重洗需要的通道，不重复计算 ECG 和 EMG
  4. 输出新版本：
      *_physio_reclean_200Hz.csv
      *_reclean_filterstats.csv
      *_reclean_filtercheck.png
  5. 不覆盖旧文件，便于对比

依赖：
  pip install numpy pandas scipy matplotlib neurokit2
"""

import os, re, math, warnings, traceback
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, detrend, iirnotch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# ========================================================
# 目录配置（请修改为你的路径）
# ========================================================
ROOT_IN = Path(r"F:\数据集处理\datasetprocess\数据预处理\生理数据处理_200Hz")
ROOT_OUT = Path(r"F:\数据集处理\datasetprocess\数据预处理\生理数据处理_200Hz_reclean_v2")

FS_OUT = 200.0   # 输出采样率

# ========================================================
# 滤波器
# ========================================================

def butter_band(sig, fs, low, high, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, sig)

def butter_low(sig, fs, cutoff, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff/nyq, btype='low')
    return filtfilt(b, a, sig)

def notch50(sig, fs, Q=30):
    b, a = iirnotch(50/(fs/2), Q)
    return filtfilt(b, a, sig)


# ========================================================
# 判断是否需要重洗
# ========================================================
def should_reclean(signal_name, row_stats, clean_df):

    if signal_name == "EDA":
        std_raw = np.nanstd(clean_df["EDA_raw200"])
        if row_stats["Delta_STD_percent"] > -10:     # 平滑度不足
            return True
        if row_stats["SNR_gain_db"] < 5:             # SNR 改善不足
            return True
        if std_raw < 1e-3:                           # 几乎一条直线
            return True
        return False

    if signal_name == "RESP":
        std_raw = np.nanstd(clean_df["RESP_raw200"])
        if row_stats["Delta_STD_percent"] > -15:
            return True
        if row_stats["SNR_gain_db"] < 3:
            return True
        if std_raw < 1e-3:
            return True
        return False

    # ECG / EMG 不重洗
    return False


# ========================================================
# 重洗单个通道
# ========================================================
def reclean_channel(raw200, fs, sig_type):
    if sig_type == "EDA":
        # 更严格去噪
        sig = butter_low(raw200, fs, cutoff=3.0)
        sig = detrend(sig, type="constant")
        return sig

    if sig_type == "RESP":
        sig = butter_low(raw200, fs, cutoff=2.0)
        sig = detrend(sig, type="constant")
        return sig

    return raw200


# ========================================================
# 主函数：处理单个文件
# ========================================================
def process_one(st_filtstats, st_clean, out_dir):

    df_stats = pd.read_csv(st_filtstats)
    df_clean = pd.read_csv(st_clean)

    out_dir.mkdir(parents=True, exist_ok=True)

    # 判定重洗的通道
    reclean_list = []
    for _, row in df_stats.iterrows():
        sig = row["Signal"]
        if should_reclean(sig, row, df_clean):
            reclean_list.append(sig)

    print(f"需要重洗通道：{reclean_list}")

    # 如果没有要重洗的
    if len(reclean_list) == 0:
        print("无通道需要重洗，直接复制原结果。")
        df_clean.to_csv(out_dir / (st_clean.stem+"_reclean_200Hz.csv"),
                        index=False, encoding="utf-8-sig")
        return

    # 复制原始数据
    df_out = df_clean.copy()

    # 对需要重洗的通道重新滤波
    for sig in reclean_list:
        raw_col = f"{sig}_raw200"
        if raw_col not in df_out.columns:
            continue
        print(f"重洗：{sig}")

        raw = df_out[raw_col].to_numpy()
        filt_new = reclean_channel(raw, FS_OUT, sig)
        df_out[f"{sig}_filt200"] = filt_new

    # 保存新版本
    save_csv = out_dir / (st_clean.stem + "_reclean_200Hz.csv")
    df_out.to_csv(save_csv, index=False, encoding="utf-8-sig")

    # 输出检查图
    plt.figure(figsize=(15,8))
    t = df_out["t_s"]
    for i, sig in enumerate(reclean_list):
        plt.subplot(len(reclean_list), 1, i+1)
        plt.plot(t, df_clean[f"{sig}_filt200"], label="旧滤波", alpha=0.6)
        plt.plot(t, df_out[f"{sig}_filt200"], label="新滤波", alpha=0.9)
        plt.title(f"{sig} 重洗前后")
        plt.legend()
        plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / (st_clean.stem + "_reclean_filtercheck.png"), dpi=150)
    plt.close()

    print(f"重洗完成 → {save_csv}")


# ========================================================
# 批量运行
# ========================================================
def main():

    subs = [p for p in ROOT_IN.iterdir() if p.is_dir()]
    if not subs:
        subs = [ROOT_IN]

    for sub in subs:
        print(f"\n=== 处理被试：{sub.name} ===")

        filtstats_files = list(sub.glob("*_physio_filterstats.csv"))
        clean_files = list(sub.glob("*_physio_cleaned_200Hz.csv"))
        if not filtstats_files or not clean_files:
            print("缺少必要文件，跳过。")
            continue

        out_dir = ROOT_OUT / sub.name

        for fs, cl in zip(filtstats_files, clean_files):
            try:
                process_one(fs, cl, out_dir)
            except Exception:
                print(f"❌ 处理失败：{fs}\n{traceback.format_exc()}")

    print("\n全部处理完成。")


if __name__ == "__main__":
    main()
