# -*- coding: utf-8 -*-
"""
Physio raw signal visualization
--------------------------------
- 遍历指定 physio 文件夹
- 自动识别 ECG / EDA / EMG / RESP 等信号
- 用于初步检查生理数据质量与缺失情况
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

# =========================
# 路径配置
# =========================
PHYSIO_DIR = r"F:\数据集处理\data_process\datasetprocess\多模态数据\被试数据集合\zx\physio"

# 采样率（如果你确定是 200 Hz）
FS = 200.0

# =========================
# 常见生理信号列名候选
# （按你之前的数据习惯来）
# =========================
SIGNAL_CANDIDATES = {
    "ECG":  ["ECG", "ECG_raw200", "ECG_filt200"],
    "EDA":  ["EDA", "EDA_raw200", "EDA_filt200", "EDA_Tonic", "EDA_Phasic"],
    "EMG":  ["EMG", "EMG_raw200", "EMG_filt200", "EMG_RMS"],
    "RESP": ["RESP", "RESP_raw200", "RESP_filt200", "RESP_BPM", "RESP_Amplitude"],
    "HR":   ["HR", "HR_bpm"],
    "HRV":  ["HRV", "HRV_RMSSD"]
}

# =========================
# 小工具：在列名中找信号
# =========================
def find_signal_columns(df_cols, candidates):
    found = []
    for c in candidates:
        if c in df_cols:
            found.append(c)
    return found

# =========================
# 主流程
# =========================
csv_files = sorted(glob.glob(os.path.join(PHYSIO_DIR, "*.csv")))

print(f"📂 发现 {len(csv_files)} 个生理数据文件")

for csv_path in csv_files:
    print(f"\n🔍 正在查看：{os.path.basename(csv_path)}")
    df = pd.read_csv(csv_path)

    if df.empty:
        print("⚠ 文件为空，跳过")
        continue

    cols = df.columns.tolist()

    # 时间轴
    if "t_s" in cols:
        t = df["t_s"].values
        xlabel = "Time (s)"
    else:
        t = np.arange(len(df)) / FS
        xlabel = "Time (s, assumed)"

    # 找到可画的信号
    plot_items = []
    for sig_name, candidates in SIGNAL_CANDIDATES.items():
        found_cols = find_signal_columns(cols, candidates)
        for c in found_cols:
            # 跳过全 NaN
            if df[c].isna().all():
                print(f"  ⛔ {c} 全 NaN，跳过")
                continue
            plot_items.append((sig_name, c))

    if len(plot_items) == 0:
        print("⚠ 未识别到任何常见生理信号")
        continue

    # =========================
    # 画图
    # =========================
    n = len(plot_items)
    plt.figure(figsize=(12, 2.2 * n))

    for i, (sig_name, col) in enumerate(plot_items, 1):
        ax = plt.subplot(n, 1, i)
        ax.plot(t, df[col].values, linewidth=0.8)
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.3)
        if i == 1:
            ax.set_title(os.path.basename(csv_path))

    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.show()
