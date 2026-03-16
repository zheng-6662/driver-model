# -*- coding: utf-8 -*-
"""
physio_filter_quality_summary_v1.py
-----------------------------------
功能：
  - 扫描所有被试目录下的 *_physio_filterstats.csv
  - 合并为一个汇总表 filter_summary_ALL.xlsx
  - 自动计算是否平滑有效 / SNR 提升 / 综合有效
  - 生成每被试的平均滤波性能统计

使用方法：
  1. 修改 ROOT_DIR 为你的生理数据处理输出根目录
  2. 运行脚本后，会在根目录生成 filter_summary_ALL.xlsx

依赖：
  pip install pandas openpyxl
"""

import pandas as pd
from pathlib import Path
import numpy as np

# ========== 路径配置 ==========
ROOT_DIR = Path(r"F:\数据集处理\datasetprocess\数据预处理\生理数据处理_200Hz")  # 修改为你的输出目录
OUT_XLSX = ROOT_DIR / "filter_summary_ALL.xlsx"

# ========== 扫描与合并 ==========
rows = []
for sub in ROOT_DIR.rglob("*_physio_filterstats.csv"):
    try:
        df = pd.read_csv(sub)
        subject = sub.parent.name
        file_stem = sub.stem
        # 添加被试与文件名
        df["Subject"] = subject
        df["File"] = file_stem

        # 判定有效性
        df["Effective_Smooth"] = df["Delta_STD_percent"] < -10
        df["Effective_SNR"] = df["SNR_gain_db"] > 1.0
        df["Overall_Effective"] = df["Effective_Smooth"] & df["Effective_SNR"]

        rows.append(df)
    except Exception as e:
        print(f"❌ 读取失败：{sub.name} ({e})")

if not rows:
    raise RuntimeError("未在指定目录下找到任何 _physio_filterstats.csv 文件")

df_all = pd.concat(rows, ignore_index=True)

# ========== 计算每个被试的平均指标 ==========
summary = (
    df_all.groupby("Subject")[["Delta_STD_percent", "SNR_gain_db"]]
    .mean()
    .rename(columns={"Delta_STD_percent": "Mean_Delta_STD(%)", "SNR_gain_db": "Mean_SNR(dB)"})
)
summary["有效通道比例(%)"] = (
    df_all.groupby("Subject")["Overall_Effective"].mean() * 100
).round(1)

# ========== 导出到 Excel ==========
with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
    df_all.to_excel(writer, index=False, sheet_name="All_Filter_Stats")
    summary.to_excel(writer, sheet_name="Summary_By_Subject")

print(f"✅ 汇总完成：{OUT_XLSX}")
print("包含工作表：All_Filter_Stats（明细），Summary_By_Subject（平均效果）")
