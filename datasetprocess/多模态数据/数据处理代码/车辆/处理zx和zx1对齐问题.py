# -*- coding: utf-8 -*-
"""
vehicle_align_batch_v3.py
=====================================================
批量车辆数据处理（最终版本）
功能：
1. 批量读取原始车辆数据
2. 删除 zx / zx1 空白行
3. 删除 zx / zx1 中整列为空的列
4. 若 zx 与 zx1 行数差 1 行 → 删除最后一行
5. 优先使用 zx 的 StorageTime
6. 合并 zx + zx1
7. 合并后再次删除整列为空的列
8. 自动分析采样频率
9. 输出到正式目录
=====================================================
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

# -------------------------
# 路径设置
# -------------------------
RAW_ROOT = Path(r"F:\数据集处理\data_process\datasetprocess\数据预处理\原始车辆数据")
OUT_ROOT = Path(r"F:\数据集处理\data_process\datasetprocess\数据预处理\车辆数据处理\正式车辆数据处理")
OUT_ROOT.mkdir(parents=True, exist_ok=True)


def analyze_sampling_rate(df, subject, fname):
    """分析采样频率（基于 StorageTime）"""
    if "StorageTime" not in df.columns:
        print("  ⚠ 无 StorageTime，无法分析采样频率")
        return

    t = pd.to_datetime(df["StorageTime"], errors="coerce")
    dt = (t.diff().dt.total_seconds()).dropna()

    if len(dt) == 0:
        print("  ⚠ 时间戳不足，无法分析采样率")
        return

    mean_dt = dt.mean()
    freq = 1 / mean_dt if mean_dt > 0 else np.nan

    print(f"  === 采样频率分析 [{subject} - {fname}] ===")
    print(f"  平均采样间隔： {mean_dt:.6f} 秒")
    print(f"  平均采样频率： {freq:.2f} Hz")
    print(f"  最小 Δt：       {dt.min():.6f}")
    print(f"  最大 Δt：       {dt.max():.6f}")
    print(f"  Δt 标准差：     {dt.std():.6f}")
    print("  ======================================")


print("=== 批处理开始 ===")

# -------------------------
# 批量处理每个被试
# -------------------------
for subj in RAW_ROOT.iterdir():
    if not subj.is_dir():
        continue

    subj_name = subj.name
    print(f"\n>>> 处理被试：{subj_name}")

    out_dir = OUT_ROOT / subj_name
    out_dir.mkdir(exist_ok=True)

    csv_files = list(subj.glob("*_vehicle.csv"))
    if len(csv_files) == 0:
        print(f"⚠ 未找到车辆文件，跳过：{subj_name}")
        continue

    for f_csv in csv_files:
        print(f"—— 处理文件：{f_csv.name}")

        df = pd.read_csv(f_csv)

        # 找 zx 与 zx1 列
        zx_cols = [c for c in df.columns if c.startswith("zx|")]
        zx1_cols = [c for c in df.columns if c.startswith("zx1|")]

        # 标记有效行
        df["zx_valid"] = df[zx_cols].notna().any(axis=1)
        df["zx1_valid"] = df[zx1_cols].notna().any(axis=1)

        # 提取有效行 & 删除空白列
        df_zx = df[df["zx_valid"]].reset_index(drop=True)
        df_zx1 = df[df["zx1_valid"]].reset_index(drop=True)

        df_zx = df_zx[zx_cols + ["StorageTime"]].dropna(axis=1, how="all")
        df_zx1 = df_zx1[zx1_cols + ["StorageTime"]].dropna(axis=1, how="all")

        print(f"  zx有效行数：  {len(df_zx)}")
        print(f"  zx1有效行数： {len(df_zx1)}")

        # 长度差一行 → 删除多余行
        if len(df_zx) > len(df_zx1):
            df_zx = df_zx.iloc[:len(df_zx1)]
            print("  → 删除 zx 最后一行")
        elif len(df_zx1) > len(df_zx):
            df_zx1 = df_zx1.iloc[:len(df_zx)]
            print("  → 删除 zx1 最后一行")

        # ---- 保留唯一 StorageTime ----
        storage_time = df_zx["StorageTime"].reset_index(drop=True)
        df_zx = df_zx.drop(columns=["StorageTime"], errors="ignore")
        df_zx1 = df_zx1.drop(columns=["StorageTime"], errors="ignore")

        # ---- 合并 ----
        df_merged = pd.concat(
            [
                storage_time,
                df_zx.reset_index(drop=True),
                df_zx1.reset_index(drop=True)
            ],
            axis=1
        )

        # ---- 合并后删除空白列 ----
        df_merged = df_merged.dropna(axis=1, how="all")

        # 输出文件
        out_file = out_dir / f"{f_csv.stem}_aligned_cleaned.csv"
        df_merged.to_csv(out_file, index=False, encoding="utf-8-sig")
        print(f"  ✔ 已输出：{out_file}")

        # 分析采样频率
        analyze_sampling_rate(df_merged, subj_name, f_csv.name)

print("\n=== 批处理完成 ===")
