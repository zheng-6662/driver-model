# -*- coding: utf-8 -*-
import os
import pandas as pd
from glob import glob

ROOT = r"F:\数据集处理\data_process\datasetprocess\数据预处理\车辆数据处理\正式车辆数据处理"


# =======================
# 你提供的时间生成函数（我只略微增强了安全性）
# =======================
def get_ts(df):
    # 如果已有 t_s，直接使用
    if "t_s" in df.columns:
        return pd.to_numeric(df["t_s"], errors="coerce").values

    # 优先使用 StorageTime
    if "StorageTime" in df.columns:
        # 先尝试当时间戳解析（支持 yyyy-mm-dd HH:MM:SS.ms）
        t_raw = pd.to_datetime(df["StorageTime"], errors="coerce")

        # 如果解析成功（不是全部NaN）
        if not t_raw.isna().all():
            return (t_raw - t_raw.iloc[0]).dt.total_seconds().values

        # 否则作为数值（毫秒 → 秒）
        t_num = pd.to_numeric(df["StorageTime"], errors="coerce")
        if not t_num.isna().all():
            t0 = t_num.iloc[0]
            return (t_num - t0) / 1000.0

    # 最后兜底：使用第一列并尝试数字化
    print("⚠ 使用第一列作为时间基准")
    col0 = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    t0 = col0.iloc[0]
    return (col0 - t0).values


# =======================
# 批处理
# =======================
def process_file(fpath):
    print(f"🔧 正在处理：{fpath}")

    df = pd.read_csv(fpath)

    # 生成 t_s
    t_s = get_ts(df)

    # 添加到 DF
    df["t_s"] = t_s

    # 让 t_s 在最前面
    cols = ["t_s"] + [c for c in df.columns if c != "t_s"]
    df = df[cols]

    # 覆盖写回
    df.to_csv(fpath, index=False)
    print("   ✅ 已添加 t_s 并覆盖写回\n")


def main():
    count = 0

    for root, dirs, files in os.walk(ROOT):
        for fname in files:
            if fname.endswith("_vehicle_aligned_cleaned.csv"):
                fpath = os.path.join(root, fname)
                try:
                    process_file(fpath)
                    count += 1
                except Exception as e:
                    print(f"   ❌ 出错：{e}\n")

    print(f"🎉 完成！共处理 {count} 个文件")


if __name__ == "__main__":
    main()
