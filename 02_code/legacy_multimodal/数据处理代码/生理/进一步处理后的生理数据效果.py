# -*- coding: utf-8 -*-
"""
reclean_physio_quality_summary_v2.py
--------------------------------------
功能：
  - 扫描所有被试的 *reclean_200Hz.csv
  - 自动重新计算：均值、标准差、ΔSTD%、SNR 增益
  - 并生成：*_reclean_filterstats.csv
  - 最后汇总所有被试结果 → reclean_filter_summary_ALL.xlsx

依赖：
  pip install numpy pandas scipy openpyxl
"""

import numpy as np
import pandas as pd
from scipy.signal import filtfilt
from pathlib import Path
import warnings

# ==============================
# 路径配置（改成你的路径）
# ==============================
ROOT_DIR = Path(r"F:\数据集处理\datasetprocess\数据预处理\生理数据处理_200Hz_reclean_v2")
OUT_XLSX = ROOT_DIR / "reclean_filter_summary_ALL.xlsx"

# ==============================
# 计算 SNR 增益
# ==============================
def snr_gain_db(raw, filt):
    noise = raw - filt
    p_sig = np.nanvar(filt)
    p_noise = np.nanvar(noise)
    if p_noise < 1e-12:
        return np.inf
    return 10 * np.log10(p_sig / p_noise)

# ==============================
# 对单个文件生成 stats
# ==============================
def compute_stats_for_file(csv_path: Path):

    df = pd.read_csv(csv_path)
    cols = df.columns

    stats_rows = []
    signals = ["ECG", "EDA", "EMG", "RESP"]

    for sig in signals:
        raw_col = f"{sig}_raw200"
        filt_col = f"{sig}_filt200"

        if raw_col not in cols or filt_col not in cols:
            continue

        raw = df[raw_col].to_numpy()
        filt = df[filt_col].to_numpy()

        mu_b, sd_b = np.nanmean(raw), np.nanstd(raw)
        mu_a, sd_a = np.nanmean(filt), np.nanstd(filt)
        delta_std = (sd_a - sd_b) / sd_b * 100 if sd_b > 1e-12 else np.nan
        snr_db = snr_gain_db(raw, filt)

        stats_rows.append({
            "File": csv_path.name,
            "Signal": sig,
            "Mean_before": mu_b,
            "Mean_after": mu_a,
            "STD_before": sd_b,
            "STD_after": sd_a,
            "Delta_STD_percent": delta_std,
            "SNR_gain_db": snr_db
        })

    df_stats = pd.DataFrame(stats_rows)
    out_csv = csv_path.with_name(csv_path.stem.replace("_reclean_200Hz", "_reclean_filterstats") + ".csv")
    df_stats.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"[√] 写入重洗 stats → {out_csv.name}")
    return df_stats


# ==============================
# 批量处理
# ==============================
def main():
    all_stats = []

    subs = [p for p in ROOT_DIR.iterdir() if p.is_dir()]
    if not subs:
        subs = [ROOT_DIR]

    for sub in subs:
        print(f"\n=== 处理被试：{sub.name} ===")

        reclean_files = list(sub.glob("*_reclean_200Hz.csv"))
        if not reclean_files:
            print("无 reclean 文件，跳过。")
            continue

        for csv in reclean_files:
            df_stats = compute_stats_for_file(csv)
            df_stats["Subject"] = sub.name
            all_stats.append(df_stats)

    if not all_stats:
        print("⚠️ 未找到任何 reclean 文件。")
        return

    df_all = pd.concat(all_stats, ignore_index=True)

    # 计算每个被试平均性能
    summary = (
        df_all.groupby("Subject")[["Delta_STD_percent", "SNR_gain_db"]]
        .mean()
        .rename(columns={
            "Delta_STD_percent": "Mean_Delta_STD(%)",
            "SNR_gain_db": "Mean_SNR(dB)"
        })
    )

    # 写出汇总 Excel
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        df_all.to_excel(writer, index=False, sheet_name="Reclean_Filter_Stats")
        summary.to_excel(writer, sheet_name="Summary_By_Subject")

    print(f"\n🎉 汇总完成 → {OUT_XLSX}")


if __name__ == "__main__":
    main()
