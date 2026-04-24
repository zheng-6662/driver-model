# -*- coding: utf-8 -*-
"""
reclean_physio_selected_v1.py
-----------------------------------
鍔熻兘锛?
  1. 鑷姩鎵弿鎵€鏈夎璇曠殑 *_physio_filterstats.csv
  2. 鏍规嵁涓ユ牸瑙勫垯鍒ゅ畾鏄惁闇€瑕佸 EDA / RESP 閲嶆礂
  3. 浠呴噸娲楅渶瑕佺殑閫氶亾锛屼笉閲嶅璁＄畻 ECG 鍜?EMG
  4. 杈撳嚭鏂扮増鏈細
      *_physio_reclean_200Hz.csv
      *_reclean_filterstats.csv
      *_reclean_filtercheck.png
  5. 涓嶈鐩栨棫鏂囦欢锛屼究浜庡姣?

渚濊禆锛?
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
# 鐩綍閰嶇疆锛堣淇敼涓轰綘鐨勮矾寰勶級
# ========================================================
ROOT_IN = Path(r"F:\data_set_process\data_process\01_datasets\数据预处理\生理数据处理_200Hz")
ROOT_OUT = Path(r"F:\data_set_process\data_process\01_datasets\数据预处理\生理数据处理_200Hz_reclean_v2")

FS_OUT = 200.0   # 杈撳嚭閲囨牱鐜?

# ========================================================
# 婊ゆ尝鍣?
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
# 鍒ゆ柇鏄惁闇€瑕侀噸娲?
# ========================================================
def should_reclean(signal_name, row_stats, clean_df):

    if signal_name == "EDA":
        std_raw = np.nanstd(clean_df["EDA_raw200"])
        if row_stats["Delta_STD_percent"] > -10:     # 骞虫粦搴︿笉瓒?
            return True
        if row_stats["SNR_gain_db"] < 5:             # SNR 鏀瑰杽涓嶈冻
            return True
        if std_raw < 1e-3:                           # 鍑犱箮涓€鏉＄洿绾?
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

    # ECG / EMG 涓嶉噸娲?
    return False


# ========================================================
# 閲嶆礂鍗曚釜閫氶亾
# ========================================================
def reclean_channel(raw200, fs, sig_type):
    if sig_type == "EDA":
        # 鏇翠弗鏍煎幓鍣?
        sig = butter_low(raw200, fs, cutoff=3.0)
        sig = detrend(sig, type="constant")
        return sig

    if sig_type == "RESP":
        sig = butter_low(raw200, fs, cutoff=2.0)
        sig = detrend(sig, type="constant")
        return sig

    return raw200


# ========================================================
# 涓诲嚱鏁帮細澶勭悊鍗曚釜鏂囦欢
# ========================================================
def process_one(st_filtstats, st_clean, out_dir):

    df_stats = pd.read_csv(st_filtstats)
    df_clean = pd.read_csv(st_clean)

    out_dir.mkdir(parents=True, exist_ok=True)

    # 鍒ゅ畾閲嶆礂鐨勯€氶亾
    reclean_list = []
    for _, row in df_stats.iterrows():
        sig = row["Signal"]
        if should_reclean(sig, row, df_clean):
            reclean_list.append(sig)

    print(f"闇€瑕侀噸娲楅€氶亾锛歿reclean_list}")

    # 濡傛灉娌℃湁瑕侀噸娲楃殑
    if len(reclean_list) == 0:
        print("鏃犻€氶亾闇€瑕侀噸娲楋紝鐩存帴澶嶅埗鍘熺粨鏋溿€?)
        df_clean.to_csv(out_dir / (st_clean.stem+"_reclean_200Hz.csv"),
                        index=False, encoding="utf-8-sig")
        return

    # 澶嶅埗鍘熷鏁版嵁
    df_out = df_clean.copy()

    # 瀵归渶瑕侀噸娲楃殑閫氶亾閲嶆柊婊ゆ尝
    for sig in reclean_list:
        raw_col = f"{sig}_raw200"
        if raw_col not in df_out.columns:
            continue
        print(f"閲嶆礂锛歿sig}")

        raw = df_out[raw_col].to_numpy()
        filt_new = reclean_channel(raw, FS_OUT, sig)
        df_out[f"{sig}_filt200"] = filt_new

    # 淇濆瓨鏂扮増鏈?
    save_csv = out_dir / (st_clean.stem + "_reclean_200Hz.csv")
    df_out.to_csv(save_csv, index=False, encoding="utf-8-sig")

    # 杈撳嚭妫€鏌ュ浘
    plt.figure(figsize=(15,8))
    t = df_out["t_s"]
    for i, sig in enumerate(reclean_list):
        plt.subplot(len(reclean_list), 1, i+1)
        plt.plot(t, df_clean[f"{sig}_filt200"], label="鏃ф护娉?, alpha=0.6)
        plt.plot(t, df_out[f"{sig}_filt200"], label="鏂版护娉?, alpha=0.9)
        plt.title(f"{sig} 閲嶆礂鍓嶅悗")
        plt.legend()
        plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / (st_clean.stem + "_reclean_filtercheck.png"), dpi=150)
    plt.close()

    print(f"閲嶆礂瀹屾垚 鈫?{save_csv}")


# ========================================================
# 鎵归噺杩愯
# ========================================================
def main():

    subs = [p for p in ROOT_IN.iterdir() if p.is_dir()]
    if not subs:
        subs = [ROOT_IN]

    for sub in subs:
        print(f"\n=== 澶勭悊琚瘯锛歿sub.name} ===")

        filtstats_files = list(sub.glob("*_physio_filterstats.csv"))
        clean_files = list(sub.glob("*_physio_cleaned_200Hz.csv"))
        if not filtstats_files or not clean_files:
            print("缂哄皯蹇呰鏂囦欢锛岃烦杩囥€?)
            continue

        out_dir = ROOT_OUT / sub.name

        for fs, cl in zip(filtstats_files, clean_files):
            try:
                process_one(fs, cl, out_dir)
            except Exception:
                print(f"鉂?澶勭悊澶辫触锛歿fs}\n{traceback.format_exc()}")

    print("\n鍏ㄩ儴澶勭悊瀹屾垚銆?)


if __name__ == "__main__":
    main()
