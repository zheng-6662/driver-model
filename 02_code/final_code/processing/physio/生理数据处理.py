# -*- coding: utf-8 -*-
"""
batch_physio_process_v2_final.py

閽堝 PhysioLAB Pro1(...)|CHx-XXX 鍛藉悕鏍煎紡瀹氬埗锛?
- 璇诲彇姣忎釜 CSV
- 鎸?StorageTime 鍘婚噸锛堜繚鐣欌€滄渶鍚庝竴鏉♀€濓級锛屽苟鎸夋椂闂存帓搴?
- 瀵规暟鍊煎垪鍋氭寜鏃堕棿鎻掑€?
- 鍒嗛€氶亾婊ゆ尝锛堝惈 50Hz 宸ラ闄锋尝锛?
- 缁熶竴閲嶉噰鏍峰埌 200 Hz锛岃緭鍑?raw200 + filt200
- 鎻愬彇鍩虹鐗瑰緛锛欻R銆丠RV_RMSSD锛堝彲閫夛細闇€ neurokit2锛夈€丒MG_RMS銆丷ESP_BPM/Amplitude銆丒DA_Tonic/Phasic锛堣嫢闈炲叏闆讹級
- 鐢熸垚涓€寮?4 琛屽瓙鍥剧殑鍓嶅悗瀵规瘮鍥?+ 閲忓寲缁熻琛?

渚濊禆锛?
  pip install numpy pandas scipy matplotlib neurokit2
"""

import os, re, math, warnings, traceback
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, detrend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 涓枃瀛椾綋锛堝彲閫夛級
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

# ========= 璺緞閰嶇疆 =========
INPUT_ROOT  = Path(r"F:\data_set_process\data_process\01_datasets\数据预处理\原始生理数据")    # 每个被试子文件夹
OUTPUT_ROOT = Path(r"F:\data_set_process\data_process\01_datasets\数据预处理\生理数据处理_200Hz")

# 鐩爣閲囨牱鐜?
FS_OUT = 200.0

# 婊ゆ尝鍣ㄨ缃?
ECG_BAND = (0.5, 40.0)
EDA_LOW  = 5.0
EMG_BAND = (20.0, 250.0)
RESP_LOW = 2.0
USE_NOTCH_50 = True

# EMG RMS 绐楅暱锛堢锛?
EMG_RMS_WIN = 0.25
RESP_BPM_MIN = 6      # 娆?鍒嗛挓
RESP_BPM_MAX = 30     # 娆?鍒嗛挓

# 閫氶亾鍏抽敭瀛楋紙鎸変綘瀹為檯鍛藉悕锛?
ECG_KEY = "ECG"
EMG_KEY = "EMG"
EDA_KEY = "EDA"
RESP_KEY= "RESP"

# neurokit2锛堝彲閫夛級
USE_NK = True
try:
    import neurokit2 as nk
except Exception:
    USE_NK = False
    warnings.warn("鏈娴嬪埌 neurokit2锛屽皢浣跨敤绠€鍖栫壒寰侊紙寤鸿锛歱ip install neurokit2锛?)

# ========= 灏忓伐鍏?=========
def infer_fs_from_time(tseries: pd.Series) -> float:
    t = pd.to_datetime(tseries, errors="coerce")
    t = t.dropna()
    if len(t) < 3:
        return np.nan
    dt = np.diff(t.astype("int64") / 1e9)
    dt = dt[np.isfinite(dt)]
    if len(dt) == 0:
        return np.nan
    med = np.median(dt)
    return float(1.0 / med) if med > 0 else np.nan

def butter_bandpass(sig, fs, lo, hi, order=4):
    nyq = 0.5*fs
    b, a = butter(order, [lo/nyq, hi/nyq], btype='band')
    return filtfilt(b, a, sig)

def butter_lowpass(sig, fs, cutoff, order=4):
    nyq = 0.5*fs
    b, a = butter(order, cutoff/nyq, btype='low')
    return filtfilt(b, a, sig)

def apply_notch_50(sig, fs, Q=30.0):
    b, a = iirnotch(w0=50.0/(fs/2), Q=Q)
    return filtfilt(b, a, sig)

def moving_rms(x, fs, win_s):
    n = max(1, int(round(win_s*fs)))
    x2 = x.astype(float)**2
    kernel = np.ones(n)/n
    return np.sqrt(np.convolve(x2, kernel, mode="same"))

def snr_gain_db(raw, filt):
    noise = raw - filt
    p_sig = np.nanvar(filt)
    p_noise = np.nanvar(noise)
    if p_noise <= 1e-12:
        return np.inf
    return 10.0*np.log10(p_sig/p_noise)

def resp_bpm_amp(resp_filt, fs, bpm_min=RESP_BPM_MIN, bpm_max=RESP_BPM_MAX):
    x = np.asarray(resp_filt, float)
    x = detrend(x, type="constant")
    sign = np.sign(x)
    zc = np.where(np.diff(sign) > 0)[0]
    if len(zc) > 1:
        periods = np.diff(zc) / fs
        med = np.median(periods) if len(periods) else np.nan
        bpm = 60.0/med if med and med > 0 else np.nan
    else:
        bpm = np.nan
    if not np.isnan(bpm) and (bpm < bpm_min or bpm > bpm_max):
        bpm = np.nan
    amp = np.nanmax(x) - np.nanmin(x) if len(x) else np.nan
    return bpm, amp

def resample_numeric_df(df_num: pd.DataFrame, start, end, fs_out: float) -> pd.DataFrame:
    idx_new = pd.date_range(start=start, end=end, freq=f"{int(round(1000.0/fs_out))}ms")
    out = df_num.reindex(idx_new).interpolate(method="time", limit_direction="both")
    out.index = idx_new
    return out

# ========= 涓诲鐞?=========
def process_one_csv(csv_path: Path, out_dir: Path):
    print(f"[澶勭悊] {csv_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 璇诲叆
    df0 = pd.read_csv(csv_path)
    if "StorageTime" not in df0.columns:
        raise RuntimeError(f"鏈壘鍒?StorageTime 鍒楋細{csv_path.name}")

    # 鎺ㄦ柇鍘熷閲囨牱鐜囷紙鐢ㄤ簬鏃ュ織锛?
    fs_infer = infer_fs_from_time(df0["StorageTime"])
    print(f"  閲囨牱鐜団増 {fs_infer:.2f} Hz")

    # 鏁板€煎寲骞跺幓鎺夐潪鏁板€煎垪锛堜繚鐣?StorageTime锛?
    df = df0.copy()
    for c in df.columns:
        if c != "StorageTime":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["StorageTime"])
    t = pd.to_datetime(df["StorageTime"], errors="coerce")
    df = df.loc[t.notna()].copy()
    t = pd.to_datetime(df["StorageTime"])

    # 鍘婚噸 & 鎺掑簭锛堝叧閿級
    dup_n = t.duplicated(keep="first").sum()
    if dup_n:
        print(f"  鈿狅笍 妫€娴嬪埌閲嶅鏃堕棿鎴?{dup_n} 涓紝宸插幓閲嶏紙鍙栨渶鍚庝竴鏉★級")
    df.index = t
    df = df.sort_index()
    # 鍒嗙粍鎸夋椂闂村彇鈥滄渶鍚庝竴鏉♀€濅互鍘婚噸
    num_cols = df.columns.drop(["StorageTime"], errors="ignore")
    df_num = df[num_cols].groupby(level=0).last()

    # 鎸夋椂闂存彃鍊硷紙浠ヤ究鍚庣画婊ゆ尝/閲嶉噰鏍凤級
    df_num = df_num.interpolate(method="time", limit_direction="both")

    # 閫氶亾璇嗗埆
    def find_col(key):
        for c in df_num.columns:
            if key in c:
                return c
        return None

    col_ecg = find_col(ECG_KEY)
    col_emg = find_col(EMG_KEY)
    col_eda = find_col(EDA_KEY)
    col_resp= find_col(RESP_KEY)
    print("  閫氶亾鏄犲皠锛?, { 'ECG': col_ecg, 'EMG': col_emg, 'EDA': col_eda, 'RESP': col_resp })

    # 鍘熷锛堟彃鍊煎悗锛夊簭鍒?
    ecg = df_num[col_ecg].to_numpy() if col_ecg else None
    emg = df_num[col_emg].to_numpy() if col_emg else None
    eda = df_num[col_eda].to_numpy() if col_eda else None
    resp= df_num[col_resp].to_numpy() if col_resp else None
    fs_in = fs_infer if np.isfinite(fs_infer) else 1000.0

    # 婊ゆ尝
    filt = {}
    stats_rows = []

    def add_stats(name, raw_sig, filt_sig):
        mu_b, sd_b = np.nanmean(raw_sig), np.nanstd(raw_sig)
        mu_a, sd_a = np.nanmean(filt_sig), np.nanstd(filt_sig)
        delta_std = (sd_a - sd_b)/sd_b*100.0 if sd_b > 1e-12 else np.nan
        snr_db = snr_gain_db(raw_sig, filt_sig)
        stats_rows.append({"Signal": name, "Mean_before": mu_b, "Mean_after": mu_a,
                           "STD_before": sd_b, "STD_after": sd_a,
                           "Delta_STD_percent": delta_std, "SNR_gain_db": snr_db})

    # ECG
    if ecg is not None:
        x = ecg.copy()
        if USE_NOTCH_50: x = apply_notch_50(x, fs_in)
        x = butter_bandpass(x, fs_in, *ECG_BAND)
        filt["ECG"] = x; add_stats("ECG", ecg, x)
    # EDA
    if eda is not None:
        x = butter_lowpass(eda.copy(), fs_in, EDA_LOW)
        filt["EDA"] = x; add_stats("EDA", eda, x)
    # EMG
    if emg is not None:
        x = emg.copy()
        if USE_NOTCH_50: x = apply_notch_50(x, fs_in)
        x = butter_bandpass(x, fs_in, *EMG_BAND)
        filt["EMG"] = x; add_stats("EMG", emg, x)
    # RESP
    if resp is not None:
        x = butter_lowpass(resp.copy(), fs_in, RESP_LOW)
        filt["RESP"] = x; add_stats("RESP", resp, x)

    # 鐗瑰緛
    feats = {}
    n = len(df_num)
    # ECG -> HR銆丠RV锛堝彲閫夛級
    if "ECG" in filt and USE_NK:
        try:
            processed = nk.ecg_process(filt["ECG"], sampling_rate=fs_in)
            hr = processed[0]["ECG_Rate"].to_numpy()
            # 鎻掑€煎埌 n
            feats["HR_bpm"] = np.interp(np.arange(n), np.linspace(0, n-1, len(hr)), hr)
            peaks = processed[1].get("ECG_R_Peaks", None)
            if peaks is not None:
                r_idx = np.where(peaks == 1)[0]
                if len(r_idx) > 2:
                    rr = np.diff(r_idx) / fs_in
                    rr_ms = rr * 1000.0
                    rmssd = np.sqrt(np.mean(np.square(np.diff(rr_ms)))) if len(rr_ms) > 1 else np.nan
                else:
                    rmssd = np.nan
            else:
                rmssd = np.nan
            feats["HRV_RMSSD"] = np.full(n, rmssd)
        except Exception:
            warnings.warn("neurokit2 ECG 澶勭悊澶辫触锛屽洖閫€绠€鍖?)
            feats["HR_bpm"] = np.full(n, np.nan)
            feats["HRV_RMSSD"] = np.full(n, np.nan)
    else:
        feats["HR_bpm"] = np.full(n, np.nan)
        feats["HRV_RMSSD"] = np.full(n, np.nan)

    # EDA -> Tonic/Phasic锛堣嫢涓嶆槸鎭掗浂锛?
    if "EDA" in filt and USE_NK and (np.nanstd(filt["EDA"]) > 1e-8):
        try:
            eda_sig = pd.Series(filt["EDA"]).interpolate().bfill().values
            eda_proc = nk.eda_process(eda_sig, sampling_rate=fs_in)[0]
            def align(v):
                v = v.to_numpy()
                return np.interp(np.arange(n), np.linspace(0, n-1, len(v)), v)
            feats["EDA_Tonic"]  = align(eda_proc["EDA_Tonic"])
            feats["EDA_Phasic"] = align(eda_proc["EDA_Phasic"])
        except Exception:
            warnings.warn("neurokit2 EDA 澶勭悊澶辫触锛屽洖閫€绌哄€?)
            feats["EDA_Tonic"]  = np.full(n, np.nan)
            feats["EDA_Phasic"] = np.full(n, np.nan)
    else:
        feats["EDA_Tonic"]  = np.full(n, np.nan)
        feats["EDA_Phasic"] = np.full(n, np.nan)

    # EMG -> RMS
    if "EMG" in filt:
        feats["EMG_RMS"] = moving_rms(filt["EMG"], fs_in, EMG_RMS_WIN)
    else:
        feats["EMG_RMS"] = np.full(n, np.nan)

    # RESP -> BPM / Amplitude
    if "RESP" in filt:
        bpm, amp = resp_bpm_amp(filt["RESP"], fs_in)
        feats["RESP_BPM"] = np.full(n, bpm)
        feats["RESP_Amplitude"] = np.full(n, amp)
    else:
        feats["RESP_BPM"] = np.full(n, np.nan)
        feats["RESP_Amplitude"] = np.full(n, np.nan)

    # 缁勭粐 DataFrame锛堟寜鍘婚噸鍚庤繛缁椂杞达級
    work = pd.DataFrame(index=df_num.index)
    work["StorageTime"] = df_num.index
    # raw锛堟彃鍊煎悗锛? filt锛堝師閲囨牱鐜囷級
    if col_ecg:  work["ECG_raw"]  = df_num[col_ecg].to_numpy()
    if col_emg:  work["EMG_raw"]  = df_num[col_emg].to_numpy()
    if col_eda:  work["EDA_raw"]  = df_num[col_eda].to_numpy()
    if col_resp: work["RESP_raw"] = df_num[col_resp].to_numpy()
    for k, v in filt.items():
        work[f"{k}_filt"] = v
    for k, v in feats.items():
        work[k] = v

    # 閲嶉噰鏍峰埌 200 Hz锛堟彃鍊硷級锛屽苟鐢熸垚 t_s
    start, end = work.index[0], work.index[-1]
    num_cols_rs = work.select_dtypes(include=[np.number]).columns
    out200 = resample_numeric_df(work[num_cols_rs], start, end, FS_OUT)
    out200["StorageTime"] = out200.index
    out200["t_s"] = (out200.index - out200.index[0]).total_seconds()

    # 涓轰究浜庨槄璇伙紝閲嶅懡鍚嶄负 raw200/filt200
    rename_map = {}
    for c in out200.columns:
        if c.endswith("_raw"):  rename_map[c] = c.replace("_raw", "_raw200")
        if c.endswith("_filt"): rename_map[c] = c.replace("_filt", "_filt200")
    out200 = out200.rename(columns=rename_map).reset_index(drop=True)

    # 杈撳嚭璺緞
    stem = csv_path.stem
    clean_stem = re.sub(r"\s*\(.*?\)", "", stem).strip()
    out_csv  = out_dir / f"{clean_stem}_physio_cleaned_200Hz.csv"
    out_png  = out_dir / f"{clean_stem}_physio_filtercheck.png"
    out_stat = out_dir / f"{clean_stem}_physio_filterstats.csv"

    # 淇濆瓨鏁版嵁涓庣粺璁?
    out200.to_csv(out_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(stats_rows).to_csv(out_stat, index=False, encoding="utf-8-sig")

    # 鐢诲ぇ鍥?
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    axes = axes.flatten()
    t_in = (work.index - work.index[0]).total_seconds()

    def plot_pair(ax, key, title):
        raw_c  = f"{key}_raw"
        filt_c = f"{key}_filt"
        if raw_c in work.columns:
            ax.plot(t_in, work[raw_c], alpha=0.7, label=f"{key} raw")
        if filt_c in work.columns:
            ax.plot(t_in, work[filt_c], alpha=0.9, label=f"{key} filt")
        ax.set_title(f"{title}锛堝墠鍚庡姣旓級")
        ax.grid(True, alpha=0.3); ax.legend(loc="upper right")

    plot_pair(axes[0], "ECG",  "ECG")
    plot_pair(axes[1], "EDA",  "EDA")
    plot_pair(axes[2], "EMG",  "EMG")
    plot_pair(axes[3], "RESP", "RESP")
    axes[-1].set_xlabel(f"鏃堕棿锛堢锛?| 鍘熷鈮坽fs_infer:.2f} Hz 鈫?杈撳嚭锛歿FS_OUT:.0f} Hz")
    fig.suptitle(f"{clean_stem} 鐢熺悊淇″彿婊ゆ尝鍓嶅悗瀵规瘮", y=0.98)
    fig.tight_layout(rect=[0,0,1,0.96])
    fig.savefig(out_png, dpi=150); plt.close(fig)

    print(f"  鉁?杈撳嚭锛歕n    {out_csv}\n    {out_png}\n    {out_stat}")

def main():
    try:
        subs = [p for p in INPUT_ROOT.iterdir() if p.is_dir()]
        if not subs: subs = [INPUT_ROOT]
        for sub in subs:
            csvs = list(sub.glob("*.csv")) or list(sub.rglob("*.csv"))
            if not csvs:
                print(f"[璺宠繃] {sub} 鏃?CSV")
                continue
            out_dir = OUTPUT_ROOT / sub.name
            for f in csvs:
                try:
                    process_one_csv(f, out_dir)
                except Exception:
                    print(f"鉂?澶勭悊澶辫触锛歿f}\n{traceback.format_exc()}")
    except Exception:
        print("杩愯澶辫触锛歕n", traceback.format_exc())

if __name__ == "__main__":
    main()
