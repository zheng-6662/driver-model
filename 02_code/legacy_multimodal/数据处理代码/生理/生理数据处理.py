# -*- coding: utf-8 -*-
"""
batch_physio_process_v2_final.py

针对 PhysioLAB Pro1(...)|CHx-XXX 命名格式定制：
- 读取每个 CSV
- 按 StorageTime 去重（保留“最后一条”），并按时间排序
- 对数值列做按时间插值
- 分通道滤波（含 50Hz 工频陷波）
- 统一重采样到 200 Hz，输出 raw200 + filt200
- 提取基础特征：HR、HRV_RMSSD（可选：需 neurokit2）、EMG_RMS、RESP_BPM/Amplitude、EDA_Tonic/Phasic（若非全零）
- 生成一张 4 行子图的前后对比图 + 量化统计表

依赖：
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

# 中文字体（可选）
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

# ========= 路径配置 =========
INPUT_ROOT  = Path(r"F:\数据集处理\datasetprocess\数据预处理\原始生理数据")    # 每个被试子文件夹
OUTPUT_ROOT = Path(r"F:\数据集处理\datasetprocess\数据预处理\生理数据处理_200Hz")

# 目标采样率
FS_OUT = 200.0

# 滤波器设置
ECG_BAND = (0.5, 40.0)
EDA_LOW  = 5.0
EMG_BAND = (20.0, 250.0)
RESP_LOW = 2.0
USE_NOTCH_50 = True

# EMG RMS 窗长（秒）
EMG_RMS_WIN = 0.25
RESP_BPM_MIN = 6      # 次/分钟
RESP_BPM_MAX = 30     # 次/分钟

# 通道关键字（按你实际命名）
ECG_KEY = "ECG"
EMG_KEY = "EMG"
EDA_KEY = "EDA"
RESP_KEY= "RESP"

# neurokit2（可选）
USE_NK = True
try:
    import neurokit2 as nk
except Exception:
    USE_NK = False
    warnings.warn("未检测到 neurokit2，将使用简化特征（建议：pip install neurokit2）")

# ========= 小工具 =========
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

# ========= 主处理 =========
def process_one_csv(csv_path: Path, out_dir: Path):
    print(f"[处理] {csv_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 读入
    df0 = pd.read_csv(csv_path)
    if "StorageTime" not in df0.columns:
        raise RuntimeError(f"未找到 StorageTime 列：{csv_path.name}")

    # 推断原始采样率（用于日志）
    fs_infer = infer_fs_from_time(df0["StorageTime"])
    print(f"  采样率≈ {fs_infer:.2f} Hz")

    # 数值化并去掉非数值列（保留 StorageTime）
    df = df0.copy()
    for c in df.columns:
        if c != "StorageTime":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["StorageTime"])
    t = pd.to_datetime(df["StorageTime"], errors="coerce")
    df = df.loc[t.notna()].copy()
    t = pd.to_datetime(df["StorageTime"])

    # 去重 & 排序（关键）
    dup_n = t.duplicated(keep="first").sum()
    if dup_n:
        print(f"  ⚠️ 检测到重复时间戳 {dup_n} 个，已去重（取最后一条）")
    df.index = t
    df = df.sort_index()
    # 分组按时间取“最后一条”以去重
    num_cols = df.columns.drop(["StorageTime"], errors="ignore")
    df_num = df[num_cols].groupby(level=0).last()

    # 按时间插值（以便后续滤波/重采样）
    df_num = df_num.interpolate(method="time", limit_direction="both")

    # 通道识别
    def find_col(key):
        for c in df_num.columns:
            if key in c:
                return c
        return None

    col_ecg = find_col(ECG_KEY)
    col_emg = find_col(EMG_KEY)
    col_eda = find_col(EDA_KEY)
    col_resp= find_col(RESP_KEY)
    print("  通道映射：", { 'ECG': col_ecg, 'EMG': col_emg, 'EDA': col_eda, 'RESP': col_resp })

    # 原始（插值后）序列
    ecg = df_num[col_ecg].to_numpy() if col_ecg else None
    emg = df_num[col_emg].to_numpy() if col_emg else None
    eda = df_num[col_eda].to_numpy() if col_eda else None
    resp= df_num[col_resp].to_numpy() if col_resp else None
    fs_in = fs_infer if np.isfinite(fs_infer) else 1000.0

    # 滤波
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

    # 特征
    feats = {}
    n = len(df_num)
    # ECG -> HR、HRV（可选）
    if "ECG" in filt and USE_NK:
        try:
            processed = nk.ecg_process(filt["ECG"], sampling_rate=fs_in)
            hr = processed[0]["ECG_Rate"].to_numpy()
            # 插值到 n
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
            warnings.warn("neurokit2 ECG 处理失败，回退简化")
            feats["HR_bpm"] = np.full(n, np.nan)
            feats["HRV_RMSSD"] = np.full(n, np.nan)
    else:
        feats["HR_bpm"] = np.full(n, np.nan)
        feats["HRV_RMSSD"] = np.full(n, np.nan)

    # EDA -> Tonic/Phasic（若不是恒零）
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
            warnings.warn("neurokit2 EDA 处理失败，回退空值")
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

    # 组织 DataFrame（按去重后连续时轴）
    work = pd.DataFrame(index=df_num.index)
    work["StorageTime"] = df_num.index
    # raw（插值后）+ filt（原采样率）
    if col_ecg:  work["ECG_raw"]  = df_num[col_ecg].to_numpy()
    if col_emg:  work["EMG_raw"]  = df_num[col_emg].to_numpy()
    if col_eda:  work["EDA_raw"]  = df_num[col_eda].to_numpy()
    if col_resp: work["RESP_raw"] = df_num[col_resp].to_numpy()
    for k, v in filt.items():
        work[f"{k}_filt"] = v
    for k, v in feats.items():
        work[k] = v

    # 重采样到 200 Hz（插值），并生成 t_s
    start, end = work.index[0], work.index[-1]
    num_cols_rs = work.select_dtypes(include=[np.number]).columns
    out200 = resample_numeric_df(work[num_cols_rs], start, end, FS_OUT)
    out200["StorageTime"] = out200.index
    out200["t_s"] = (out200.index - out200.index[0]).total_seconds()

    # 为便于阅读，重命名为 raw200/filt200
    rename_map = {}
    for c in out200.columns:
        if c.endswith("_raw"):  rename_map[c] = c.replace("_raw", "_raw200")
        if c.endswith("_filt"): rename_map[c] = c.replace("_filt", "_filt200")
    out200 = out200.rename(columns=rename_map).reset_index(drop=True)

    # 输出路径
    stem = csv_path.stem
    clean_stem = re.sub(r"\s*\(.*?\)", "", stem).strip()
    out_csv  = out_dir / f"{clean_stem}_physio_cleaned_200Hz.csv"
    out_png  = out_dir / f"{clean_stem}_physio_filtercheck.png"
    out_stat = out_dir / f"{clean_stem}_physio_filterstats.csv"

    # 保存数据与统计
    out200.to_csv(out_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(stats_rows).to_csv(out_stat, index=False, encoding="utf-8-sig")

    # 画大图
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
        ax.set_title(f"{title}（前后对比）")
        ax.grid(True, alpha=0.3); ax.legend(loc="upper right")

    plot_pair(axes[0], "ECG",  "ECG")
    plot_pair(axes[1], "EDA",  "EDA")
    plot_pair(axes[2], "EMG",  "EMG")
    plot_pair(axes[3], "RESP", "RESP")
    axes[-1].set_xlabel(f"时间（秒） | 原始≈{fs_infer:.2f} Hz → 输出：{FS_OUT:.0f} Hz")
    fig.suptitle(f"{clean_stem} 生理信号滤波前后对比", y=0.98)
    fig.tight_layout(rect=[0,0,1,0.96])
    fig.savefig(out_png, dpi=150); plt.close(fig)

    print(f"  ✅ 输出：\n    {out_csv}\n    {out_png}\n    {out_stat}")

def main():
    try:
        subs = [p for p in INPUT_ROOT.iterdir() if p.is_dir()]
        if not subs: subs = [INPUT_ROOT]
        for sub in subs:
            csvs = list(sub.glob("*.csv")) or list(sub.rglob("*.csv"))
            if not csvs:
                print(f"[跳过] {sub} 无 CSV")
                continue
            out_dir = OUTPUT_ROOT / sub.name
            for f in csvs:
                try:
                    process_one_csv(f, out_dir)
                except Exception:
                    print(f"❌ 处理失败：{f}\n{traceback.format_exc()}")
    except Exception:
        print("运行失败：\n", traceback.format_exc())

if __name__ == "__main__":
    main()
