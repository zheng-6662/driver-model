# -*- coding: utf-8 -*-
"""
multimodal_feature_extractor_batch_v3.py
==================================================
《支持每个被试多个实验版本》
- 完全保留你原有逻辑
- 每个被试可以有多个 event/vehicle/physio/eeg 文件
- 对每个 event_*.csv 逐一生成对应特征文件
- 不会跳过剩余实验
"""

import os
import numpy as np
import pandas as pd
from scipy.signal import welch
import mne
from glob import glob
import warnings

warnings.filterwarnings("ignore")

# ======================
# 批量路径
# ======================
ROOT = r"F:\数据集处理\data_process\datasetprocess\多模态数据\被试数据集合"

SUBJECT_LIST = [
    "byx","cwh","gf","gzj","hzh","jy",
    "lx","lxy","rjy","txj","tyy","xst",
    "yyl","yzy","zdq","zt","zx","zxy"
]

# ======================
# 辅助函数（不改你逻辑）
# ======================
def find_files(folder, pattern):
    if not os.path.exists(folder):
        return []
    return glob(os.path.join(folder, pattern))

def safe_load_csv(path):
    if path is None or not os.path.exists(path):
        return None
    return pd.read_csv(path)

def get_idx_nearest_t(df, t0, t_col="t_s"):
    t = df[t_col].values
    return int(np.argmin(np.abs(t - t0)))

def compute_bandpower(psd, freqs, fmin, fmax):
    mask = (freqs >= fmin) & (freqs < fmax)
    if not np.any(mask):
        return np.nan
    return np.trapz(psd[..., mask], freqs[mask], axis=-1)

# -------------------------
# 【完全保留你原 EEG 特征代码】
# -------------------------
def eeg_extract_features(raw, start_s, end_s):
    feats = {
        "EEG_theta": np.nan, "EEG_alpha": np.nan, "EEG_beta": np.nan,
        "EEG_delta": np.nan, "EEG_gamma": np.nan,
        "EEG_FmTheta": np.nan, "EEG_OccAlpha": np.nan,
        "EEG_Theta_Alpha_ratio": np.nan, "EEG_Beta_Alpha_ratio": np.nan,
        "EEG_EngagementIndex": np.nan,
    }
    if raw is None:
        return feats

    sf = raw.info["sfreq"]
    tmin = max(0, start_s)
    tmax = min(end_s, raw.times[-1])
    if tmax - tmin <= 0.2:
        return feats

    seg = raw.copy().crop(tmin=tmin, tmax=tmax)
    data = seg.get_data()

    nperseg = min(256, data.shape[1])
    if nperseg < 32:
        return feats

    psd_list = []
    for ch in data:
        f, p = welch(ch, fs=sf, nperseg=nperseg)
        psd_list.append(p)
    psd = np.vstack(psd_list)

    theta = compute_bandpower(psd, f, 4, 8)
    alpha = compute_bandpower(psd, f, 8, 12)
    beta  = compute_bandpower(psd, f,12, 30)
    delta = compute_bandpower(psd, f, 1,  4)
    gamma = compute_bandpower(psd, f,30, 45)

    # 全部保持你原来的逻辑
    ch_names = np.array(seg.ch_names)
    frontal = np.isin(ch_names,
        ["Fz","F3","F4","F7","F8","Fp1","Fp2","FC1","FC2","FC5","FC6","AF3","AF4"])
    occipital = np.isin(ch_names, ["O1","O2","Oz","PO3","PO4"])

    fm_theta = np.nanmean(theta[frontal]) if np.any(frontal) else np.nan
    occ_alpha = np.nanmean(alpha[occipital]) if np.any(occipital) else np.nan

    theta_g = np.nanmean(theta)
    alpha_g = np.nanmean(alpha)
    beta_g = np.nanmean(beta)
    delta_g = np.nanmean(delta)
    gamma_g = np.nanmean(gamma)

    feats.update({
        "EEG_theta": theta_g, "EEG_alpha": alpha_g, "EEG_beta": beta_g,
        "EEG_delta": delta_g, "EEG_gamma": gamma_g,
        "EEG_FmTheta": fm_theta, "EEG_OccAlpha": occ_alpha,
        "EEG_Theta_Alpha_ratio": theta_g/alpha_g if alpha_g!=0 else np.nan,
        "EEG_Beta_Alpha_ratio": beta_g/alpha_g if alpha_g!=0 else np.nan,
        "EEG_EngagementIndex": beta_g/(alpha_g+theta_g) if alpha_g+theta_g!=0 else np.nan,
    })

    return feats

# -------------------------
# 【完全保留你的生理特征函数】
# -------------------------
def physio_extract_features(df_phy, start_s, end_s):
    feats = {
        "HR_mean": np.nan, "HR_std": np.nan,
        "EMG_RMS_mean": np.nan, "EMG_RMS_max": np.nan,
        "RESP_BPM_mean": np.nan, "RESP_Amplitude_mean": np.nan,
        "ECG_valid": 0, "EMG_valid": 0, "RESP_valid": 0, "EDA_valid": 0,
    }

    if df_phy is None:
        return feats
    mask = (df_phy["t_s"] >= start_s) & (df_phy["t_s"] <= end_s)
    seg = df_phy.loc[mask]
    if len(seg) < 10:
        return feats

    if "HR_bpm" in seg:
        hr = seg["HR_bpm"]
        if np.isfinite(hr).sum()>5:
            feats["HR_mean"] = np.nanmean(hr)
            feats["HR_std"] = np.nanstd(hr)
            feats["ECG_valid"] = 1

    if "EMG_RMS" in seg:
        emg = seg["EMG_RMS"]
        if np.isfinite(emg).sum()>5:
            feats["EMG_RMS_mean"] = np.nanmean(emg)
            feats["EMG_RMS_max"] = np.nanmax(emg)
            feats["EMG_valid"] = 1

    if "RESP_BPM" in seg and "RESP_Amplitude" in seg:
        resp_bpm = seg["RESP_BPM"]
        resp_amp = seg["RESP_Amplitude"]
        if np.isfinite(resp_amp).sum()>5:
            feats["RESP_BPM_mean"] = np.nanmean(resp_bpm)
            feats["RESP_Amplitude_mean"] = np.nanmean(resp_amp)
            feats["RESP_valid"] = 1

    return feats

# -------------------------
# 【完全保留你的车辆瞬时状态函数】
# -------------------------
def vehicle_state_at_start(df_veh, t0):
    state = {k: np.nan for k in [
        "v_kmh_start","vx_start","ax_start","ay_start",
        "roll_start","pitch_start","yaw_start",
        "steer_start","lateraldistance_start","lanecurvature_start"
    ]}

    if df_veh is None:
        return state

    idx = get_idx_nearest_t(df_veh, t0)
    row = df_veh.iloc[idx]

    def get(col):
        return float(row[col]) if col in df_veh else np.nan

    state["v_kmh_start"] = get("zx1|v_km/h")
    state["vx_start"] = get("zx|vx")
    state["ax_start"] = get("zx|ax")
    state["ay_start"] = get("zx|ay")
    state["roll_start"] = get("zx|roll")
    state["pitch_start"] = get("zx|pitch")
    state["yaw_start"] = get("zx|yaw")
    state["steer_start"] = get("zx|SteeringWheel")
    state["lateraldistance_start"] = get("zx1|lateraldistance")
    state["lanecurvature_start"] = get("zx1|lanecurvatureXY")

    return state


# ============================================================
# 主流程：处理一个被试的所有实验
# ============================================================

def process_subject(sub):

    print(f"\n======= 处理被试 {sub} =======")

    event_dir = os.path.join(ROOT, sub, "event")
    vehicle_dir = os.path.join(ROOT, sub, "vehicle")
    physio_dir = os.path.join(ROOT, sub, "physio")
    eeg_dir = os.path.join(ROOT, sub, "eeg")

    event_files = find_files(event_dir, "*_vehicle_aligned_cleaned_events_v312.csv")

    if len(event_files) == 0:
        print("❌ 没有事件文件，跳过此被试")
        return

    # —— 每个事件文件 = 一次实验 —— #
    for evt_path in event_files:

        print(f"\n→ 处理实验：{os.path.basename(evt_path)}")

        # 匹配同实验名称的 vehicle/physio/eeg 文件
        prefix = os.path.basename(evt_path).split("_vehicle_aligned_cleaned_events")[0]

        veh_path = find_files(vehicle_dir, prefix + "_vehicle_aligned_cleaned.csv")
        phy_path = find_files(physio_dir, prefix + "_physio_physio_cleaned_200Hz_reclean_200Hz.csv")
        eeg_path = find_files(eeg_dir, prefix + "_eeg_raw_clean_resamp200_ica_final.fif")

        veh_path = veh_path[0] if veh_path else None
        phy_path = phy_path[0] if phy_path else None
        eeg_path = eeg_path[0] if eeg_path else None

        # 读取
        df_evt = safe_load_csv(evt_path)
        df_veh = safe_load_csv(veh_path)
        df_phy = safe_load_csv(phy_path)

        if eeg_path:
            raw_eeg = mne.io.read_raw_fif(eeg_path, preload=False)
        else:
            raw_eeg = None

        records = []

        for i, row_evt in df_evt.iterrows():
            start_s = row_evt["start_s"]
            end_s = row_evt["end_s"]

            rec = {}

            # 基本信息
            rec["subject_id"] = sub
            rec["event_index"] = i
            rec["start_s"] = start_s
            rec["end_s"] = end_s
            rec["duration"] = row_evt["duration"]

            # 拷贝事件特征
            for col in [
                "roll_peak","LTR_peak","steer_range",
                "steer_rate_peak","yaw_rate_peak","roll_rate_peak",
                "vx_mean","z_range","dz_max",
                "offroad_any","offroad_ratio","offroad_max_cont_time",
                "trigger_type","event_level","episode_id","phase_type","keep_for_training"
            ]:
                rec[col] = row_evt.get(col, np.nan)

            # 车辆状态
            rec.update(vehicle_state_at_start(df_veh, start_s))

            # 生理
            rec.update(physio_extract_features(df_phy, start_s, end_s))

            # EEG
            rec.update(eeg_extract_features(raw_eeg, start_s, end_s))

            records.append(rec)

        # 保存文件
        out_name = prefix + "_multimodal_event_features_v1.csv"
        out_path = os.path.join(event_dir, out_name)

        pd.DataFrame(records).to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"   ✔ 已生成：{out_path}")


# ============================================================
# 开始批量执行
# ============================================================

if __name__ == "__main__":

    for sub in SUBJECT_LIST:
        process_subject(sub)

    print("\n🎉 全部被试全部实验处理完成！")
