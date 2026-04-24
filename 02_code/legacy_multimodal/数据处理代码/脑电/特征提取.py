# -*- coding: utf-8 -*-
"""
EEG Event-level Feature Extraction (ROLL_PEAK aligned)
=====================================================
- One event -> one row
- EEG window: [roll_peak - WIN_SEC, roll_peak)
- Robust to missing channels / bad channels
"""

import os
from glob import glob
import numpy as np
import pandas as pd
import mne

# =========================
# CONFIG
# =========================
FS = 200
WIN_SEC = 2.0
WIN_LEN = int(FS * WIN_SEC)

STRONG_LABELS = ["medium_active", "strong_active", "extreme_active"]

BANDS = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

EPS = 1e-12


# =========================
# UTILS
# =========================
def find_col(cols, names):
    for n in names:
        if n in cols:
            return n
    return None


def bandpower(psd, freqs, fmin, fmax):
    idx = (freqs >= fmin) & (freqs < fmax)
    if not np.any(idx):
        return np.nan
    return np.trapezoid(psd[idx], freqs[idx])


def safe_roi_mean(values, ch_names, roi):
    idx = [ch_names.index(c) for c in roi if c in ch_names]
    if len(idx) == 0:
        return np.nan
    return float(np.nanmean(values[idx]))


def pick_asym_pair(ch_names):
    if "F4" in ch_names and "F3" in ch_names:
        return "F4", "F3", "F4F3"
    if "AF4" in ch_names and "AF3" in ch_names:
        return "AF4", "AF3", "AF4AF3"
    if "F8" in ch_names and "F7" in ch_names:
        return "F8", "F7", "F8F7"
    return None, None, "NA"


# =========================
# CORE FEATURE FUNCTION
# =========================
def compute_eeg_features(raw, start_idx, end_idx):
    ch_all = raw.ch_names

    frontal_roi   = [c for c in ["Fp1","Fp2","AF3","AF4","F3","Fz","F7","F8","FC1","FC5","FC6"] if c in ch_all]
    temporal_roi  = [c for c in ["T7","T8"] if c in ch_all]
    occipital_roi = [c for c in ["O1","O2","Oz","PO3"] if c in ch_all]

    need_ch = sorted(set(frontal_roi + temporal_roi + occipital_roi + ["F3","F4","AF3","AF4","F7","F8"]))

    actual_ch = []
    picks = []
    for ch in need_ch:
        if ch in ch_all:
            actual_ch.append(ch)
            picks.append(ch_all.index(ch))

    if len(picks) < 2:
        return None, "too_few_channels"

    data = raw.get_data(picks=picks, start=start_idx, stop=end_idx)
    if data.shape[1] < 64:
        return None, "window_too_short"

    psd, freqs = mne.time_frequency.psd_array_welch(
        data,
        sfreq=raw.info["sfreq"],
        fmin=4.0,
        fmax=45.0,
        n_fft=256,
        n_per_seg=256,
        n_overlap=128,
        average="mean",
        verbose=False
    )

    bp = {}
    for band, (f1, f2) in BANDS.items():
        bp[band] = np.array([bandpower(psd[i], freqs, f1, f2) for i in range(psd.shape[0])])

    total = bp["theta"] + bp["alpha"] + bp["beta"] + bp["gamma"] + EPS

    # ta_beta
    def ta_beta(theta, alpha, beta):
        if np.isnan(theta) or np.isnan(alpha) or np.isnan(beta):
            return np.nan
        return (theta + alpha) / (beta + EPS)

    # gamma_rel
    def gamma_rel(gamma, tot):
        if np.isnan(gamma) or np.isnan(tot):
            return np.nan
        return gamma / (tot + EPS)

    # ROI means
    theta_f = safe_roi_mean(bp["theta"], actual_ch, frontal_roi)
    alpha_f = safe_roi_mean(bp["alpha"], actual_ch, frontal_roi)
    beta_f  = safe_roi_mean(bp["beta"],  actual_ch, frontal_roi)
    gamma_f = safe_roi_mean(bp["gamma"], actual_ch, frontal_roi)
    total_f = safe_roi_mean(total,        actual_ch, frontal_roi)

    theta_t = safe_roi_mean(bp["theta"], actual_ch, temporal_roi)
    alpha_t = safe_roi_mean(bp["alpha"], actual_ch, temporal_roi)
    beta_t  = safe_roi_mean(bp["beta"],  actual_ch, temporal_roi)
    gamma_t = safe_roi_mean(bp["gamma"], actual_ch, temporal_roi)
    total_t = safe_roi_mean(total,        actual_ch, temporal_roi)

    theta_o = safe_roi_mean(bp["theta"], actual_ch, occipital_roi)
    alpha_o = safe_roi_mean(bp["alpha"], actual_ch, occipital_roi)
    beta_o  = safe_roi_mean(bp["beta"],  actual_ch, occipital_roi)
    gamma_o = safe_roi_mean(bp["gamma"], actual_ch, occipital_roi)
    total_o = safe_roi_mean(total,        actual_ch, occipital_roi)

    # alpha asymmetry
    r, l, tag = pick_asym_pair(actual_ch)
    if r is None:
        alpha_asym = np.nan
    else:
        ridx = actual_ch.index(r)
        lidx = actual_ch.index(l)
        alpha_asym = np.log(bp["alpha"][ridx] + EPS) - np.log(bp["alpha"][lidx] + EPS)

    feats = {
        f"Frontal_alpha_asym_{tag}": float(alpha_asym),
        "Occipital_ta_beta": ta_beta(theta_o, alpha_o, beta_o),
        "Frontal_ta_beta":   ta_beta(theta_f, alpha_f, beta_f),
        "Temporal_ta_beta":  ta_beta(theta_t, alpha_t, beta_t),
        "Occipital_alpha_abs": float(alpha_o),
        "Temporal_gamma_rel":  gamma_rel(gamma_t, total_t),
        "Occipital_gamma_rel": gamma_rel(gamma_o, total_o),
        "Frontal_gamma_rel":   gamma_rel(gamma_f, total_f),
    }

    return feats, "ok"


# =========================
# ONE RECORDING
# =========================
def process_one_recording(vehicle_csv, event_csv, eeg_fif):
    df_v = pd.read_csv(vehicle_csv)
    df_e = pd.read_csv(event_csv)

    col_roll = find_col(df_v.columns, ["zx|roll"])
    if col_roll is None:
        raise RuntimeError("roll column missing")

    df_e = df_e[df_e["event_level"].isin(STRONG_LABELS)]
    if len(df_e) == 0:
        return

    raw = mne.io.read_raw_fif(eeg_fif, preload=False, verbose=False)

    roll = df_v[col_roll].to_numpy()
    rows = []

    for idx, ev in df_e.iterrows():
        i0 = int(ev["start_s"] * FS)
        i1 = int(ev["end_s"] * FS)
        if i1 <= i0:
            continue

        peak_rel = np.argmax(roll[i0:i1])
        peak_idx = i0 + peak_rel

        s = peak_idx - WIN_LEN
        e = peak_idx
        if s < 0 or e > raw.n_times:
            continue

        feats, status = compute_eeg_features(raw, s, e)
        if feats is None:
            continue

        row = {
            "event_row_index": idx,
            "event_level": ev["event_level"],
            "roll_peak_s": peak_idx / FS,
        }
        row.update(feats)
        rows.append(row)

    if len(rows) == 0:
        return

    out = pd.DataFrame(rows)
    out_csv = eeg_fif.replace(
        "_eeg_raw_clean_resamp200_ica_final_qc.fif",
        f"_eeg_event_features_rollpeak_hist{int(WIN_SEC)}s.csv"
    )
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"✅ Saved: {out_csv}")


# =========================
# BATCH
# =========================
def batch_process(ROOT):
    eeg_files = glob(os.path.join(ROOT, "*", "eeg_clean", "*_qc.fif"))
    print(f"🔍 Found {len(eeg_files)} EEG files")

    for eeg_fif in eeg_files:
        subj = os.path.dirname(os.path.dirname(eeg_fif))

        vehicle_csv = os.path.join(
            subj, "vehicle",
            os.path.basename(eeg_fif).replace(
                "_eeg_raw_clean_resamp200_ica_final_qc.fif",
                "_vehicle_aligned_cleaned.csv")
        )

        event_csv = os.path.join(
            subj, "event",
            os.path.basename(eeg_fif).replace(
                "_eeg_raw_clean_resamp200_ica_final_qc.fif",
                "_vehicle_aligned_cleaned_events_v312.csv")
        )

        if not (os.path.exists(vehicle_csv) and os.path.exists(event_csv)):
            continue

        process_one_recording(vehicle_csv, event_csv, eeg_fif)


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    ROOT = r"F:\数据集处理\data_process\datasetprocess\多模态数据\被试数据集合"
    batch_process(ROOT)
