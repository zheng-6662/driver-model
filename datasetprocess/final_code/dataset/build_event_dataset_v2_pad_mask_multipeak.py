# -*- coding: utf-8 -*-
"""
=========================================================
EVENT-LEVEL MULTIMODAL DATASET BUILDER v2.0
Pad + Mask + Multi-Peak
=========================================================
"""

import os
import re
import glob
import numpy as np
import pandas as pd
from scipy.io import savemat
from scipy.signal import find_peaks
import mne

# ======================================================
# CONFIG
# ======================================================
ROOT_IN = r"F:\数据集处理\data_process\datasetprocess\多模态数据\被试数据集合"
ROOT_OUT = os.path.join(os.path.dirname(ROOT_IN), "Event_Dataset_v2")

TIME_COL = "t_s"
FS = 200.0
WINDOW_PRE = 2.0
WINDOW_POST = 2.0
N_SAMPLES = int((WINDOW_PRE + WINDOW_POST) * FS) + 1  # 801
PEAK_MIN_DIST_SEC = 0.8   # 多峰最小间距（秒）

# ======================================================
# SIGNAL LISTS
# ======================================================
PHYSIO_SIGNALS = [
    "ECG_raw200", "EMG_raw200", "EDA_raw200", "RESP_raw200",
    "ECG_filt200", "EMG_filt200", "EDA_filt200", "RESP_filt200"
]

VEHICLE_SIGNALS = [
    "zx|x", "zx|y", "zx|z",
    "zx|AcceleratorPedal", "zx|BrakePedal", "zx|SteeringWheel",
    "zx|ax", "zx|ay", "zx|vx", "zx|vy",
    "zx|ayaw", "zx|aroll", "zx|apitch",
    "zx|roll", "zx|pitch", "zx|yaw",
    "zx|vroll", "zx|vpitch", "zx|vyaw",
    "zx1|v_km/h", "zx1|lanecurvatureXY",
    "zx1|mu", "zx1|lateraldistance"
]

# ======================================================
# HELPERS
# ======================================================
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def to_double(x):
    return np.asarray(x, dtype=np.float64)

def extract_stamp(path):
    m = re.search(
        r"Entity_Recording_(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})",
        os.path.basename(path)
    )
    return m.group(1) if m else None

def slice_with_pad_df(df, t0, t1, time_col):
    """
    固定长度切窗 + pad + mask
    """
    t_full = np.linspace(t0, t1, N_SAMPLES)
    out = {}
    mask = np.zeros(N_SAMPLES, dtype=bool)

    for c in df.columns:
        if c == time_col:
            continue
        y = np.interp(
            t_full,
            df[time_col].values,
            df[c].values,
            left=np.nan,
            right=np.nan
        )
        out[c] = y.astype(np.float64)
        mask |= ~np.isnan(y)

    return out, mask, t_full

# ======================================================
# MAIN
# ======================================================
def main():

    ensure_dir(ROOT_OUT)
    out_vehicle = os.path.join(ROOT_OUT, "vehicle")
    out_physio = os.path.join(ROOT_OUT, "physio")
    out_eeg = os.path.join(ROOT_OUT, "eeg")
    for p in [out_vehicle, out_physio, out_eeg]:
        ensure_dir(p)

    index_rows = []

    subjects = [
        d for d in os.listdir(ROOT_IN)
        if os.path.isdir(os.path.join(ROOT_IN, d))
    ]

    for subj in subjects:
        print(f"\n[SUBJ] {subj}")
        subj_dir = os.path.join(ROOT_IN, subj)

        vehicle_files = [
            f for f in glob.glob(os.path.join(subj_dir, "**", "*.csv"), recursive=True)
            if ("vehicle" in os.path.basename(f).lower())
            and ("events" not in os.path.basename(f).lower())
        ]
        vehicle_files.sort()

        for exp_idx, veh_file in enumerate(vehicle_files, start=1):
            print(f"  [VEH] {os.path.basename(veh_file)}")

            veh_df = pd.read_csv(veh_file)
            veh_df.columns = [c.strip().replace("\ufeff", "") for c in veh_df.columns]
            if TIME_COL not in veh_df.columns:
                continue

            stamp = extract_stamp(veh_file)
            if stamp is None:
                continue

            events_files = [
                f for f in glob.glob(os.path.join(subj_dir, "event", "*.csv"))
                if (stamp in os.path.basename(f))
                and ("_events_v312" in os.path.basename(f))
            ]
            if not events_files:
                continue

            ev = pd.read_csv(events_files[0])
            ev = ev[ev["phase_type"] == "primary"].reset_index(drop=True)
            if ev.empty:
                continue

            phys_df = None
            phys_files = glob.glob(os.path.join(subj_dir, "physio", "*.csv"))
            if phys_files:
                phys_df = pd.read_csv(phys_files[0])
                phys_df.columns = [c.strip().replace("\ufeff", "") for c in phys_df.columns]
                if TIME_COL not in phys_df.columns:
                    phys_df = None

            raw_eeg = None
            eeg_files = glob.glob(os.path.join(subj_dir, "EEG", f"*{stamp}*.fif"))
            if eeg_files:
                raw_eeg = mne.io.read_raw_fif(eeg_files[0], preload=True, verbose=False)

            for i, row in ev.iterrows():

                seg = veh_df[
                    (veh_df[TIME_COL] >= row["start_s"]) &
                    (veh_df[TIME_COL] <= row["end_s"])
                ]
                if seg.empty:
                    continue

                # -------- anchor selection --------
                anchor_col, anchor_type = None, None
                for col, typ in [
                    ("steer_rate", "steer_rate"),
                    ("zx|SteeringWheel", "steer_rate"),
                    ("yaw_rate", "yaw_rate"),
                    ("zx|vyaw", "yaw_rate"),
                    ("roll", "roll"),
                    ("zx|roll", "roll")
                ]:
                    if col in seg.columns:
                        anchor_col, anchor_type = col, typ
                        break
                if anchor_col is None:
                    continue

                y = np.abs(seg[anchor_col].values)
                peaks, _ = find_peaks(
                    y,
                    distance=int(PEAK_MIN_DIST_SEC * FS)
                )

                for k, p in enumerate(peaks, start=1):
                    t_anchor = float(seg.iloc[p][TIME_COL])
                    event_id = f"{subj}_E{exp_idx}_L{row['event_level']}_P{i+1:02d}_K{k}"

                    w0, w1 = t_anchor - WINDOW_PRE, t_anchor + WINDOW_POST

                    # -------- VEHICLE --------
                    veh_data, veh_mask, t_full = slice_with_pad_df(
                        veh_df[ [TIME_COL] + [c for c in VEHICLE_SIGNALS if c in veh_df.columns] ],
                        w0, w1, TIME_COL
                    )

                    savemat(
                        os.path.join(out_vehicle, f"{event_id}.mat"),
                        dict(
                            data={k.replace("|", "_"): v for k, v in veh_data.items()},
                            time=t_full - t_anchor,
                            valid_mask=veh_mask,
                            valid_ratio=float(np.mean(veh_mask)),
                            fs=np.array([FS]),
                            meta=dict(
                                subj=subj, exp=exp_idx,
                                anchor_type=anchor_type,
                                anchor_time_global=t_anchor,
                                window=[-2.0, 2.0]
                            )
                        )
                    )

                    # -------- PHYSIO --------
                    if phys_df is not None:
                        phys_data, phys_mask, _ = slice_with_pad_df(
                            phys_df[[TIME_COL] + [c for c in PHYSIO_SIGNALS if c in phys_df.columns]],
                            w0, w1, TIME_COL
                        )
                        savemat(
                            os.path.join(out_physio, f"{event_id}.mat"),
                            dict(
                                data=phys_data,
                                time=t_full - t_anchor,
                                valid_mask=phys_mask,
                                valid_ratio=float(np.mean(phys_mask)),
                                fs=np.array([FS])
                            )
                        )

                    # -------- EEG --------
                    if raw_eeg is not None:
                        try:
                            eeg_seg = raw_eeg.copy().crop(tmin=w0, tmax=w1, include_tmax=True)
                            eeg_data = eeg_seg.get_data().astype(np.float64)
                            eeg_time = eeg_seg.times + w0 - t_anchor

                            savemat(
                                os.path.join(out_eeg, f"{event_id}.mat"),
                                dict(
                                    data=eeg_data,
                                    time=eeg_time,
                                    fs=np.array([raw_eeg.info["sfreq"]]),
                                    ch_names=eeg_seg.ch_names
                                )
                            )
                        except Exception:
                            pass

                    index_rows.append(dict(
                        event_id=event_id,
                        subj=subj,
                        exp=exp_idx,
                        level=row["event_level"],
                        anchor_type=anchor_type
                    ))

    pd.DataFrame(index_rows).to_csv(
        os.path.join(ROOT_OUT, "events_index.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print("\n[DONE] Event Dataset v2.0 generated.")

# ======================================================
if __name__ == "__main__":
    main()
