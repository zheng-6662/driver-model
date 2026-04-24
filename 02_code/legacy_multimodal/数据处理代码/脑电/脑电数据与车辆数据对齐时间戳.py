# -*- coding: utf-8 -*-
"""
eeg_vehicle_time_alignment_checker.py
=====================================

功能：
  - 给定 EEG fif 文件（*_raw_clean_ica_final.fif）
  - 给定 车辆数据 CSV（*_vehicle_physio_aligned_200Hz.csv）
  - 分别计算两者的时间长度
  - 判断是否对齐（差值 < 0.05 秒）

使用：
  修改 EEG_FILE 和 VEH_FILE 为你的路径
  python eeg_vehicle_time_alignment_checker.py
"""

from pathlib import Path
import mne
import pandas as pd

# ================== 修改这里即可 ==================
EEG_FILE = Path(
    r"F:\数据集处理\data_process\datasetprocess\数据预处理\脑电数据处理\cwh\eeg_preprocessed_v2\Entity_Recording_2025_09_26_19_23_28_eeg_raw_clean_ica_final.fif"
)

VEH_FILE = Path(
    r"F:\数据集处理\data_process\datasetprocess\数据预处理\对齐后数据_车辆生理_200Hz\cwh\Entity_Recording_2025_09_26_19_23_28_vehicle_timealign_fixed_200Hz_v14_vehicle_physio_aligned_200Hz.csv"
)
# ===================================================


def check_alignment(eeg_path: Path, veh_path: Path):
    print("\n=============== 时间对齐检查 ===============")

    print(f"EEG 文件: {eeg_path.name}")
    print(f"车辆 CSV: {veh_path.name}")

    # ====== 1. 加载 EEG ======
    raw = mne.io.read_raw_fif(eeg_path, preload=False, verbose=False)

    eeg_duration = raw.times[-1]  # 秒
    print(f"\nEEG 时长（秒）: {eeg_duration:.4f}")

    # ====== 2. 加载车辆数据 ======
    df = pd.read_csv(veh_path)

    if "t_s" not in df.columns:
        raise RuntimeError("车辆 CSV 中没有 t_s 列！")

    veh_start = df["t_s"].iloc[0]
    veh_end   = df["t_s"].iloc[-1]
    veh_duration = veh_end - veh_start

    print(f"车辆时长（秒）: {veh_duration:.4f}")

    # ====== 3. 对齐判断 ======
    diff = abs(eeg_duration - veh_duration)
    print(f"\n时间差（绝对值 秒）: {diff:.4f}")

    if diff < 0.05:
        print("\n🎉 时间轴完全一致 → EEG 与车辆数据成功对齐！")
    else:
        print("\n⚠️ 时间轴不一致 → 可能未对齐，请检查！")

    print("=============================================\n")


if __name__ == "__main__":
    check_alignment(EEG_FILE, VEH_FILE)
