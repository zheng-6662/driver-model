# -*- coding: utf-8 -*-
"""
eeg_vehicle_time_alignment_checker.py
=====================================

鍔熻兘锛?
  - 缁欏畾 EEG fif 鏂囦欢锛?_raw_clean_ica_final.fif锛?
  - 缁欏畾 杞﹁締鏁版嵁 CSV锛?_vehicle_physio_aligned_200Hz.csv锛?
  - 鍒嗗埆璁＄畻涓よ€呯殑鏃堕棿闀垮害
  - 鍒ゆ柇鏄惁瀵归綈锛堝樊鍊?< 0.05 绉掞級

浣跨敤锛?
  淇敼 EEG_FILE 鍜?VEH_FILE 涓轰綘鐨勮矾寰?
  python eeg_vehicle_time_alignment_checker.py
"""

from pathlib import Path
import mne
import pandas as pd

# ================== 淇敼杩欓噷鍗冲彲 ==================
EEG_FILE = Path(
    r"F:\data_set_process\data_process\01_datasets\数据预处理\脑电数据处理\cwh\eeg_preprocessed_v2\Entity_Recording_2025_09_26_19_23_28_eeg_raw_clean_ica_final.fif"
)

VEH_FILE = Path(
    r"F:\data_set_process\data_process\01_datasets\数据预处理\对齐后数据_车辆生理_200Hz\cwh\Entity_Recording_2025_09_26_19_23_28_vehicle_timealign_fixed_200Hz_v14_vehicle_physio_aligned_200Hz.csv"
)
# ===================================================


def check_alignment(eeg_path: Path, veh_path: Path):
    print("\n=============== 鏃堕棿瀵归綈妫€鏌?===============")

    print(f"EEG 鏂囦欢: {eeg_path.name}")
    print(f"杞﹁締 CSV: {veh_path.name}")

    # ====== 1. 鍔犺浇 EEG ======
    raw = mne.io.read_raw_fif(eeg_path, preload=False, verbose=False)

    eeg_duration = raw.times[-1]  # 绉?
    print(f"\nEEG 鏃堕暱锛堢锛? {eeg_duration:.4f}")

    # ====== 2. 鍔犺浇杞﹁締鏁版嵁 ======
    df = pd.read_csv(veh_path)

    if "t_s" not in df.columns:
        raise RuntimeError("杞﹁締 CSV 涓病鏈?t_s 鍒楋紒")

    veh_start = df["t_s"].iloc[0]
    veh_end   = df["t_s"].iloc[-1]
    veh_duration = veh_end - veh_start

    print(f"杞﹁締鏃堕暱锛堢锛? {veh_duration:.4f}")

    # ====== 3. 瀵归綈鍒ゆ柇 ======
    diff = abs(eeg_duration - veh_duration)
    print(f"\n鏃堕棿宸紙缁濆鍊?绉掞級: {diff:.4f}")

    if diff < 0.05:
        print("\n馃帀 鏃堕棿杞村畬鍏ㄤ竴鑷?鈫?EEG 涓庤溅杈嗘暟鎹垚鍔熷榻愶紒")
    else:
        print("\n鈿狅笍 鏃堕棿杞翠笉涓€鑷?鈫?鍙兘鏈榻愶紝璇锋鏌ワ紒")

    print("=============================================\n")


if __name__ == "__main__":
    check_alignment(EEG_FILE, VEH_FILE)
