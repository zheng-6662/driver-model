# # -*- coding: utf-8 -*-
# """
# 鑴戠數棰勫鐞哶V3_batch.py
# =======================
#
# V3 鐗堟湰鍗囩骇瑕佺偣锛?
#   1. 涓嶅啀鍋囧畾 fs=1000Hz锛岃€屾槸浠?StorageTime 鑷姩浼拌鐪熷疄閲囨牱鐜囷紙鈮?00Hz锛?
#   2. 鎵归噺閬嶅巻鎵€鏈夎璇曠殑鍘熷 EEG CSV锛岃緭鍑哄埌 鑴戠數鏁版嵁澶勭悊\<琚瘯>\eeg_preprocessed_v2\
#   3. 澶勭悊娴佺▼锛?
#         - 璇诲彇 CSV锛屾埅鎺夊紑澶村叏 NaN 鐨?EEG 娈?
#         - 瀵瑰皯閲?NaN 鎻掑€?鍓嶅悗濉厖
#         - 鐢?StorageTime 浼拌 fs锛屾瀯寤?MNE Raw
#         - 1鈥?0Hz 甯﹂€?+ 50Hz notch + 骞冲潎鍙傝€?
#         - 绠€鍗曞潖閬撴娴嬶紙鍩轰簬閫氶亾 std z-score锛?
#         - 鎷熷悎 ICA锛屽苟淇濆瓨 ICA 妯″瀷 + raw_clean + raw_clean_ica
#
# 鍚庣画 ICA 鎴愬垎鑷姩鍓旈櫎锛?
#   - 浣跨敤浣犱箣鍓嶇殑 ica_auto_clean_final.py 鍗冲彲锛堜笉闇€瑕佹敼锛屽彧鏄椂闂磋酱鍙樻纭簡锛?
#
# """
#
# from pathlib import Path
# import numpy as np
# import pandas as pd
# import mne
# from mne.preprocessing import ICA
#
#
# # =============== 璺緞閰嶇疆 ===============
#
# # 鍘熷鑴戠數 CSV 鏍圭洰褰曪紙姣忎釜琚瘯涓€涓瓙鏂囦欢澶癸級
# RAW_EEG_ROOT = Path(
#     r"F:\鏁版嵁闆嗗鐞哱data_process\datasetprocess\鏁版嵁棰勫鐞哱鍘熷鑴戠數鏁版嵁"
# )
#
# # 澶勭悊鍚?EEG 杈撳嚭鏍圭洰褰曪紙姣忎釜琚瘯涓€涓瓙鏂囦欢澶癸級
# OUT_ROOT = Path(
#     r"F:\鏁版嵁闆嗗鐞哱data_process\datasetprocess\鏁版嵁棰勫鐞哱鑴戠數鏁版嵁澶勭悊\姝ｇ‘澶勭悊鐗堟湰"
# )
#
# # EEG 缂╂斁锛氬師濮嬪崟浣嶈嫢涓?碌V锛屽垯涔?1e-6 鍙樻垚 V锛圡NE 鏍囧噯锛?
# EEG_SCALE = 1e-6
#
# # 閫氶亾鍚嶆槧灏勶細index 涓?0-based锛坈hannel0 鈫?Ch1锛?
# CHANNEL_NAME_MAP = {
#     0: "P7",
#     1: "P4",
#     2: "Cz",
#     3: "Pz",
#     4: "P3",
#     5: "P8",
#     6: "O1",
#     7: "O2",
#     8: "T8",
#     9: "F8",
#     10: "C4",
#     11: "F4",
#     12: "Fp2",
#     13: "Fz",
#     14: "C3",
#     15: "F3",
#     16: "Fp1",
#     17: "T7",
#     18: "F7",
#     19: "Oz",
#     20: "PO4",
#     21: "FC6",
#     22: "FC2",
#     23: "AF4",
#     24: "CP6",
#     25: "CP2",
#     26: "CP1",
#     27: "CP5",
#     28: "FC1",
#     29: "FC5",
#     30: "AF3",
#     31: "PO3",
# }
#
#
# # =============== 鏍稿績鍑芥暟 ===============
#
# def load_eeg_from_csv(csv_path: Path) -> mne.io.Raw:
#     """
#     璇诲彇鍗曚釜 EEG CSV锛?
#       - 璇嗗埆 EEG 閫氶亾鍒?
#       - 涓㈠純寮€澶村叏 NaN
#       - 鎻掑€煎～鍏?NaN
#       - 鍒╃敤 StorageTime 鑷姩浼拌閲囨牱鐜?
#       - 鍒涘缓甯︾湡瀹炴椂闂磋酱鐨?Raw 瀵硅薄
#     """
#     print(f"\n>>> 璇诲彇 EEG CSV: {csv_path}")
#     df = pd.read_csv(csv_path)
#
#     # 1) 鍙彇 EEG 閫氶亾鍒?
#     eeg_cols = [c for c in df.columns if c.startswith("LSLOutletStreamName-EEG|channel")]
#     eeg_cols = sorted(eeg_cols, key=lambda x: int(x.split("channel")[-1]))
#     if not eeg_cols:
#         raise RuntimeError("鏈彂鐜?'LSLOutletStreamName-EEG|channelX' 褰㈠紡鐨勫垪锛岃妫€鏌?CSV 鏍煎紡銆?)
#
#     print(f"妫€娴嬪埌 {len(eeg_cols)} 涓?EEG 閫氶亾鍒椼€?)
#     df_eeg = df[eeg_cols].copy()
#
#     # 2) 涓㈠純寮€澶村叏 NaN 鐨?EEG 琛岋紝骞跺悓姝ヨ鍓?StorageTime 绛?
#     all_nan_mask = df_eeg.isna().all(axis=1)
#     if all_nan_mask.iloc[0]:
#         first_non_nan_idx = (~all_nan_mask).idxmax()
#         print(f"  寮€澶村瓨鍦?EEG 鍏ㄤ负 NaN 琛岋紝涓㈠純鍓?{first_non_nan_idx} 琛屻€?)
#         df_eeg = df_eeg.iloc[first_non_nan_idx:].reset_index(drop=True)
#         df = df.iloc[first_non_nan_idx:].reset_index(drop=True)
#     else:
#         print("  寮€澶存病鏈?EEG 鍏?NaN 琛屻€?)
#
#     # 3) 瀵规畫浣?NaN 鍋氭彃鍊?+ 鍓嶅悗鍚戝～鍏?
#     if df_eeg.isna().any().any():
#         print("  妫€娴嬪埌 EEG 涓瓨鍦?NaN 鈫?杩涜绾挎€ф彃鍊?+ 鍓嶅悗鍚戝～鍏呫€?)
#         df_eeg = df_eeg.interpolate(axis=0).ffill().bfill()
#
#     # 4) 浣跨敤 StorageTime 鑷姩浼拌閲囨牱鐜?
#     print("\n  === 鍩轰簬 StorageTime 鑷姩浼拌閲囨牱鐜?===")
#     if "StorageTime" not in df.columns:
#         raise RuntimeError("CSV 涓己灏?StorageTime 鍒楋紝鏃犳硶浼拌閲囨牱鐜囥€?)
#
#     ts = pd.to_datetime(df["StorageTime"], format="%Y/%m/%d %H:%M:%S.%f")
#     dt_sec = (ts.iloc[-1] - ts.iloc[0]).total_seconds()
#     n_samples = len(df_eeg)
#     fs_est = (n_samples - 1) / dt_sec
#
#     print(f"    鎬绘椂闀匡紙StorageTime锛? {dt_sec:.3f} 绉?)
#     print(f"    鏍锋湰鐐规暟: {n_samples}")
#     print(f"    浼拌閲囨牱鐜?鈮?{fs_est:.3f} Hz")
#
#     # 5) 杞?numpy 骞剁缉鏀惧埌 Volt
#     data = df_eeg.to_numpy(dtype=float).T * EEG_SCALE
#     n_channels, n_times = data.shape
#     print(f"  EEG 鏁版嵁缁村害: {n_channels} 閫氶亾 脳 {n_times} 鐐?)
#
#     # 6) 鏋勯€犻€氶亾鍚?
#     ch_names = []
#     for idx in range(n_channels):
#         ch_names.append(CHANNEL_NAME_MAP.get(idx, f"EEG{idx+1}"))
#
#     info = mne.create_info(
#         ch_names=ch_names,
#         sfreq=fs_est,
#         ch_types=["eeg"] * n_channels
#     )
#
#     raw = mne.io.RawArray(data=data, info=info)
#     print("  Raw 瀵硅薄宸插垱寤恒€?)
#
#     # 鍙€夛細娣诲姞鏍囧噯 10-20 鐢垫瀬浣嶇疆锛堝拷鐣ョ己澶遍€氶亾锛?
#     try:
#         raw.set_montage("standard_1020", on_missing="ignore")
#         print("  宸茶缃?standard_1020 鐢垫瀬浣嶇疆淇℃伅銆?)
#     except Exception as e:
#         print(f"  璁剧疆钂欏お濂囧け璐ワ紙鍙互蹇界暐锛夛細{e}")
#
#     return raw
#
#
# def basic_preprocess(raw: mne.io.Raw) -> mne.io.Raw:
#     """
#     1鈥?0 Hz 甯﹂€?+ 50 Hz 宸ラ闄锋尝 + 骞冲潎鍙傝€?
#     """
#     raw = raw.copy()
#
#     print("  寮€濮?1鈥?0 Hz 甯﹂€氭护娉?..")
#     raw.filter(l_freq=1., h_freq=40., fir_design="firwin")
#
#     print("  鏂藉姞 50 Hz notch 婊ゆ尝...")
#     raw.notch_filter(freqs=[50.])
#
#     print("  璁剧疆骞冲潎鍙傝€?..")
#     raw.set_eeg_reference("average", projection=False)
#
#     return raw
#
#
# def detect_bad_channels_std(raw: mne.io.Raw, z_thresh: float = 4.0):
#     """
#     绠€鍗曞潖閬撴娴嬶細鏍规嵁閫氶亾 std 鐨?z-score銆?
#     缁撴灉鍐欏叆 raw.info['bads']锛屽悓鏃舵墦鍗板墠鑻ュ共涓€氶亾銆?
#     """
#     data = raw.get_data()
#     ch_std = data.std(axis=1)
#     mean_std = ch_std.mean()
#     std_std = ch_std.std()
#     z_vals = (ch_std - mean_std) / (std_std + 1e-12)
#
#     bads = [raw.ch_names[i] for i, z in enumerate(z_vals) if z > z_thresh]
#     print("  閫氶亾 std z-score Top10锛?)
#     for name, z in sorted(zip(raw.ch_names, z_vals), key=lambda x: x[1], reverse=True)[:10]:
#         print(f"    {name:>4s}: z = {z:6.2f}")
#
#     if bads:
#         print(f"  妫€娴嬪埌鐤戜技鍧忛亾锛坺 > {z_thresh}锛夛細{bads}")
#     else:
#         print("  鏈娴嬪埌鏄庢樉鍧忛亾銆?)
#
#     raw.info["bads"] = bads
#     return bads
#
#
# def run_ica(raw: mne.io.Raw, n_components: int = 20, random_state: int = 97) -> ICA:
#     """
#     鎷熷悎 ICA 妯″瀷锛堜笉鍋氳嚜鍔ㄥ墧闄わ級锛屼粎淇濆瓨鏉冮噸锛?
#     鍚庣画鐢?ica_auto_clean_final.py 鍋氳嚜鍔ㄥ墧闄?+ apply銆?
#     """
#     print("  寮€濮?ICA 鎷熷悎...")
#     ica = ICA(n_components=n_components, random_state=random_state, max_iter="auto")
#     ica.fit(raw)
#     print("  ICA 鎷熷悎瀹屾垚锛宯_components =", ica.n_components_)
#     return ica
#
#
# def process_one_csv(csv_path: Path, out_dir: Path):
#     """
#     澶勭悊鍗曚釜鍘熷 EEG CSV锛岃緭鍑猴細
#       - *_raw_clean.fif
#       - *_raw_clean_ica.fif锛堟澶勫厛涓嶅墧闄ゆ垚鍒嗭紝鍙槸鍗犱綅锛?
#       - *_ica.fif
#     """
#     out_dir.mkdir(parents=True, exist_ok=True)
#
#     base_name = csv_path.stem  # e.g. Entity_Recording_..._eeg
#     base_name = base_name.replace("_eeg", "")  # 鍘绘帀灏鹃儴 _eeg锛岀粺涓€鍛藉悕
#
#     # 1. 鍔犺浇鍘熷 EEG
#     raw = load_eeg_from_csv(csv_path)
#
#     # 2. 鍩虹棰勫鐞?
#     raw_clean = basic_preprocess(raw)
#
#     # 3. 鍧忛亾妫€娴嬶紙鍙槸鏍囪锛屼笉鎻掑€硷級
#     detect_bad_channels_std(raw_clean, z_thresh=4.0)
#
#     # 4. 淇濆瓨 raw_clean
#     out_raw_clean = out_dir / f"{base_name}_eeg_raw_clean.fif"
#     print(f"  淇濆瓨鍩虹棰勫鐞嗗悗鐨?Raw 鍒? {out_raw_clean}")
#     raw_clean.save(out_raw_clean, overwrite=True)
#
#     # 5. ICA 鎷熷悎锛堝湪 bad 鏍囪浣嗘湭鎻掑€肩殑 raw_clean 涓婏級
#     ica = run_ica(raw_clean)
#
#     # 6. 淇濆瓨 ICA 妯″瀷
#     out_ica = out_dir / f"{base_name}_eeg_ica.fif"
#     print(f"  淇濆瓨 ICA 妯″瀷鍒? {out_ica}")
#     ica.save(out_ica, overwrite=True)
#
#     # 7. 鍏堢敓鎴愪竴涓€滄湭鍓旈櫎鎴愬垎鈥濈殑 raw_clean_ica 鐗堟湰锛堟柟渚垮悗缁姣?鍏煎锛?
#     raw_clean_ica = ica.apply(raw_clean.copy(), exclude=[])
#     out_raw_clean_ica = out_dir / f"{base_name}_eeg_raw_clean_ica.fif"
#     print(f"  淇濆瓨鏆傛湭鍓旈櫎鎴愬垎鐨?Raw 鍒? {out_raw_clean_ica}")
#     raw_clean_ica.save(out_raw_clean_ica, overwrite=True)
#
#     print("  鉁?褰撳墠鏂囦欢澶勭悊瀹屾垚銆俓n")
#
#
# # =============== 鎵瑰鐞嗕富绋嬪簭 ===============
#
# def main():
#     print("========== EEG 棰勫鐞?V3 鎵瑰鐞嗗紑濮?==========")
#     print("鍘熷 EEG 鏍圭洰褰?", RAW_EEG_ROOT)
#     print("杈撳嚭鏍圭洰褰?", OUT_ROOT)
#
#     for subj_dir in RAW_EEG_ROOT.iterdir():
#         if not subj_dir.is_dir():
#             continue
#
#         subj_name = subj_dir.name
#         print(f"\n====== 澶勭悊琚瘯锛歿subj_name} ======")
#
#         out_subj_dir = OUT_ROOT / subj_name / "eeg_preprocessed_v2"
#
#         # 閬嶅巻璇ヨ璇曚笅鎵€鏈?*_eeg.csv
#         csv_files = sorted(subj_dir.glob("*_eeg.csv"))
#         if not csv_files:
#             print("  鈿?鏈壘鍒?*_eeg.csv锛岃烦杩囪琚瘯銆?)
#             continue
#
#         for csv_path in csv_files:
#             try:
#                 process_one_csv(csv_path, out_subj_dir)
#             except Exception as e:
#                 print(f"  鉂?澶勭悊澶辫触锛歿csv_path.name} -> {e}")
#
#     print("\n========== EEG 棰勫鐞?V3 鍏ㄩ儴瀹屾垚 ==========")
#
#
# if __name__ == "__main__":
#     main()

# -*- coding: utf-8 -*-
"""
eeg_fix_and_resample_200Hz_v2.py
================================

閫傞厤 MNE 鏈€鏂扮増鏈細
  - 涓嶈兘鍐嶄娇鐢?raw.info['sfreq'] = fs
  - 蹇呴』閲嶆柊鏋勯€?RawArray 鏉ヤ慨姝ｉ噰鏍风巼锛堟纭椂闂磋酱锛?
  - 鐒跺悗 resample 鍒?200Hz
"""

from pathlib import Path
import mne
import pandas as pd
import numpy as np

RAW_CSV_ROOT = Path(
    r"F:\data_set_process\data_process\01_datasets\数据预处理\原始脑电数据"
)

# 娉ㄦ剰锛氳繖閲屾寚鍚戔€滄纭鐞嗙増鏈€?
EEG_PROC_ROOT = Path(
    r"F:\data_set_process\data_process\01_datasets\数据预处理\脑电数据处理\正确处理版本"
)

TARGET_FS = 200  # 闄嶉噰鏍风洰鏍?

def estimate_fs_from_csv(csv_path: Path):
    df = pd.read_csv(csv_path)
    ts = pd.to_datetime(df["StorageTime"], format="%Y/%m/%d %H:%M:%S.%f")
    dt_sec = (ts.iloc[-1] - ts.iloc[0]).total_seconds()
    n_samples = len(df)
    fs_est = (n_samples - 1) / dt_sec
    return fs_est

def rebuild_raw_with_correct_fs(raw_old: mne.io.Raw, fs_new: float):
    """
    閲嶆柊鏋勯€?RawArray锛堜娇鐢ㄦ纭殑 sampling rate锛?
    """
    data = raw_old.get_data()
    ch_names = raw_old.ch_names
    ch_types = ["eeg"] * len(ch_names)

    info_new = mne.create_info(
        ch_names=ch_names,
        sfreq=fs_new,
        ch_types=ch_types
    )

    raw_new = mne.io.RawArray(data, info_new)

    print(f"  鉁?宸查噸鏂版瀯寤?RawArray锛屾纭?fs = {fs_new:.3f}")
    print(f"  鉁?鏂版椂闀匡細{raw_new.times[-1]:.3f} 绉?)

    return raw_new

def process_one(csv_path: Path, fif_path: Path, out_path: Path):
    print(f"\n==== 淇 + 闄嶉噰鏍?====")
    print("CSV :", csv_path.name)
    print("FIF :", fif_path.name)

    # 1. 浼拌鐪熷疄閲囨牱鐜?
    fs_est = estimate_fs_from_csv(csv_path)
    print(f"  鐪熷疄閲囨牱鐜?鈮?{fs_est:.2f} Hz")

    # 2. 璇诲彇鍘熷 FIF
    raw_old = mne.io.read_raw_fif(fif_path, preload=True)

    print(f"  鍘熼敊璇椂闀匡細{raw_old.times[-1]:.3f}")

    # 3. 浣跨敤姝ｇ‘ fs 閲嶆瀯 Raw
    raw_fixed = rebuild_raw_with_correct_fs(raw_old, fs_est)

    # 4. 闄嶉噰鏍峰埌200Hz
    print("  闄嶉噰鏍疯嚦 200Hz ...")
    raw_200 = raw_fixed.copy().resample(TARGET_FS)
    print(f"  闄嶉噰鏍峰悗鏃堕暱锛歿raw_200.times[-1]:.3f}")

    # 5. 淇濆瓨
    raw_200.save(out_path, overwrite=True)
    print(f"  鉁?淇濆瓨涓猴細{out_path.name}")


def main():
    print("========== 淇閲囨牱鐜?+ 闄嶉噰鏍?00Hz 寮€濮?==========")

    for subj_dir in RAW_CSV_ROOT.iterdir():
        if not subj_dir.is_dir():
            continue

        subj = subj_dir.name
        print(f"\n====== 琚瘯锛歿subj} ======")

        # CSV 鏂囦欢
        csv_files = sorted(subj_dir.glob("*_eeg.csv"))
        if not csv_files:
            print("  鈿?鏃?CSV锛岃烦杩?)
            continue

        # FIF 澶勭悊鐩綍
        subj_proc = EEG_PROC_ROOT / subj / "eeg_preprocessed_v2"
        if not subj_proc.exists():
            print("  鈿?鎵句笉鍒板鐞嗗悗 FIF 鐩綍锛岃烦杩?)
            continue

        for csv in csv_files:
            base = csv.stem.replace("_eeg", "")

            fif_clean = subj_proc / f"{base}_eeg_raw_clean.fif"
            fif_clean_out = subj_proc / f"{base}_eeg_raw_clean_resamp200.fif"

            fif_final = subj_proc / f"{base}_eeg_raw_clean_ica_final.fif"
            fif_final_out = subj_proc / f"{base}_eeg_raw_clean_ica_final_resamp200.fif"

            if fif_clean.exists():
                process_one(csv, fif_clean, fif_clean_out)

            if fif_final.exists():
                process_one(csv, fif_final, fif_final_out)

    print("\n========== 鍏ㄩ儴瀹屾垚 ==========")
    print("鎺ヤ笅鏉ュ彲杩愯锛歩ca_auto_clean_final.py 杩涜缁堟瀬 ICA 娓呮礂")


if __name__ == "__main__":
    main()

