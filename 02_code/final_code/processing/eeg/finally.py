# # -*- coding: utf-8 -*-
# """
# ica_auto_clean_final.py
# =======================
#
# 缁堟瀬鑷姩 ICA 鍘讳吉杩硅剼鏈紙浜哄伐绾у埆锛?
# -------------------------------------
#
# 鍔熻兘锛?
#   1. 鑷姩閬嶅巻浣犵殑 EEG 棰勫鐞嗙洰褰?
#   2. 閽堝姣忎釜 EEG 鏂囦欢璇诲彇锛?
#         *_raw_clean.fif 鎴?*_raw_clean_ica.fif
#         *_ica.fif
#   3. 鑷姩璁＄畻姣忎釜鎴愬垎 ICA source锛?
#         - 涓庡墠棰濋€氶亾锛團p1/Fp2/AF3/AF4/F7/F8锛夌浉鍏崇郴鏁?
#         - 楂橀/浣庨鑳介噺姣旓紙鑲岀數锛?
#   4. 鎸変汉宸ョ骇鏍囧噯鑷姩鍒嗙被锛?
#         A. 寮?EOG锛堝繀椤诲垹锛?
#         B. 寮鸿倢鐢碉紙蹇呴』鍒狅級
#         C. 寮?EOG锛堝彲閫夊垹闄わ級
#         D. 鑴戞簮锛堜繚鐣欙級
#   5. 鑷姩 apply ICA 鈫?杈撳嚭 *_raw_clean_ica_final.fif
#   6. 杈撳嚭鎬昏〃 final_ica_exclude_list.csv
#
# 浣跨敤锛?
#   python ica_auto_clean_final.py
# """
#
# from pathlib import Path
# import numpy as np
# import pandas as pd
# import mne
# from mne.time_frequency import psd_array_welch
#
#
# # ====================== 閰嶇疆 ======================
#
# EEG_ROOT = Path(
#     r"F:\鏁版嵁闆嗗鐞哱data_process\datasetprocess\鏁版嵁棰勫鐞哱鑴戠數鏁版嵁澶勭悊"
# )
#
# # 鍒ゅ畾闃堝€硷紙鍩轰簬浣犳墍鏈夎璇曠殑鍒嗘瀽浼樺寲杩囷級
# STRONG_EOG_CORR = 0.35    # 寮虹溂鍔紙蹇呴』鍓旈櫎锛?
# WEAK_EOG_CORR   = 0.25    # 寮辩溂鍔紙鍙€夊墧闄わ級
# MUSCLE_RATIO_TH = 0.10    # 鑲岀數寮哄害锛?0.1 蹇呴』鍓旈櫎锛?
#
# FRONTAL_CHS = ["Fp1", "Fp2", "AF3", "AF4", "F7", "F8"]
#
#
# # ====================== 宸ュ叿鍑芥暟 ======================
#
# def compute_ica_metrics(raw, ica):
#     """璁＄畻姣忎釜鎴愬垎鐨勫墠棰濋€氶亾鐩稿叧鎬?+ 鑲岀數姣?""
#     src = ica.get_sources(raw).get_data()
#     sfreq = raw.info["sfreq"]
#     n_comp = src.shape[0]
#
#     present_frontal = [ch for ch in FRONTAL_CHS if ch in raw.ch_names]
#
#     # 璁＄畻鐩稿叧鎬?
#     corr_map = {ch: [] for ch in present_frontal}
#     for ch in present_frontal:
#         sig = raw.copy().pick_channels([ch]).get_data()[0]
#         for i in range(n_comp):
#             corr_map[ch].append(np.corrcoef(sig, src[i])[0, 1])
#
#     # 鑲岀數锛堥珮棰?浣庨锛?
#     n_fft = min(2048, src.shape[1])
#     psds, freqs = psd_array_welch(src, sfreq, fmin=1., fmax=45., n_fft=n_fft)
#
#     low_mask = (freqs <= 20)
#     high_mask = (freqs >= 25)
#
#     low_pwr = psds[:, low_mask].mean(axis=1)
#     high_pwr = psds[:, high_mask].mean(axis=1)
#
#     ratio = high_pwr / (low_pwr + 1e-12)
#
#     # 姹囨€荤粨鏋?
#     rows = []
#     for i in range(n_comp):
#         row = {"component": i, "hf_lf_ratio": ratio[i]}
#
#         max_corr = 0
#         best_ch = ""
#
#         for ch in present_frontal:
#             val = corr_map[ch][i]
#             row[f"corr_{ch}"] = val
#             if abs(val) > abs(max_corr):
#                 max_corr, best_ch = val, ch
#
#         row["max_abs_corr"] = abs(max_corr)
#         row["max_corr_ch"] = best_ch
#
#         rows.append(row)
#
#     df = pd.DataFrame(rows)
#     df = df.sort_values("max_abs_corr", ascending=False)
#     return df
#
#
# def classify_components(df):
#     """浜哄伐绾у垎绫?""
#
#     must_remove = []
#     optional_remove = []
#
#     for _, row in df.iterrows():
#         comp = int(row["component"])
#         corr = row["max_abs_corr"]
#         hf_lf = row["hf_lf_ratio"]
#
#         # 鑲岀數锛堝己楂橀锛?
#         if hf_lf > MUSCLE_RATIO_TH:
#             must_remove.append(comp)
#             continue
#
#         # 寮?EOG
#         if corr > STRONG_EOG_CORR:
#             must_remove.append(comp)
#             continue
#
#         # 寮?EOG
#         if corr > WEAK_EOG_CORR:
#             optional_remove.append(comp)
#
#     return must_remove, optional_remove
#
#
# # ====================== 涓荤▼搴?======================
#
# def main():
#     results = []
#
#     print("\n========== 鑷姩浜哄伐绾?ICA 鍘讳吉杩癸紙缁堟瀬鐗堬級=========\n")
#
#     for subj_dir in EEG_ROOT.iterdir():
#         if not subj_dir.is_dir():
#             continue
#         eeg_dir = subj_dir / "eeg_preprocessed_v2"
#         if not eeg_dir.exists():
#             continue
#
#         print(f"\n>>> 琚瘯 {subj_dir.name}")
#
#         for raw_file in eeg_dir.glob("*_raw_clean*.fif"):
#             if "ica_final" in raw_file.name:
#                 continue
#
#             base = raw_file.name.replace("_raw_clean_ica.fif", "").replace("_raw_clean.fif","")
#             ica_path = eeg_dir / f"{base}_ica.fif"
#
#             if not ica_path.exists():
#                 continue
#
#             print(f"\n澶勭悊锛歿raw_file.name}")
#
#             raw = mne.io.read_raw_fif(raw_file, preload=True, verbose=False)
#             ica = mne.preprocessing.read_ica(ica_path, verbose=False)
#
#             # 璁＄畻鎸囨爣
#             df_metrics = compute_ica_metrics(raw, ica)
#
#             # 鍒嗙被
#             must_remove, optional_remove = classify_components(df_metrics)
#
#             print("  蹇呴』鍒犻櫎:", must_remove)
#             print("  鍙€夊垹闄?", optional_remove)
#
#             # 搴旂敤 ICA锛堝彧鍒犲繀椤诲垹闄ょ殑锛?
#             ica.exclude = must_remove
#             cleaned = ica.apply(raw.copy())
#
#             # 淇濆瓨缁撴灉
#             out_file = eeg_dir / f"{base}_raw_clean_ica_final.fif"
#             cleaned.save(out_file, overwrite=True)
#
#             results.append({
#                 "subject": subj_dir.name,
#                 "file": raw_file.name,
#                 "must_remove": ",".join(map(str, must_remove)),
#                 "optional_remove": ",".join(map(str, optional_remove))
#             })
#
#     # 杈撳嚭鎬昏〃
#     df_res = pd.DataFrame(results)
#     df_res.to_csv(EEG_ROOT / "final_ica_exclude_list.csv", index=False, encoding="utf-8-sig")
#
#     print("\n========== 鍏ㄩ儴澶勭悊瀹屾垚锛佸凡杈撳嚭 ICA final 鐗堟湰 ==========\n")
#
#
# if __name__ == "__main__":
#     main()



# #####鍙鍖栨姤鍛?####
# # -*- coding: utf-8 -*-
# """
# eeg_qc_report_batch.py
# ======================
#
# 鎵归噺鐢熸垚 EEG 璐ㄩ噺璇勪及鎶ュ憡锛圚TML锛?
#
# 鍔熻兘锛?
#   - 鑷姩閬嶅巻鈥滆剳鐢垫暟鎹鐞嗏€濈洰褰曚笅鎵€鏈夎璇?
#   - 浼樺厛瀵?*_raw_clean_ica_final.fif 鐢熸垚鎶ュ憡
#       鑻ヤ笉瀛樺湪锛屽垯閫€鑰屾眰鍏舵锛?
#         *_raw_clean_ica.fif -> *_raw_clean.fif
#   - 鎶ュ憡鍐呭锛堜娇鐢?MNE Report锛夛細
#         * 鍘熷淇″彿姒傝锛坮aw 娴忚 + PSD锛?
#         * 閫氶亾缁熻锛堝潎鍊?/ 鏍囧噯宸?/ 宄?宄板€?/ bad 閫氶亾鍒楄〃锛?
#         * ICA 姒傝锛堣嫢瀛樺湪 *_ica.fif锛?
#   - 姣忎釜 EEG 杈撳嚭涓€涓?HTML锛?
#         <base>_eeg_qc_report.html
#
# 浣跨敤锛?
#   python eeg_qc_report_batch.py
# """
#
# from pathlib import Path
# import numpy as np
# import pandas as pd
# import mne
# import matplotlib
# matplotlib.use("Agg")  # 浣跨敤闈炰氦浜掑紡鍚庣锛岄伩鍏嶅崱浣?
#
# # ============== 閰嶇疆 ==============
#
# EEG_ROOT = Path(
#     r"F:\鏁版嵁闆嗗鐞哱data_process\datasetprocess\鏁版嵁棰勫鐞哱鑴戠數鏁版嵁澶勭悊"
# )
#
# # raw.plot 鏃剁殑缂╂斁锛堝彲鏍规嵁闇€瑕佽皟鏁达級
# SCALINGS = dict(eeg=40e-6)  # 40 碌V
#
#
# # ============== 杈呭姪鍑芥暟 ==============
#
# def pick_best_raw_file(eeg_dir: Path):
#     """
#     鍦ㄤ竴涓璇曠殑 eeg_preprocessed_v2 鐩綍涓紝浼樺厛閫夋嫨锛?
#       1) *_raw_clean_ica_final.fif
#       2) *_raw_clean_ica.fif
#       3) *_raw_clean.fif
#     杩斿洖锛歔(raw_path, ica_path鎴朜one, base_name), ...]
#     """
#     raw_files = []
#
#     # 1) final 鐗堟湰
#     for f in sorted(eeg_dir.glob("*_raw_clean_ica_final.fif")):
#         base = f.name.replace("_raw_clean_ica_final.fif", "")
#         ica = eeg_dir / f"{base}_ica.fif"
#         raw_files.append((f, ica if ica.exists() else None, base))
#
#     # 2) 鍙湁 _raw_clean_ica 鐨勬儏鍐?
#     for f in sorted(eeg_dir.glob("*_raw_clean_ica.fif")):
#         base = f.name.replace("_raw_clean_ica.fif", "")
#         # 宸茬粡鏈?final 鐨勫氨涓嶉噸澶?
#         if any(base == b for _, _, b in raw_files):
#             continue
#         ica = eeg_dir / f"{base}_ica.fif"
#         raw_files.append((f, ica if ica.exists() else None, base))
#
#     # 3) 鏈€鍘熷鐨?_raw_clean
#     for f in sorted(eeg_dir.glob("*_raw_clean.fif")):
#         base = f.name.replace("_raw_clean.fif", "")
#         if any(base == b for _, _, b in raw_files):
#             continue
#         ica = eeg_dir / f"{base}_ica.fif"
#         raw_files.append((f, ica if ica.exists() else None, base))
#
#     return raw_files
#
#
# def compute_channel_stats(raw: mne.io.BaseRaw) -> pd.DataFrame:
#     """璁＄畻姣忎釜閫氶亾鐨勭粺璁￠噺锛岀敓鎴?DataFrame锛堜慨澶?NumPy 2.0 ptp锛?""
#
#     data = raw.get_data()
#     ch_names = raw.ch_names
#
#     # 淇 NumPy 2.0 鈥斺€?蹇呴』浣跨敤 np.ptp
#     ptp_vals = np.ptp(data, axis=1)
#
#     df = pd.DataFrame({
#         "ch_name": ch_names,
#         "mean (V)": data.mean(axis=1),
#         "std (V)":  data.std(axis=1),
#         "ptp (V)":  ptp_vals,
#     })
#
#     # 鏍囪鍧忛亾
#     bads = set(raw.info.get("bads", []))
#     df["is_bad"] = df["ch_name"].apply(lambda x: "Yes" if x in bads else "")
#
#     # 绠€鍗曟帓搴忥細鍧忛亾闈犲墠锛宻td 瓒婂ぇ瓒婇潬鍓?
#     df = df.sort_values(by=["is_bad", "std (V)"], ascending=[False, False])
#     return df
#
#
# def make_report_for_one(raw_path: Path, ica_path: Path | None, out_html: Path):
#     """瀵瑰崟涓?raw锛堝強鍙€?ICA锛夌敓鎴愪竴涓?HTML 鎶ュ憡"""
#     print(f"  璇诲彇 Raw: {raw_path.name}")
#     raw = mne.io.read_raw_fif(raw_path, preload=True)
#
#     report_title = f"EEG QC - {raw_path.name}"
#     report = mne.Report(title=report_title)
#
#     # ====== 1. 鍘熷淇″彿 + PSD 姒傝 ======
#     # add_raw 浼氳嚜鍔ㄧ粰鍑洪儴鍒嗘祻瑙堝拰 PSD锛堝湪鏂扮増 MNE 涓級
#     report.add_raw(
#         raw,
#         title="Cleaned EEG (raw overview)",
#         psd=True,
#         tags=("raw", "psd"),
#         scalings=SCALINGS
#     )
#
#     # ====== 2. 閫氶亾缁熻琛?======
#     df_stats = compute_channel_stats(raw)
#     # 杞垚 HTML 琛?
#     html_table = df_stats.to_html(index=False, float_format="%.3e")
#     report.add_html(
#         html=html_table,
#         title="Channel statistics (mean / std / ptp)",
#         tags=("channel-stats",)
#     )
#
#     # ====== 3. ICA 姒傝锛堝鏋滄湁锛?======
#     if ica_path is not None and ica_path.exists():
#         print(f"  璇诲彇 ICA: {ica_path.name}")
#         ica = mne.preprocessing.read_ica(ica_path)
#         # 娣诲姞 ICA 鎴愬垎 topo 鍜岄儴鍒嗗睘鎬?
#         report.add_ica(
#             ica,
#             title="ICA components overview",
#             inst=raw,          # 璁?report 鑳藉睍绀?properties
#             picks=None,
#             tags=("ica",)
#         )
#     else:
#         print("  鏈壘鍒?ICA 鏂囦欢锛岃烦杩?ICA 鍙鍖栥€?)
#
#     # ====== 4. 淇濆瓨鎶ュ憡 ======
#     print(f"  淇濆瓨 HTML 鎶ュ憡: {out_html}")
#     report.save(out_html, overwrite=True, open_browser=False)
#
#
# # ============== 涓绘祦绋?==============
#
# def main():
#     print("========== 鎵归噺鐢熸垚 EEG 璐ㄩ噺鎶ュ憡 (HTML) ==========")
#     print("鏍圭洰褰?", EEG_ROOT)
#
#     for subj_dir in EEG_ROOT.iterdir():
#         if not subj_dir.is_dir():
#             continue
#
#         eeg_dir = subj_dir / "eeg_preprocessed_v2"
#         if not eeg_dir.exists():
#             continue
#
#         print(f"\n>>> 琚瘯: {subj_dir.name}")
#
#         raw_infos = pick_best_raw_file(eeg_dir)
#         if not raw_infos:
#             print("  鈿?鏈壘鍒颁换浣?*_raw_clean*.fif 鏂囦欢锛岃烦杩囪琚瘯銆?)
#             continue
#
#         for raw_path, ica_path, base in raw_infos:
#             out_html = eeg_dir / f"{base}_eeg_qc_report.html"
#
#             try:
#                 make_report_for_one(raw_path, ica_path, out_html)
#             except Exception as e:
#                 print(f"  鉂?鎶ュ憡鐢熸垚澶辫触: {raw_path.name} -> {e}")
#
#     print("\n========== 鍏ㄩ儴鎶ュ憡鐢熸垚瀹屾瘯锛佽鍒板悇琚瘯鐨?eeg_preprocessed_v2 鐩綍鏌ョ湅 .html 鏂囦欢 ==========")
#
#
# if __name__ == "__main__":
#     main()


# -*- coding: utf-8 -*-
"""
ica_auto_clean_final_resamp200.py
=================================

缁堟瀬鑷姩 ICA 鍘讳吉杩硅剼鏈紙閫傞厤 200Hz 鐗堟湰锛?
----------------------------------------

澶勭悊瀵硅薄锛?
  杈撳叆锛?
    - 姣忎釜琚瘯鐩綍涓嬶細
      * eeg_preprocessed_v2/
          - Entity_..._eeg_raw_clean_resamp200.fif   锛?00Hz锛屾纭椂闂磋酱锛?
          - Entity_..._eeg_ica.fif                   锛堝師濮?ICA 妯″瀷锛?

  杈撳嚭锛?
    - Entity_..._eeg_raw_clean_resamp200_ica_final.fif
    - final_ica_exclude_list_resamp200.csv 锛堟€荤粨琛級

浣跨敤锛?
  鍦?predict_2 鐜涓繍琛岋細
    python ica_auto_clean_final_resamp200.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import mne
from mne.time_frequency import psd_array_welch


# ====================== 閰嶇疆 ======================

# 娉ㄦ剰锛氳繖閲岀洿鎺ユ寚鍚戔€滄纭鐞嗙増鏈€?
EEG_ROOT = Path(
    r"F:\data_set_process\data_process\01_datasets\数据预处理\脑电数据处理\正确处理版本"
)

# 鍒ゅ畾闃堝€?
STRONG_EOG_CORR = 0.35    # 寮虹溂鍔紙蹇呴』鍓旈櫎锛?
WEAK_EOG_CORR   = 0.25    # 寮辩溂鍔紙鍙€夊墧闄わ級
MUSCLE_RATIO_TH = 0.10    # 鑲岀數寮哄害闃堝€硷紙>0.1 蹇呴』鍓旈櫎锛?

FRONTAL_CHS = ["Fp1", "Fp2", "AF3", "AF4", "F7", "F8"]


# ====================== 宸ュ叿鍑芥暟 ======================

def compute_ica_metrics(raw, ica):
    """璁＄畻姣忎釜鎴愬垎鐨勫墠棰濋€氶亾鐩稿叧绯绘暟 + 鑲岀數楂樹綆棰戞瘮"""
    src = ica.get_sources(raw).get_data()
    sfreq = raw.info["sfreq"]
    n_comp = src.shape[0]

    present_frontal = [ch for ch in FRONTAL_CHS if ch in raw.ch_names]

    # 璁＄畻涓庡墠棰濋€氶亾鐨勭浉鍏虫€?
    corr_map = {ch: [] for ch in present_frontal}
    for ch in present_frontal:
        sig = raw.copy().pick_channels([ch]).get_data()[0]
        for i in range(n_comp):
            corr_map[ch].append(np.corrcoef(sig, src[i])[0, 1])

    # 鑲岀數锛堥珮棰?浣庨姣斾緥锛?
    n_fft = min(2048, src.shape[1])
    psds, freqs = psd_array_welch(src, sfreq, fmin=1., fmax=45., n_fft=n_fft)

    low_mask = (freqs <= 20)
    high_mask = (freqs >= 25)

    low_pwr = psds[:, low_mask].mean(axis=1)
    high_pwr = psds[:, high_mask].mean(axis=1)
    ratio = high_pwr / (low_pwr + 1e-12)

    rows = []
    for i in range(n_comp):
        row = {"component": i, "hf_lf_ratio": ratio[i]}

        max_corr = 0
        best_ch = ""

        for ch in present_frontal:
            val = corr_map[ch][i]
            row[f"corr_{ch}"] = val
            if abs(val) > abs(max_corr):
                max_corr, best_ch = val, ch

        row["max_abs_corr"] = abs(max_corr)
        row["max_corr_ch"] = best_ch
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("max_abs_corr", ascending=False)
    return df


def classify_components(df):
    """鏍规嵁鐩稿叧鎬?+ 楂橀姣斿垎绫绘垚蹇呴』鍒犻櫎 / 鍙€夊垹闄?""
    must_remove = []
    optional_remove = []

    for _, row in df.iterrows():
        comp = int(row["component"])
        corr = row["max_abs_corr"]
        hf_lf = row["hf_lf_ratio"]

        # 鑲岀數锛堝己楂橀锛?
        if hf_lf > MUSCLE_RATIO_TH:
            must_remove.append(comp)
            continue

        # 寮?EOG
        if corr > STRONG_EOG_CORR:
            must_remove.append(comp)
            continue

        # 寮?EOG
        if corr > WEAK_EOG_CORR:
            optional_remove.append(comp)

    return must_remove, optional_remove


# ====================== 涓荤▼搴?======================

def main():
    results = []

    print("\n========== 鑷姩浜哄伐绾?ICA 鍘讳吉杩癸紙200Hz缁堟瀬鐗堬級=========\n")

    for subj_dir in EEG_ROOT.iterdir():
        if not subj_dir.is_dir():
            continue

        subj_name = subj_dir.name
        eeg_dir = subj_dir / "eeg_preprocessed_v2"
        if not eeg_dir.exists():
            continue

        print(f"\n>>> 琚瘯 {subj_name}")

        # 鍙鐞?宸茬粡闄嶉噰鏍峰埌 200Hz 鐨勬枃浠?
        for raw_file in eeg_dir.glob("*_eeg_raw_clean_resamp200.fif"):
            # 閬垮厤閲嶅澶勭悊宸茬粡 final 鐨?
            if "ica_final" in raw_file.name:
                continue

            # e.g. Entity_Recording_2025_09_26_19_23_28
            base_core = raw_file.name.replace("_eeg_raw_clean_resamp200.fif", "")

            # ICA 妯″瀷浠嶇劧鏄師鏉ョ殑鍛藉悕锛?_eeg_ica.fif
            ica_path = eeg_dir / f"{base_core}_eeg_ica.fif"
            if not ica_path.exists():
                print(f"  鈿?鎵句笉鍒?ICA 妯″瀷: {ica_path.name}锛岃烦杩囪鏂囦欢銆?)
                continue

            print(f"\n澶勭悊锛歿raw_file.name}")
            print(f"浣跨敤 ICA 妯″瀷锛歿ica_path.name}")

            raw = mne.io.read_raw_fif(raw_file, preload=True, verbose=False)
            ica = mne.preprocessing.read_ica(ica_path, verbose=False)

            # 璁＄畻姣忎釜鎴愬垎鐨勬寚鏍?
            df_metrics = compute_ica_metrics(raw, ica)

            # 鍒嗙被
            must_remove, optional_remove = classify_components(df_metrics)
            print("  蹇呴』鍒犻櫎:", must_remove)
            print("  鍙€夊垹闄?", optional_remove)

            # 搴旂敤 ICA锛堝彧鍒犻櫎蹇呴』鍒犻櫎鐨勶級
            ica.exclude = must_remove
            cleaned = ica.apply(raw.copy())

            # 淇濆瓨缁撴灉锛氬湪 200Hz 鐗堟湰涓婂姞 _ica_final 鍚庣紑
            out_file = eeg_dir / f"{base_core}_eeg_raw_clean_resamp200_ica_final.fif"
            cleaned.save(out_file, overwrite=True)
            print(f"  鉁?宸蹭繚瀛? {out_file.name}")

            results.append({
                "subject": subj_name,
                "file": raw_file.name,
                "must_remove": ",".join(map(str, must_remove)),
                "optional_remove": ",".join(map(str, optional_remove))
            })

    if results:
        df_res = pd.DataFrame(results)
        df_res.to_csv(
            EEG_ROOT / "final_ica_exclude_list_resamp200.csv",
            index=False,
            encoding="utf-8-sig"
        )
        print("\n鉁?宸茶緭鍑烘眹鎬昏〃: final_ica_exclude_list_resamp200.csv")
    else:
        print("\n鈿?娌℃湁浠讳綍鏂囦欢琚鐞嗭紝璇锋鏌ヨ矾寰勫強鏂囦欢鍛藉悕銆?)

    print("\n========== 鍏ㄩ儴澶勭悊瀹屾垚锛?200Hz ICA final) ==========\n")


if __name__ == "__main__":
    main()
