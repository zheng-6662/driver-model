# # -*- coding: utf-8 -*-
# """
# ica_auto_clean_final.py
# =======================
#
# 终极自动 ICA 去伪迹脚本（人工级别）
# -------------------------------------
#
# 功能：
#   1. 自动遍历你的 EEG 预处理目录
#   2. 针对每个 EEG 文件读取：
#         *_raw_clean.fif 或 *_raw_clean_ica.fif
#         *_ica.fif
#   3. 自动计算每个成分 ICA source：
#         - 与前额通道（Fp1/Fp2/AF3/AF4/F7/F8）相关系数
#         - 高频/低频能量比（肌电）
#   4. 按人工级标准自动分类：
#         A. 强 EOG（必须删）
#         B. 强肌电（必须删）
#         C. 弱 EOG（可选删除）
#         D. 脑源（保留）
#   5. 自动 apply ICA → 输出 *_raw_clean_ica_final.fif
#   6. 输出总表 final_ica_exclude_list.csv
#
# 使用：
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
# # ====================== 配置 ======================
#
# EEG_ROOT = Path(
#     r"F:\数据集处理\data_process\datasetprocess\数据预处理\脑电数据处理"
# )
#
# # 判定阈值（基于你所有被试的分析优化过）
# STRONG_EOG_CORR = 0.35    # 强眼动（必须剔除）
# WEAK_EOG_CORR   = 0.25    # 弱眼动（可选剔除）
# MUSCLE_RATIO_TH = 0.10    # 肌电强度（>0.1 必须剔除）
#
# FRONTAL_CHS = ["Fp1", "Fp2", "AF3", "AF4", "F7", "F8"]
#
#
# # ====================== 工具函数 ======================
#
# def compute_ica_metrics(raw, ica):
#     """计算每个成分的前额通道相关性 + 肌电比"""
#     src = ica.get_sources(raw).get_data()
#     sfreq = raw.info["sfreq"]
#     n_comp = src.shape[0]
#
#     present_frontal = [ch for ch in FRONTAL_CHS if ch in raw.ch_names]
#
#     # 计算相关性
#     corr_map = {ch: [] for ch in present_frontal}
#     for ch in present_frontal:
#         sig = raw.copy().pick_channels([ch]).get_data()[0]
#         for i in range(n_comp):
#             corr_map[ch].append(np.corrcoef(sig, src[i])[0, 1])
#
#     # 肌电（高频/低频）
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
#     # 汇总结果
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
#     """人工级分类"""
#
#     must_remove = []
#     optional_remove = []
#
#     for _, row in df.iterrows():
#         comp = int(row["component"])
#         corr = row["max_abs_corr"]
#         hf_lf = row["hf_lf_ratio"]
#
#         # 肌电（强高频）
#         if hf_lf > MUSCLE_RATIO_TH:
#             must_remove.append(comp)
#             continue
#
#         # 强 EOG
#         if corr > STRONG_EOG_CORR:
#             must_remove.append(comp)
#             continue
#
#         # 弱 EOG
#         if corr > WEAK_EOG_CORR:
#             optional_remove.append(comp)
#
#     return must_remove, optional_remove
#
#
# # ====================== 主程序 ======================
#
# def main():
#     results = []
#
#     print("\n========== 自动人工级 ICA 去伪迹（终极版）=========\n")
#
#     for subj_dir in EEG_ROOT.iterdir():
#         if not subj_dir.is_dir():
#             continue
#         eeg_dir = subj_dir / "eeg_preprocessed_v2"
#         if not eeg_dir.exists():
#             continue
#
#         print(f"\n>>> 被试 {subj_dir.name}")
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
#             print(f"\n处理：{raw_file.name}")
#
#             raw = mne.io.read_raw_fif(raw_file, preload=True, verbose=False)
#             ica = mne.preprocessing.read_ica(ica_path, verbose=False)
#
#             # 计算指标
#             df_metrics = compute_ica_metrics(raw, ica)
#
#             # 分类
#             must_remove, optional_remove = classify_components(df_metrics)
#
#             print("  必须删除:", must_remove)
#             print("  可选删除:", optional_remove)
#
#             # 应用 ICA（只删必须删除的）
#             ica.exclude = must_remove
#             cleaned = ica.apply(raw.copy())
#
#             # 保存结果
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
#     # 输出总表
#     df_res = pd.DataFrame(results)
#     df_res.to_csv(EEG_ROOT / "final_ica_exclude_list.csv", index=False, encoding="utf-8-sig")
#
#     print("\n========== 全部处理完成！已输出 ICA final 版本 ==========\n")
#
#
# if __name__ == "__main__":
#     main()



# #####可视化报告#####
# # -*- coding: utf-8 -*-
# """
# eeg_qc_report_batch.py
# ======================
#
# 批量生成 EEG 质量评估报告（HTML）
#
# 功能：
#   - 自动遍历“脑电数据处理”目录下所有被试
#   - 优先对 *_raw_clean_ica_final.fif 生成报告
#       若不存在，则退而求其次：
#         *_raw_clean_ica.fif -> *_raw_clean.fif
#   - 报告内容（使用 MNE Report）：
#         * 原始信号概览（raw 浏览 + PSD）
#         * 通道统计（均值 / 标准差 / 峰-峰值 / bad 通道列表）
#         * ICA 概览（若存在 *_ica.fif）
#   - 每个 EEG 输出一个 HTML：
#         <base>_eeg_qc_report.html
#
# 使用：
#   python eeg_qc_report_batch.py
# """
#
# from pathlib import Path
# import numpy as np
# import pandas as pd
# import mne
# import matplotlib
# matplotlib.use("Agg")  # 使用非交互式后端，避免卡住
#
# # ============== 配置 ==============
#
# EEG_ROOT = Path(
#     r"F:\数据集处理\data_process\datasetprocess\数据预处理\脑电数据处理"
# )
#
# # raw.plot 时的缩放（可根据需要调整）
# SCALINGS = dict(eeg=40e-6)  # 40 µV
#
#
# # ============== 辅助函数 ==============
#
# def pick_best_raw_file(eeg_dir: Path):
#     """
#     在一个被试的 eeg_preprocessed_v2 目录中，优先选择：
#       1) *_raw_clean_ica_final.fif
#       2) *_raw_clean_ica.fif
#       3) *_raw_clean.fif
#     返回：[(raw_path, ica_path或None, base_name), ...]
#     """
#     raw_files = []
#
#     # 1) final 版本
#     for f in sorted(eeg_dir.glob("*_raw_clean_ica_final.fif")):
#         base = f.name.replace("_raw_clean_ica_final.fif", "")
#         ica = eeg_dir / f"{base}_ica.fif"
#         raw_files.append((f, ica if ica.exists() else None, base))
#
#     # 2) 只有 _raw_clean_ica 的情况
#     for f in sorted(eeg_dir.glob("*_raw_clean_ica.fif")):
#         base = f.name.replace("_raw_clean_ica.fif", "")
#         # 已经有 final 的就不重复
#         if any(base == b for _, _, b in raw_files):
#             continue
#         ica = eeg_dir / f"{base}_ica.fif"
#         raw_files.append((f, ica if ica.exists() else None, base))
#
#     # 3) 最原始的 _raw_clean
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
#     """计算每个通道的统计量，生成 DataFrame（修复 NumPy 2.0 ptp）"""
#
#     data = raw.get_data()
#     ch_names = raw.ch_names
#
#     # 修复 NumPy 2.0 —— 必须使用 np.ptp
#     ptp_vals = np.ptp(data, axis=1)
#
#     df = pd.DataFrame({
#         "ch_name": ch_names,
#         "mean (V)": data.mean(axis=1),
#         "std (V)":  data.std(axis=1),
#         "ptp (V)":  ptp_vals,
#     })
#
#     # 标记坏道
#     bads = set(raw.info.get("bads", []))
#     df["is_bad"] = df["ch_name"].apply(lambda x: "Yes" if x in bads else "")
#
#     # 简单排序：坏道靠前，std 越大越靠前
#     df = df.sort_values(by=["is_bad", "std (V)"], ascending=[False, False])
#     return df
#
#
# def make_report_for_one(raw_path: Path, ica_path: Path | None, out_html: Path):
#     """对单个 raw（及可选 ICA）生成一个 HTML 报告"""
#     print(f"  读取 Raw: {raw_path.name}")
#     raw = mne.io.read_raw_fif(raw_path, preload=True)
#
#     report_title = f"EEG QC - {raw_path.name}"
#     report = mne.Report(title=report_title)
#
#     # ====== 1. 原始信号 + PSD 概览 ======
#     # add_raw 会自动给出部分浏览和 PSD（在新版 MNE 中）
#     report.add_raw(
#         raw,
#         title="Cleaned EEG (raw overview)",
#         psd=True,
#         tags=("raw", "psd"),
#         scalings=SCALINGS
#     )
#
#     # ====== 2. 通道统计表 ======
#     df_stats = compute_channel_stats(raw)
#     # 转成 HTML 表
#     html_table = df_stats.to_html(index=False, float_format="%.3e")
#     report.add_html(
#         html=html_table,
#         title="Channel statistics (mean / std / ptp)",
#         tags=("channel-stats",)
#     )
#
#     # ====== 3. ICA 概览（如果有） ======
#     if ica_path is not None and ica_path.exists():
#         print(f"  读取 ICA: {ica_path.name}")
#         ica = mne.preprocessing.read_ica(ica_path)
#         # 添加 ICA 成分 topo 和部分属性
#         report.add_ica(
#             ica,
#             title="ICA components overview",
#             inst=raw,          # 让 report 能展示 properties
#             picks=None,
#             tags=("ica",)
#         )
#     else:
#         print("  未找到 ICA 文件，跳过 ICA 可视化。")
#
#     # ====== 4. 保存报告 ======
#     print(f"  保存 HTML 报告: {out_html}")
#     report.save(out_html, overwrite=True, open_browser=False)
#
#
# # ============== 主流程 ==============
#
# def main():
#     print("========== 批量生成 EEG 质量报告 (HTML) ==========")
#     print("根目录:", EEG_ROOT)
#
#     for subj_dir in EEG_ROOT.iterdir():
#         if not subj_dir.is_dir():
#             continue
#
#         eeg_dir = subj_dir / "eeg_preprocessed_v2"
#         if not eeg_dir.exists():
#             continue
#
#         print(f"\n>>> 被试: {subj_dir.name}")
#
#         raw_infos = pick_best_raw_file(eeg_dir)
#         if not raw_infos:
#             print("  ⚠ 未找到任何 *_raw_clean*.fif 文件，跳过该被试。")
#             continue
#
#         for raw_path, ica_path, base in raw_infos:
#             out_html = eeg_dir / f"{base}_eeg_qc_report.html"
#
#             try:
#                 make_report_for_one(raw_path, ica_path, out_html)
#             except Exception as e:
#                 print(f"  ❌ 报告生成失败: {raw_path.name} -> {e}")
#
#     print("\n========== 全部报告生成完毕！请到各被试的 eeg_preprocessed_v2 目录查看 .html 文件 ==========")
#
#
# if __name__ == "__main__":
#     main()


# -*- coding: utf-8 -*-
"""
ica_auto_clean_final_resamp200.py
=================================

终极自动 ICA 去伪迹脚本（适配 200Hz 版本）
----------------------------------------

处理对象：
  输入：
    - 每个被试目录下：
      * eeg_preprocessed_v2/
          - Entity_..._eeg_raw_clean_resamp200.fif   （200Hz，正确时间轴）
          - Entity_..._eeg_ica.fif                   （原始 ICA 模型）

  输出：
    - Entity_..._eeg_raw_clean_resamp200_ica_final.fif
    - final_ica_exclude_list_resamp200.csv （总结表）

使用：
  在 predict_2 环境中运行：
    python ica_auto_clean_final_resamp200.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import mne
from mne.time_frequency import psd_array_welch


# ====================== 配置 ======================

# 注意：这里直接指向“正确处理版本”
EEG_ROOT = Path(
    r"F:\数据集处理\data_process\datasetprocess\数据预处理\脑电数据处理\正确处理版本"
)

# 判定阈值
STRONG_EOG_CORR = 0.35    # 强眼动（必须剔除）
WEAK_EOG_CORR   = 0.25    # 弱眼动（可选剔除）
MUSCLE_RATIO_TH = 0.10    # 肌电强度阈值（>0.1 必须剔除）

FRONTAL_CHS = ["Fp1", "Fp2", "AF3", "AF4", "F7", "F8"]


# ====================== 工具函数 ======================

def compute_ica_metrics(raw, ica):
    """计算每个成分的前额通道相关系数 + 肌电高低频比"""
    src = ica.get_sources(raw).get_data()
    sfreq = raw.info["sfreq"]
    n_comp = src.shape[0]

    present_frontal = [ch for ch in FRONTAL_CHS if ch in raw.ch_names]

    # 计算与前额通道的相关性
    corr_map = {ch: [] for ch in present_frontal}
    for ch in present_frontal:
        sig = raw.copy().pick_channels([ch]).get_data()[0]
        for i in range(n_comp):
            corr_map[ch].append(np.corrcoef(sig, src[i])[0, 1])

    # 肌电（高频/低频比例）
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
    """根据相关性 + 高频比分类成必须删除 / 可选删除"""
    must_remove = []
    optional_remove = []

    for _, row in df.iterrows():
        comp = int(row["component"])
        corr = row["max_abs_corr"]
        hf_lf = row["hf_lf_ratio"]

        # 肌电（强高频）
        if hf_lf > MUSCLE_RATIO_TH:
            must_remove.append(comp)
            continue

        # 强 EOG
        if corr > STRONG_EOG_CORR:
            must_remove.append(comp)
            continue

        # 弱 EOG
        if corr > WEAK_EOG_CORR:
            optional_remove.append(comp)

    return must_remove, optional_remove


# ====================== 主程序 ======================

def main():
    results = []

    print("\n========== 自动人工级 ICA 去伪迹（200Hz终极版）=========\n")

    for subj_dir in EEG_ROOT.iterdir():
        if not subj_dir.is_dir():
            continue

        subj_name = subj_dir.name
        eeg_dir = subj_dir / "eeg_preprocessed_v2"
        if not eeg_dir.exists():
            continue

        print(f"\n>>> 被试 {subj_name}")

        # 只处理 已经降采样到 200Hz 的文件
        for raw_file in eeg_dir.glob("*_eeg_raw_clean_resamp200.fif"):
            # 避免重复处理已经 final 的
            if "ica_final" in raw_file.name:
                continue

            # e.g. Entity_Recording_2025_09_26_19_23_28
            base_core = raw_file.name.replace("_eeg_raw_clean_resamp200.fif", "")

            # ICA 模型仍然是原来的命名：*_eeg_ica.fif
            ica_path = eeg_dir / f"{base_core}_eeg_ica.fif"
            if not ica_path.exists():
                print(f"  ⚠ 找不到 ICA 模型: {ica_path.name}，跳过该文件。")
                continue

            print(f"\n处理：{raw_file.name}")
            print(f"使用 ICA 模型：{ica_path.name}")

            raw = mne.io.read_raw_fif(raw_file, preload=True, verbose=False)
            ica = mne.preprocessing.read_ica(ica_path, verbose=False)

            # 计算每个成分的指标
            df_metrics = compute_ica_metrics(raw, ica)

            # 分类
            must_remove, optional_remove = classify_components(df_metrics)
            print("  必须删除:", must_remove)
            print("  可选删除:", optional_remove)

            # 应用 ICA（只删除必须删除的）
            ica.exclude = must_remove
            cleaned = ica.apply(raw.copy())

            # 保存结果：在 200Hz 版本上加 _ica_final 后缀
            out_file = eeg_dir / f"{base_core}_eeg_raw_clean_resamp200_ica_final.fif"
            cleaned.save(out_file, overwrite=True)
            print(f"  ✅ 已保存: {out_file.name}")

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
        print("\n✅ 已输出汇总表: final_ica_exclude_list_resamp200.csv")
    else:
        print("\n⚠ 没有任何文件被处理，请检查路径及文件命名。")

    print("\n========== 全部处理完成！(200Hz ICA final) ==========\n")


if __name__ == "__main__":
    main()
