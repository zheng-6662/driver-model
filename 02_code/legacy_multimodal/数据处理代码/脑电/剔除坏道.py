# # -*- coding: utf-8 -*-
# import os
# import re
# import warnings
# import numpy as np
# import pandas as pd
# import mne
#
# # =========================
# # 配置区
# # =========================
# ROOT_EEG = r"F:\数据集处理\data_process\datasetprocess\多模态数据\被试数据集合"
# QC_DIR   = r"F:\数据集处理\data_process\datasetprocess\多模态数据\eeg_qc_outputs_v3"
# OUT_DIR  = r"F:\数据集处理\data_process\datasetprocess\多模态数据\eeg_qc_clean_fif_v3"
# os.makedirs(OUT_DIR, exist_ok=True)
#
# QC_SUMMARY = os.path.join(QC_DIR, "eeg_qc_summary_v3.csv")
# BAD_CH     = os.path.join(QC_DIR, "eeg_bad_channels_v3.csv")
#
# # --- 处理策略
# DROP_BADS = True          # True: 直接删坏道（推荐）
# INTERP_BADS = False       # True: 插值坏道（需要 montage，且会引入估计成分）
# OVERWRITE = True
#
# # --- 插值时需要 montage（你如果有标准10-20布局，可用这行；否则保持False）
# MONTAGE_NAME = "standard_1020"
#
# # 屏蔽 MNE 文件名 warning
# warnings.filterwarnings(
#     "ignore",
#     message="This filename .* does not conform to MNE naming conventions.*",
#     category=RuntimeWarning
# )
#
# def find_fif_path(subject: str, file_name: str) -> str:
#     """根据 subject + 原文件名，找到原始 fif 路径"""
#     return os.path.join(ROOT_EEG, subject, "eeg", file_name)
#
# def safe_basename(fn: str) -> str:
#     """生成输出文件名，不改原时间戳结构"""
#     base = fn
#     if base.lower().endswith(".fif"):
#         base = base[:-4]
#     return base + "_qc.fif"
#
# # =========================
# # 读取 QC 决策表
# # =========================
# qc = pd.read_csv(QC_SUMMARY)
# bad = pd.read_csv(BAD_CH)
#
# # 只处理最终保留的
# qc_keep = qc[qc["status_final"] == "keep"].copy()
#
# # bad 表：同一 subject+file 可能多次记录同一通道，去重
# bad = bad[["subject", "file", "ch"]].dropna().drop_duplicates()
#
# # 建索引： (subject,file) -> bad_channels list
# bad_map = {}
# for (subj, fn), grp in bad.groupby(["subject", "file"]):
#     bad_map[(subj, fn)] = sorted(grp["ch"].astype(str).unique().tolist())
#
# # 记录处理日志
# apply_rows = []
#
# # =========================
# # 批量应用：生成 qc fif
# # =========================
# for _, row in qc_keep.iterrows():
#     subj = str(row["subject"])
#     fn = str(row["file"])
#     in_path = find_fif_path(subj, fn)
#
#     if not os.path.exists(in_path):
#         apply_rows.append({
#             "subject": subj, "file": fn, "status": "missing_input_file",
#             "in_path": in_path
#         })
#         continue
#
#     out_path = os.path.join(OUT_DIR, subj)
#     os.makedirs(out_path, exist_ok=True)
#     out_file = os.path.join(out_path, safe_basename(fn))
#
#     if (not OVERWRITE) and os.path.exists(out_file):
#         apply_rows.append({
#             "subject": subj, "file": fn, "status": "skip_exists",
#             "out_file": out_file
#         })
#         continue
#
#     try:
#         raw = mne.io.read_raw_fif(in_path, preload=True, verbose=False)
#     except Exception as e:
#         apply_rows.append({
#             "subject": subj, "file": fn, "status": f"read_error:{repr(e)[:120]}",
#             "in_path": in_path
#         })
#         continue
#
#     # 取坏道名单
#     bad_chs = bad_map.get((subj, fn), [])
#     raw.info["bads"] = bad_chs
#
#     # 如果要插值，需要 montage（否则会报错或插值不可信）
#     if INTERP_BADS:
#         try:
#             montage = mne.channels.make_standard_montage(MONTAGE_NAME)
#             raw.set_montage(montage, on_missing="ignore")
#             raw.interpolate_bads(reset_bads=False, verbose=False)
#         except Exception as e:
#             apply_rows.append({
#                 "subject": subj, "file": fn, "status": f"interp_failed:{repr(e)[:120]}",
#                 "bad_n": len(bad_chs), "bad_list": ",".join(bad_chs)
#             })
#             # 插值失败则退化为直接drop
#             if DROP_BADS:
#                 raw.drop_channels(bad_chs)
#
#     # 默认推荐：直接删坏道
#     if DROP_BADS and (len(bad_chs) > 0):
#         raw.drop_channels(bad_chs)
#
#     # 保存
#     try:
#         raw.save(out_file, overwrite=True, verbose=False)
#         apply_rows.append({
#             "subject": subj, "file": fn, "status": "saved",
#             "bad_n": len(bad_chs), "bad_list": ",".join(bad_chs),
#             "out_file": out_file,
#             "n_ch_out": len(raw.ch_names),
#             "duration_s": float(raw.n_times / raw.info["sfreq"])
#         })
#     except Exception as e:
#         apply_rows.append({
#             "subject": subj, "file": fn, "status": f"save_error:{repr(e)[:120]}",
#             "out_file": out_file
#         })
#
# # 输出应用日志
# apply_df = pd.DataFrame(apply_rows)
# apply_df.to_csv(os.path.join(OUT_DIR, "eeg_qc_apply_log_v3.csv"), index=False, encoding="utf-8-sig")
# print("Done. Clean FIF saved to:", OUT_DIR)
# print("Log:", os.path.join(OUT_DIR, "eeg_qc_apply_log_v3.csv"))



# -*- coding: utf-8 -*-
import os
import shutil
import pandas as pd

# ====== 你需要改这 2 个路径 ======
ROOT_EEG = r"F:\数据集处理\data_process\datasetprocess\多模态数据\被试数据集合"
QC_OUT_DIR = r"F:\数据集处理\data_process\datasetprocess\多模态数据\eeg_qc_clean_fif_v3"  # 你生成 *_qc.fif 的输出目录
LOG_PATH = os.path.join(QC_OUT_DIR, "eeg_qc_apply_log_v3.csv")

# ====== 行为开关 ======
DO_MOVE = True      # True=移动；False=复制（建议先 False 验证一次）
OVERWRITE = True    # 目标已存在同名文件时是否覆盖

def safe_mkdir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def main():
    if not os.path.exists(LOG_PATH):
        raise FileNotFoundError(f"找不到日志文件: {LOG_PATH}")

    df = pd.read_csv(LOG_PATH)

    # 只处理成功保存的条目
    df = df[df["status"].astype(str) == "saved"].copy()

    if df.empty:
        print("日志中没有 status=saved 的记录，未执行任何操作。")
        return

    n_ok = 0
    n_skip_missing = 0
    n_skip_exists = 0
    n_fail = 0

    for _, row in df.iterrows():
        subj = str(row["subject"])
        out_file = str(row["out_file"])  # 源文件（*_qc.fif 的完整路径）

        if not os.path.exists(out_file):
            n_skip_missing += 1
            continue

        # 目标：ROOT_EEG\subj\eeg_clean\
        dst_dir = os.path.join(ROOT_EEG, subj, "eeg_clean")
        safe_mkdir(dst_dir)

        dst_path = os.path.join(dst_dir, os.path.basename(out_file))

        if os.path.exists(dst_path) and (not OVERWRITE):
            n_skip_exists += 1
            continue

        try:
            if OVERWRITE and os.path.exists(dst_path):
                os.remove(dst_path)

            if DO_MOVE:
                shutil.move(out_file, dst_path)
            else:
                shutil.copy2(out_file, dst_path)

            n_ok += 1
        except Exception as e:
            n_fail += 1
            print(f"[FAIL] {subj} | {out_file} -> {dst_path} | {repr(e)}")

    print("=" * 60)
    print("完成。")
    print("成功处理:", n_ok)
    print("源文件缺失跳过:", n_skip_missing)
    print("目标已存在跳过:", n_skip_exists)
    print("失败:", n_fail)
    print("目标目录示例:", os.path.join(ROOT_EEG, "<subject>", "eeg_clean"))

if __name__ == "__main__":
    main()
