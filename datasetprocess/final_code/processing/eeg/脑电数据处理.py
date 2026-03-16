# # -*- coding: utf-8 -*-
# """
# 脑电预处理_V3_batch.py
# =======================
#
# V3 版本升级要点：
#   1. 不再假定 fs=1000Hz，而是从 StorageTime 自动估计真实采样率（≈600Hz）
#   2. 批量遍历所有被试的原始 EEG CSV，输出到 脑电数据处理\<被试>\eeg_preprocessed_v2\
#   3. 处理流程：
#         - 读取 CSV，截掉开头全 NaN 的 EEG 段
#         - 对少量 NaN 插值/前后填充
#         - 用 StorageTime 估计 fs，构建 MNE Raw
#         - 1–40Hz 带通 + 50Hz notch + 平均参考
#         - 简单坏道检测（基于通道 std z-score）
#         - 拟合 ICA，并保存 ICA 模型 + raw_clean + raw_clean_ica
#
# 后续 ICA 成分自动剔除：
#   - 使用你之前的 ica_auto_clean_final.py 即可（不需要改，只是时间轴变正确了）
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
# # =============== 路径配置 ===============
#
# # 原始脑电 CSV 根目录（每个被试一个子文件夹）
# RAW_EEG_ROOT = Path(
#     r"F:\数据集处理\data_process\datasetprocess\数据预处理\原始脑电数据"
# )
#
# # 处理后 EEG 输出根目录（每个被试一个子文件夹）
# OUT_ROOT = Path(
#     r"F:\数据集处理\data_process\datasetprocess\数据预处理\脑电数据处理\正确处理版本"
# )
#
# # EEG 缩放：原始单位若为 µV，则乘 1e-6 变成 V（MNE 标准）
# EEG_SCALE = 1e-6
#
# # 通道名映射：index 为 0-based（channel0 → Ch1）
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
# # =============== 核心函数 ===============
#
# def load_eeg_from_csv(csv_path: Path) -> mne.io.Raw:
#     """
#     读取单个 EEG CSV：
#       - 识别 EEG 通道列
#       - 丢弃开头全 NaN
#       - 插值填充 NaN
#       - 利用 StorageTime 自动估计采样率
#       - 创建带真实时间轴的 Raw 对象
#     """
#     print(f"\n>>> 读取 EEG CSV: {csv_path}")
#     df = pd.read_csv(csv_path)
#
#     # 1) 只取 EEG 通道列
#     eeg_cols = [c for c in df.columns if c.startswith("LSLOutletStreamName-EEG|channel")]
#     eeg_cols = sorted(eeg_cols, key=lambda x: int(x.split("channel")[-1]))
#     if not eeg_cols:
#         raise RuntimeError("未发现 'LSLOutletStreamName-EEG|channelX' 形式的列，请检查 CSV 格式。")
#
#     print(f"检测到 {len(eeg_cols)} 个 EEG 通道列。")
#     df_eeg = df[eeg_cols].copy()
#
#     # 2) 丢弃开头全 NaN 的 EEG 行，并同步裁剪 StorageTime 等
#     all_nan_mask = df_eeg.isna().all(axis=1)
#     if all_nan_mask.iloc[0]:
#         first_non_nan_idx = (~all_nan_mask).idxmax()
#         print(f"  开头存在 EEG 全为 NaN 行，丢弃前 {first_non_nan_idx} 行。")
#         df_eeg = df_eeg.iloc[first_non_nan_idx:].reset_index(drop=True)
#         df = df.iloc[first_non_nan_idx:].reset_index(drop=True)
#     else:
#         print("  开头没有 EEG 全 NaN 行。")
#
#     # 3) 对残余 NaN 做插值 + 前后向填充
#     if df_eeg.isna().any().any():
#         print("  检测到 EEG 中存在 NaN → 进行线性插值 + 前后向填充。")
#         df_eeg = df_eeg.interpolate(axis=0).ffill().bfill()
#
#     # 4) 使用 StorageTime 自动估计采样率
#     print("\n  === 基于 StorageTime 自动估计采样率 ===")
#     if "StorageTime" not in df.columns:
#         raise RuntimeError("CSV 中缺少 StorageTime 列，无法估计采样率。")
#
#     ts = pd.to_datetime(df["StorageTime"], format="%Y/%m/%d %H:%M:%S.%f")
#     dt_sec = (ts.iloc[-1] - ts.iloc[0]).total_seconds()
#     n_samples = len(df_eeg)
#     fs_est = (n_samples - 1) / dt_sec
#
#     print(f"    总时长（StorageTime）: {dt_sec:.3f} 秒")
#     print(f"    样本点数: {n_samples}")
#     print(f"    估计采样率 ≈ {fs_est:.3f} Hz")
#
#     # 5) 转 numpy 并缩放到 Volt
#     data = df_eeg.to_numpy(dtype=float).T * EEG_SCALE
#     n_channels, n_times = data.shape
#     print(f"  EEG 数据维度: {n_channels} 通道 × {n_times} 点")
#
#     # 6) 构造通道名
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
#     print("  Raw 对象已创建。")
#
#     # 可选：添加标准 10-20 电极位置（忽略缺失通道）
#     try:
#         raw.set_montage("standard_1020", on_missing="ignore")
#         print("  已设置 standard_1020 电极位置信息。")
#     except Exception as e:
#         print(f"  设置蒙太奇失败（可以忽略）：{e}")
#
#     return raw
#
#
# def basic_preprocess(raw: mne.io.Raw) -> mne.io.Raw:
#     """
#     1–40 Hz 带通 + 50 Hz 工频陷波 + 平均参考
#     """
#     raw = raw.copy()
#
#     print("  开始 1–40 Hz 带通滤波...")
#     raw.filter(l_freq=1., h_freq=40., fir_design="firwin")
#
#     print("  施加 50 Hz notch 滤波...")
#     raw.notch_filter(freqs=[50.])
#
#     print("  设置平均参考...")
#     raw.set_eeg_reference("average", projection=False)
#
#     return raw
#
#
# def detect_bad_channels_std(raw: mne.io.Raw, z_thresh: float = 4.0):
#     """
#     简单坏道检测：根据通道 std 的 z-score。
#     结果写入 raw.info['bads']，同时打印前若干个通道。
#     """
#     data = raw.get_data()
#     ch_std = data.std(axis=1)
#     mean_std = ch_std.mean()
#     std_std = ch_std.std()
#     z_vals = (ch_std - mean_std) / (std_std + 1e-12)
#
#     bads = [raw.ch_names[i] for i, z in enumerate(z_vals) if z > z_thresh]
#     print("  通道 std z-score Top10：")
#     for name, z in sorted(zip(raw.ch_names, z_vals), key=lambda x: x[1], reverse=True)[:10]:
#         print(f"    {name:>4s}: z = {z:6.2f}")
#
#     if bads:
#         print(f"  检测到疑似坏道（z > {z_thresh}）：{bads}")
#     else:
#         print("  未检测到明显坏道。")
#
#     raw.info["bads"] = bads
#     return bads
#
#
# def run_ica(raw: mne.io.Raw, n_components: int = 20, random_state: int = 97) -> ICA:
#     """
#     拟合 ICA 模型（不做自动剔除），仅保存权重，
#     后续用 ica_auto_clean_final.py 做自动剔除 + apply。
#     """
#     print("  开始 ICA 拟合...")
#     ica = ICA(n_components=n_components, random_state=random_state, max_iter="auto")
#     ica.fit(raw)
#     print("  ICA 拟合完成，n_components =", ica.n_components_)
#     return ica
#
#
# def process_one_csv(csv_path: Path, out_dir: Path):
#     """
#     处理单个原始 EEG CSV，输出：
#       - *_raw_clean.fif
#       - *_raw_clean_ica.fif（此处先不剔除成分，只是占位）
#       - *_ica.fif
#     """
#     out_dir.mkdir(parents=True, exist_ok=True)
#
#     base_name = csv_path.stem  # e.g. Entity_Recording_..._eeg
#     base_name = base_name.replace("_eeg", "")  # 去掉尾部 _eeg，统一命名
#
#     # 1. 加载原始 EEG
#     raw = load_eeg_from_csv(csv_path)
#
#     # 2. 基础预处理
#     raw_clean = basic_preprocess(raw)
#
#     # 3. 坏道检测（只是标记，不插值）
#     detect_bad_channels_std(raw_clean, z_thresh=4.0)
#
#     # 4. 保存 raw_clean
#     out_raw_clean = out_dir / f"{base_name}_eeg_raw_clean.fif"
#     print(f"  保存基础预处理后的 Raw 到: {out_raw_clean}")
#     raw_clean.save(out_raw_clean, overwrite=True)
#
#     # 5. ICA 拟合（在 bad 标记但未插值的 raw_clean 上）
#     ica = run_ica(raw_clean)
#
#     # 6. 保存 ICA 模型
#     out_ica = out_dir / f"{base_name}_eeg_ica.fif"
#     print(f"  保存 ICA 模型到: {out_ica}")
#     ica.save(out_ica, overwrite=True)
#
#     # 7. 先生成一个“未剔除成分”的 raw_clean_ica 版本（方便后续对比/兼容）
#     raw_clean_ica = ica.apply(raw_clean.copy(), exclude=[])
#     out_raw_clean_ica = out_dir / f"{base_name}_eeg_raw_clean_ica.fif"
#     print(f"  保存暂未剔除成分的 Raw 到: {out_raw_clean_ica}")
#     raw_clean_ica.save(out_raw_clean_ica, overwrite=True)
#
#     print("  ✅ 当前文件处理完成。\n")
#
#
# # =============== 批处理主程序 ===============
#
# def main():
#     print("========== EEG 预处理 V3 批处理开始 ==========")
#     print("原始 EEG 根目录:", RAW_EEG_ROOT)
#     print("输出根目录:", OUT_ROOT)
#
#     for subj_dir in RAW_EEG_ROOT.iterdir():
#         if not subj_dir.is_dir():
#             continue
#
#         subj_name = subj_dir.name
#         print(f"\n====== 处理被试：{subj_name} ======")
#
#         out_subj_dir = OUT_ROOT / subj_name / "eeg_preprocessed_v2"
#
#         # 遍历该被试下所有 *_eeg.csv
#         csv_files = sorted(subj_dir.glob("*_eeg.csv"))
#         if not csv_files:
#             print("  ⚠ 未找到 *_eeg.csv，跳过该被试。")
#             continue
#
#         for csv_path in csv_files:
#             try:
#                 process_one_csv(csv_path, out_subj_dir)
#             except Exception as e:
#                 print(f"  ❌ 处理失败：{csv_path.name} -> {e}")
#
#     print("\n========== EEG 预处理 V3 全部完成 ==========")
#
#
# if __name__ == "__main__":
#     main()

# -*- coding: utf-8 -*-
"""
eeg_fix_and_resample_200Hz_v2.py
================================

适配 MNE 最新版本：
  - 不能再使用 raw.info['sfreq'] = fs
  - 必须重新构造 RawArray 来修正采样率（正确时间轴）
  - 然后 resample 到 200Hz
"""

from pathlib import Path
import mne
import pandas as pd
import numpy as np

RAW_CSV_ROOT = Path(
    r"F:\数据集处理\data_process\datasetprocess\数据预处理\原始脑电数据"
)

# 注意：这里指向“正确处理版本”
EEG_PROC_ROOT = Path(
    r"F:\数据集处理\data_process\datasetprocess\数据预处理\脑电数据处理\正确处理版本"
)

TARGET_FS = 200  # 降采样目标

def estimate_fs_from_csv(csv_path: Path):
    df = pd.read_csv(csv_path)
    ts = pd.to_datetime(df["StorageTime"], format="%Y/%m/%d %H:%M:%S.%f")
    dt_sec = (ts.iloc[-1] - ts.iloc[0]).total_seconds()
    n_samples = len(df)
    fs_est = (n_samples - 1) / dt_sec
    return fs_est

def rebuild_raw_with_correct_fs(raw_old: mne.io.Raw, fs_new: float):
    """
    重新构造 RawArray（使用正确的 sampling rate）
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

    print(f"  ✔ 已重新构建 RawArray，正确 fs = {fs_new:.3f}")
    print(f"  ✔ 新时长：{raw_new.times[-1]:.3f} 秒")

    return raw_new

def process_one(csv_path: Path, fif_path: Path, out_path: Path):
    print(f"\n==== 修复 + 降采样 ====")
    print("CSV :", csv_path.name)
    print("FIF :", fif_path.name)

    # 1. 估计真实采样率
    fs_est = estimate_fs_from_csv(csv_path)
    print(f"  真实采样率 ≈ {fs_est:.2f} Hz")

    # 2. 读取原始 FIF
    raw_old = mne.io.read_raw_fif(fif_path, preload=True)

    print(f"  原错误时长：{raw_old.times[-1]:.3f}")

    # 3. 使用正确 fs 重构 Raw
    raw_fixed = rebuild_raw_with_correct_fs(raw_old, fs_est)

    # 4. 降采样到200Hz
    print("  降采样至 200Hz ...")
    raw_200 = raw_fixed.copy().resample(TARGET_FS)
    print(f"  降采样后时长：{raw_200.times[-1]:.3f}")

    # 5. 保存
    raw_200.save(out_path, overwrite=True)
    print(f"  ✔ 保存为：{out_path.name}")


def main():
    print("========== 修复采样率 + 降采样200Hz 开始 ==========")

    for subj_dir in RAW_CSV_ROOT.iterdir():
        if not subj_dir.is_dir():
            continue

        subj = subj_dir.name
        print(f"\n====== 被试：{subj} ======")

        # CSV 文件
        csv_files = sorted(subj_dir.glob("*_eeg.csv"))
        if not csv_files:
            print("  ⚠ 无 CSV，跳过")
            continue

        # FIF 处理目录
        subj_proc = EEG_PROC_ROOT / subj / "eeg_preprocessed_v2"
        if not subj_proc.exists():
            print("  ⚠ 找不到处理后 FIF 目录，跳过")
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

    print("\n========== 全部完成 ==========")
    print("接下来可运行：ica_auto_clean_final.py 进行终极 ICA 清洗")


if __name__ == "__main__":
    main()

