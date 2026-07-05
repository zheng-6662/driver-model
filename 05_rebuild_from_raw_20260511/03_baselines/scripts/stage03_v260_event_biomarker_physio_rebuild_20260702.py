#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v260 事件型生理 biomarker 重构与可辨识性诊断。

本轮目的：
- v254b-v259 已经证明“把当前生理表格/序列直接拼进模型”没有稳定提升；
- v260 不再换融合层，而是从 200Hz 连续波形重新构造更有生理含义的事件状态：
  ECG 峰间期/HRV、EDA/SCR 峰与面积、RESP 零交叉/相位、EMG burst；
- 再用 train-only 诊断头检查这些 biomarker 是否比 v254b 统计特征更能解释未来行为和差样本。

边界：
- 只使用 observation_s 之前的数据；
- 不删除样本，不使用 test 后验误差作为部署输入；
- subject-disjoint 是正式泛化口径，subject-aware 仅作为个体化潜力诊断。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


ROOT = Path(r"F:\data_set_process\data_process")
REBUILD = ROOT / "05_rebuild_from_raw_20260511"
BASELINES = REBUILD / "03_baselines"

V252_SCRIPT = BASELINES / "scripts" / "stage03_v252_input_similarity_future_divergence_20260701.py"
V254A_SCRIPT = BASELINES / "scripts" / "stage03_v254a_physio_deep_signal_audit_20260701.py"
V254B_SCRIPT = BASELINES / "scripts" / "stage03_v254b_physio_200hz_event_representation_20260702.py"
PHYSIO_INVENTORY = (
    REBUILD
    / "06_physio_processing"
    / "physio_subject_collection_v1_20260603"
    / "tables"
    / "physio_recording_inventory.csv"
)
V254B_FEATURES = (
    BASELINES
    / "v254b_physio_200hz_event_representation_20260702"
    / "tables"
    / "v254b_event_physio200_features.csv"
)

OUT = BASELINES / "v260_event_biomarker_physio_rebuild_20260702"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
ZIP_PATH = BASELINES / "v260_event_biomarker_physio_rebuild_20260702_pack.zip"

SEED = 26002
TARGETS_FOR_REPORT = [
    "future_cluster4",
    "high_future_abs_q75",
    "high_future_range_q75",
    "strong_steer_existing",
    "bad_top10_v250_diagnostic",
]

PHYSIO_COLS = [
    "ECG_filt200",
    "ECG_raw200",
    "EMG_filt200",
    "EMG_RMS",
    "EDA_filt200",
    "EDA_raw200",
    "EDA_Tonic",
    "EDA_Phasic",
    "RESP_filt200",
    "RESP_raw200",
    "HR_bpm",
    "HRV_RMSSD",
    "t_s",
]

BASELINE_WINDOW = (-60.0, -20.0)
EVENT_WINDOWS = {
    "pre20_pre10": (-20.0, -10.0),
    "pre10_pre5": (-10.0, -5.0),
    "pre5_pre2": (-5.0, -2.0),
    "pre2_0": (-2.0, 0.0),
    "pre5_0": (-5.0, 0.0),
}

mpl.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
mpl.rcParams["axes.unicode_minus"] = False


def import_module_from_path(module_name: str, path: Path):
    """按路径导入前序脚本，复用已验证的数据和目标构造逻辑。"""

    if not path.exists():
        raise FileNotFoundError(f"缺少脚本：{path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法导入脚本：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V252 = import_module_from_path("stage03_v252_for_v260", V252_SCRIPT)
V254A = import_module_from_path("stage03_v254a_for_v260", V254A_SCRIPT)
V254B = import_module_from_path("stage03_v254b_for_v260", V254B_SCRIPT)


def ensure_dirs() -> None:
    """创建输出目录。"""

    for folder in (TABLES, FIGURES, REPORTS, LOGS):
        folder.mkdir(parents=True, exist_ok=True)


def clean_out_dir() -> None:
    """只清理 v260 自己的输出。"""

    if OUT.exists():
        shutil.rmtree(OUT)
    ensure_dirs()
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """CSV 统一使用 utf-8-sig。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def file_sha256(path: Path) -> str:
    """计算输入文件 SHA256。"""

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def session_stamp_from_recording(recording: str) -> str:
    """把 recording 名转成 physio inventory 的 session_stamp。"""

    return str(recording).replace("Entity_Recording_", "")


def finite(values: Iterable[object]) -> np.ndarray:
    """提取有限浮点值。"""

    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def robust_center_scale(values: np.ndarray) -> Tuple[float, float]:
    """用 median + robust scale 描述 baseline。"""

    vals = finite(values)
    if vals.size == 0:
        return math.nan, math.nan
    median = float(np.median(vals))
    q25, q75 = np.quantile(vals, [0.25, 0.75])
    iqr = float(q75 - q25)
    mad = float(np.median(np.abs(vals - median)))
    std = float(np.std(vals))
    for scale in [iqr / 1.349 if iqr > 0 else math.nan, mad * 1.4826 if mad > 0 else math.nan, std]:
        if np.isfinite(scale) and scale > 1e-9:
            return median, float(scale)
    return median, math.nan


def robust_z(values: np.ndarray, baseline_values: np.ndarray) -> np.ndarray:
    """按 baseline 做因果 robust z-score；scale 不可用时返回 NaN。"""

    center, scale = robust_center_scale(baseline_values)
    arr = np.asarray(values, dtype=float)
    if not np.isfinite(center) or not np.isfinite(scale) or scale <= 1e-9:
        return np.full_like(arr, np.nan, dtype=float)
    z = (arr - center) / scale
    z[~np.isfinite(z)] = np.nan
    return z


def slope(times: np.ndarray, vals: np.ndarray) -> float:
    """简单首尾斜率，避免小窗口线性拟合不稳定。"""

    mask = np.isfinite(times) & np.isfinite(vals)
    if int(mask.sum()) < 2:
        return math.nan
    t = times[mask]
    v = vals[mask]
    dt = float(t[-1] - t[0])
    if abs(dt) < 1e-9:
        return math.nan
    return float((v[-1] - v[0]) / dt)


def positive_area(times: np.ndarray, vals: np.ndarray) -> float:
    """正向面积，主要用于 SCR/EMG/RESP 活动强度。"""

    mask = np.isfinite(times) & np.isfinite(vals)
    if int(mask.sum()) < 2:
        return math.nan
    t = times[mask]
    v = np.maximum(vals[mask], 0.0)
    return float(np.trapezoid(v, t))


def abs_area(times: np.ndarray, vals: np.ndarray) -> float:
    """绝对面积。"""

    mask = np.isfinite(times) & np.isfinite(vals)
    if int(mask.sum()) < 2:
        return math.nan
    t = times[mask]
    v = np.abs(vals[mask])
    return float(np.trapezoid(v, t))


def local_peak_indices(vals: np.ndarray, min_distance: int, threshold: float, positive_only: bool = True) -> np.ndarray:
    """
    轻量局部峰检测。

    不依赖 scipy；先找局部极大，再按峰值从大到小做 refractory 抑制。
    """

    arr = np.asarray(vals, dtype=float)
    if arr.size < 3:
        return np.array([], dtype=int)
    work = arr.copy()
    if not positive_only:
        work = np.abs(work)
    finite_mask = np.isfinite(work)
    candidate = (
        finite_mask[1:-1]
        & (work[1:-1] >= work[:-2])
        & (work[1:-1] > work[2:])
        & (work[1:-1] >= threshold)
    )
    idx = np.where(candidate)[0] + 1
    if idx.size == 0:
        return idx.astype(int)
    order = idx[np.argsort(work[idx])[::-1]]
    selected: List[int] = []
    blocked = np.zeros(arr.size, dtype=bool)
    for i in order:
        if blocked[i]:
            continue
        selected.append(int(i))
        left = max(0, int(i) - int(min_distance))
        right = min(arr.size, int(i) + int(min_distance) + 1)
        blocked[left:right] = True
    selected.sort()
    return np.asarray(selected, dtype=int)


def peak_train_features(times: np.ndarray, vals_z: np.ndarray, prefix: str, sample_hz: float, min_distance_s: float, threshold: float) -> Dict[str, float]:
    """峰列特征：count/rate/IBI/HRV/幅值。"""

    out: Dict[str, float] = {}
    mask = np.isfinite(times) & np.isfinite(vals_z)
    if int(mask.sum()) < 3:
        for key in [
            "peak_count",
            "peak_rate_per_min",
            "peak_amp_mean",
            "peak_amp_p90",
            "ibi_mean_s",
            "ibi_sdnn_s",
            "ibi_rmssd_s",
            "bpm_from_peaks",
        ]:
            out[f"{prefix}_{key}"] = math.nan
        return out
    t = times[mask]
    v = vals_z[mask]
    min_dist = max(1, int(round(min_distance_s * sample_hz)))
    peaks = local_peak_indices(v, min_dist, threshold=threshold, positive_only=True)
    duration = max(1e-6, float(t[-1] - t[0]))
    out[f"{prefix}_peak_count"] = int(len(peaks))
    out[f"{prefix}_peak_rate_per_min"] = float(len(peaks) / duration * 60.0)
    if len(peaks) > 0:
        amps = v[peaks]
        out[f"{prefix}_peak_amp_mean"] = float(np.mean(amps))
        out[f"{prefix}_peak_amp_p90"] = float(np.quantile(amps, 0.90))
    else:
        out[f"{prefix}_peak_amp_mean"] = math.nan
        out[f"{prefix}_peak_amp_p90"] = math.nan
    if len(peaks) >= 3:
        ibi = np.diff(t[peaks])
        out[f"{prefix}_ibi_mean_s"] = float(np.mean(ibi))
        out[f"{prefix}_ibi_sdnn_s"] = float(np.std(ibi))
        out[f"{prefix}_ibi_rmssd_s"] = float(np.sqrt(np.mean(np.square(np.diff(ibi))))) if len(ibi) >= 2 else math.nan
        out[f"{prefix}_bpm_from_peaks"] = float(60.0 / np.mean(ibi)) if np.mean(ibi) > 0 else math.nan
    else:
        out[f"{prefix}_ibi_mean_s"] = math.nan
        out[f"{prefix}_ibi_sdnn_s"] = math.nan
        out[f"{prefix}_ibi_rmssd_s"] = math.nan
        out[f"{prefix}_bpm_from_peaks"] = math.nan
    return out


def burst_features(times: np.ndarray, vals_z: np.ndarray, prefix: str, threshold: float) -> Dict[str, float]:
    """EMG/SCR 类 burst 特征。"""

    out: Dict[str, float] = {}
    mask = np.isfinite(times) & np.isfinite(vals_z)
    if int(mask.sum()) < 2:
        for key in ["burst_rate", "burst_episode_count", "burst_longest_s", "z_p95", "z_abs_area", "z_pos_area", "z_slope"]:
            out[f"{prefix}_{key}"] = math.nan
        return out
    t = times[mask]
    z = vals_z[mask]
    active = z >= threshold
    out[f"{prefix}_burst_rate"] = float(np.mean(active))
    transitions = np.diff(active.astype(int))
    starts = np.where(transitions == 1)[0] + 1
    ends = np.where(transitions == -1)[0] + 1
    if active[0]:
        starts = np.r_[0, starts]
    if active[-1]:
        ends = np.r_[ends, len(active) - 1]
    durations: List[float] = []
    for s, e in zip(starts, ends):
        if e > s:
            durations.append(float(t[e] - t[s]))
    out[f"{prefix}_burst_episode_count"] = int(len(durations))
    out[f"{prefix}_burst_longest_s"] = float(max(durations)) if durations else 0.0
    out[f"{prefix}_z_p95"] = float(np.nanquantile(z, 0.95))
    out[f"{prefix}_z_abs_area"] = abs_area(t, z)
    out[f"{prefix}_z_pos_area"] = positive_area(t, z)
    out[f"{prefix}_z_slope"] = slope(t, z)
    return out


def resp_phase_features(times: np.ndarray, resp_z: np.ndarray, prefix: str) -> Dict[str, float]:
    """从 RESP 波形重算周期、BPM 和窗口末端相位。"""

    out: Dict[str, float] = {}
    mask = np.isfinite(times) & np.isfinite(resp_z)
    if int(mask.sum()) < 5:
        for key in ["zero_up_count", "bpm_zero_up", "period_mean_s", "period_std_s", "phase_sin_end", "phase_cos_end", "z_range", "z_abs_area", "z_slope"]:
            out[f"{prefix}_{key}"] = math.nan
        return out
    t = times[mask]
    z = resp_z[mask]
    z_center = z - np.nanmedian(z)
    up = np.where((z_center[:-1] < 0) & (z_center[1:] >= 0))[0] + 1
    duration = max(1e-6, float(t[-1] - t[0]))
    out[f"{prefix}_zero_up_count"] = int(len(up))
    out[f"{prefix}_bpm_zero_up"] = float(len(up) / duration * 60.0)
    if len(up) >= 2:
        periods = np.diff(t[up])
        out[f"{prefix}_period_mean_s"] = float(np.mean(periods))
        out[f"{prefix}_period_std_s"] = float(np.std(periods))
        last_up = t[up[-1]]
        period = float(np.median(periods))
        phase = ((t[-1] - last_up) / max(period, 1e-6)) % 1.0
        out[f"{prefix}_phase_sin_end"] = float(math.sin(2.0 * math.pi * phase))
        out[f"{prefix}_phase_cos_end"] = float(math.cos(2.0 * math.pi * phase))
    else:
        out[f"{prefix}_period_mean_s"] = math.nan
        out[f"{prefix}_period_std_s"] = math.nan
        out[f"{prefix}_phase_sin_end"] = math.nan
        out[f"{prefix}_phase_cos_end"] = math.nan
    out[f"{prefix}_z_range"] = float(np.nanmax(z) - np.nanmin(z))
    out[f"{prefix}_z_abs_area"] = abs_area(t, z)
    out[f"{prefix}_z_slope"] = slope(t, z)
    return out


def generic_window_features(times: np.ndarray, vals_z: np.ndarray, prefix: str) -> Dict[str, float]:
    """少量通用 z 统计，作为事件型特征的补充。"""

    out: Dict[str, float] = {}
    mask = np.isfinite(times) & np.isfinite(vals_z)
    if int(mask.sum()) < 2:
        for key in ["z_mean", "z_std", "z_p10", "z_p90", "z_range", "z_last_minus_first"]:
            out[f"{prefix}_{key}"] = math.nan
        return out
    t = times[mask]
    z = vals_z[mask]
    out[f"{prefix}_z_mean"] = float(np.mean(z))
    out[f"{prefix}_z_std"] = float(np.std(z))
    out[f"{prefix}_z_p10"] = float(np.quantile(z, 0.10))
    out[f"{prefix}_z_p90"] = float(np.quantile(z, 0.90))
    out[f"{prefix}_z_range"] = float(np.max(z) - np.min(z))
    out[f"{prefix}_z_last_minus_first"] = float(z[-1] - z[0])
    return out


def load_physio_inventory() -> Dict[Tuple[str, str], Path]:
    """读取 200Hz cleaned physio inventory。"""

    inv = pd.read_csv(PHYSIO_INVENTORY, encoding="utf-8-sig")
    out: Dict[Tuple[str, str], Path] = {}
    for _, row in inv.iterrows():
        out[(str(row["subject"]), str(row["session_stamp"]))] = Path(str(row["physio_file"]))
    return out


def read_physio_recording(path: Path) -> pd.DataFrame:
    """读取 v260 需要的 200Hz 生理列。"""

    header = pd.read_csv(path, nrows=0).columns.tolist()
    usecols = [c for c in PHYSIO_COLS if c in header]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    df["t_s"] = pd.to_numeric(df["t_s"], errors="coerce")
    df = df.sort_values("t_s").reset_index(drop=True)
    return df


def slice_by_time(times: np.ndarray, values: np.ndarray, start: float, end: float) -> Tuple[np.ndarray, np.ndarray]:
    """按时间切片，返回 t 和 values。"""

    left = int(np.searchsorted(times, start, side="left"))
    right = int(np.searchsorted(times, end, side="right"))
    return times[left:right], values[left:right]


def extract_recording_biomarkers(recording_df: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    """对同一 recording 的所有 rolling sample 提取事件型 biomarker。"""

    times = pd.to_numeric(recording_df["t_s"], errors="coerce").to_numpy(dtype=float)
    duration = float(np.nanmax(times) - np.nanmin(times)) if len(times) else 0.0
    sample_hz = 200.0
    if len(times) >= 5:
        dt = np.diff(times)
        good_dt = dt[np.isfinite(dt) & (dt > 0)]
        if good_dt.size:
            sample_hz = float(1.0 / np.median(good_dt))

    arrays: Dict[str, np.ndarray] = {}
    for col in PHYSIO_COLS:
        if col == "t_s":
            continue
        if col in recording_df.columns:
            arrays[col] = pd.to_numeric(recording_df[col], errors="coerce").to_numpy(dtype=float)
        else:
            arrays[col] = np.full(len(times), np.nan, dtype=float)

    rows: List[Dict[str, object]] = []
    for _, sample in samples.iterrows():
        obs = float(sample["observation_s"])
        out: Dict[str, object] = {
            "row_index": int(sample["row_index"]),
            "event_uid": str(sample["event_uid"]),
            "subject": str(sample["subject"]),
            "recording": str(sample["recording"]),
            "split": str(sample["split"]),
            "delay_ms": int(sample["delay_ms"]),
            "observation_s": obs,
            "bio260_status": "ok",
            "bio260_sample_hz": sample_hz,
            "bio260_recording_duration_s": duration,
            "bio260_uses_post_observation": False,
        }

        b_start = max(0.0, obs + BASELINE_WINDOW[0])
        b_end = max(0.0, obs + BASELINE_WINDOW[1])
        b_t, b_ecg = slice_by_time(times, arrays["ECG_filt200"], b_start, b_end)
        _, b_eda = slice_by_time(times, arrays["EDA_Phasic"], b_start, b_end)
        if np.isfinite(b_eda).sum() < 10:
            _, b_eda = slice_by_time(times, arrays["EDA_filt200"], b_start, b_end)
        _, b_resp = slice_by_time(times, arrays["RESP_filt200"], b_start, b_end)
        _, b_emg = slice_by_time(times, arrays["EMG_RMS"], b_start, b_end)
        _, b_hr = slice_by_time(times, arrays["HR_bpm"], b_start, b_end)
        _, b_hrv = slice_by_time(times, arrays["HRV_RMSSD"], b_start, b_end)
        out["bio260_baseline_rows"] = int(len(b_t))
        out["bio260_baseline_duration_s"] = float(b_t[-1] - b_t[0]) if len(b_t) >= 2 else 0.0

        for win_name, (offset_start, offset_end) in EVENT_WINDOWS.items():
            start = max(0.0, obs + offset_start)
            end = max(0.0, obs + offset_end)
            if end > obs + 1e-9:
                out["bio260_uses_post_observation"] = True
            win_t, ecg = slice_by_time(times, arrays["ECG_filt200"], start, end)
            _, eda = slice_by_time(times, arrays["EDA_Phasic"], start, end)
            if np.isfinite(eda).sum() < 10:
                _, eda = slice_by_time(times, arrays["EDA_filt200"], start, end)
            _, resp = slice_by_time(times, arrays["RESP_filt200"], start, end)
            _, emg = slice_by_time(times, arrays["EMG_RMS"], start, end)
            _, hr = slice_by_time(times, arrays["HR_bpm"], start, end)
            _, hrv = slice_by_time(times, arrays["HRV_RMSSD"], start, end)

            out[f"bio260_{win_name}_rows"] = int(len(win_t))
            out[f"bio260_{win_name}_duration_s"] = float(win_t[-1] - win_t[0]) if len(win_t) >= 2 else 0.0

            ecg_z = robust_z(ecg, b_ecg)
            eda_z = robust_z(eda, b_eda)
            resp_z = robust_z(resp, b_resp)
            emg_z = robust_z(emg, b_emg)
            hr_z = robust_z(hr, b_hr)
            hrv_z = robust_z(hrv, b_hrv)

            out.update(peak_train_features(win_t, ecg_z, f"bio260_{win_name}_ecg", sample_hz, min_distance_s=0.30, threshold=1.0))
            out.update(burst_features(win_t, eda_z, f"bio260_{win_name}_scr", threshold=1.0))
            out.update(resp_phase_features(win_t, resp_z, f"bio260_{win_name}_resp"))
            out.update(burst_features(win_t, emg_z, f"bio260_{win_name}_emg", threshold=2.0))
            out.update(generic_window_features(win_t, hr_z, f"bio260_{win_name}_hr"))
            out.update(generic_window_features(win_t, hrv_z, f"bio260_{win_name}_hrv_existing"))

        # 关键动态差值：最近 2 秒相对更早窗口的变化。
        for sig in ["ecg_bpm_from_peaks", "scr_z_pos_area", "resp_phase_sin_end", "resp_bpm_zero_up", "emg_burst_rate", "hr_z_mean"]:
            c_recent = f"bio260_pre2_0_{sig}"
            for ref in ["pre5_pre2", "pre10_pre5", "pre20_pre10"]:
                c_ref = f"bio260_{ref}_{sig}"
                if c_recent in out and c_ref in out and np.isfinite(out[c_recent]) and np.isfinite(out[c_ref]):
                    out[f"bio260_delta_pre2_0_minus_{ref}_{sig}"] = float(out[c_recent] - out[c_ref])
                else:
                    out[f"bio260_delta_pre2_0_minus_{ref}_{sig}"] = math.nan

        rows.append(out)
    return pd.DataFrame(rows)


def build_biomarker_features(manifest: pd.DataFrame) -> pd.DataFrame:
    """从 200Hz 连续层构造 v260 event biomarker。"""

    inventory = load_physio_inventory()
    samples = manifest[["event_uid", "subject", "recording", "split", "delay_ms", "observation_s"]].reset_index(names="row_index").copy()
    samples["session_stamp"] = samples["recording"].map(session_stamp_from_recording)
    parts: List[pd.DataFrame] = []
    missing: List[Dict[str, object]] = []
    grouped = samples.groupby(["subject", "session_stamp"], sort=False)
    n_groups = len(grouped)
    for group_i, ((subject, session), g) in enumerate(grouped, start=1):
        path = inventory.get((str(subject), str(session)))
        if path is None or not path.exists():
            print(f"[v260] missing physio recording {group_i}/{n_groups}: subject={subject} session={session} samples={len(g)}", flush=True)
            for _, row in g.iterrows():
                missing.append(
                    {
                        "row_index": int(row["row_index"]),
                        "event_uid": str(row["event_uid"]),
                        "subject": str(row["subject"]),
                        "recording": str(row["recording"]),
                        "split": str(row["split"]),
                        "delay_ms": int(row["delay_ms"]),
                        "observation_s": float(row["observation_s"]),
                        "bio260_status": "missing_recording",
                        "bio260_uses_post_observation": False,
                    }
                )
            continue
        print(f"[v260] extracting biomarkers {group_i}/{n_groups}: subject={subject} session={session} samples={len(g)}", flush=True)
        rec = read_physio_recording(path)
        parts.append(extract_recording_biomarkers(rec, g))
    if missing:
        parts.append(pd.DataFrame(missing))
    out = pd.concat(parts, ignore_index=True).sort_values("row_index").reset_index(drop=True)
    return out


def numeric_feature_columns(df: pd.DataFrame, prefixes: Tuple[str, ...]) -> List[str]:
    """按前缀选择数值特征列。"""

    skip = {"row_index", "delay_ms", "observation_s"}
    cols: List[str] = []
    for col in df.columns:
        if col in skip:
            continue
        if not col.startswith(prefixes):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def clean_train_feature_block(x: np.ndarray, train_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """只保留 train 中非空且有方差的特征列。"""

    x = np.asarray(x, dtype=float)
    train_x = x[train_mask]
    finite_count = np.isfinite(train_x).sum(axis=0)
    train_std = np.nanstd(train_x, axis=0)
    keep = (finite_count >= 20) & np.isfinite(train_std) & (train_std > 1e-12)
    return x[:, keep], keep


def make_feature_blocks(vehicle_x: np.ndarray, bio260: pd.DataFrame, split: np.ndarray, physio200: pd.DataFrame | None) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    """构造 v260 诊断特征块。"""

    train_mask = split == "train"
    bio_cols_all = numeric_feature_columns(bio260, ("bio260_",))
    bio_cols_curated = [
        c
        for c in bio_cols_all
        if not c.endswith("_rows")
        and not c.endswith("_duration_s")
        and "recording_duration" not in c
        and "sample_hz" not in c
        and "baseline" not in c
    ]
    blocks: Dict[str, Tuple[np.ndarray, List[str]]] = {
        "vehicle_only": (vehicle_x.astype(float), [f"vehicle_{i}" for i in range(vehicle_x.shape[1])]),
        "bio260_curated": (bio260[bio_cols_curated].to_numpy(dtype=float), bio_cols_curated),
        "vehicle_plus_bio260_curated": (
            np.concatenate([vehicle_x.astype(float), bio260[bio_cols_curated].to_numpy(dtype=float)], axis=1),
            [f"vehicle_{i}" for i in range(vehicle_x.shape[1])] + bio_cols_curated,
        ),
    }
    if physio200 is not None and len(physio200) == len(bio260):
        physio_cols = V254B.numeric_feature_columns(physio200, ("physio200_",))
        physio_norm = [c for c in physio_cols if "_z_" in c or c.endswith("_index") or "burst_rate" in c]
        physio_curated = [
            c
            for c in physio_norm
            if any(sig in c for sig in ["HR_bpm", "EMG_RMS", "EMG_filt200", "EDA_Phasic", "EDA_Tonic", "RESP_filt200"])
        ]
        blocks["physio200_curated_ref"] = (physio200[physio_curated].to_numpy(dtype=float), physio_curated)
        blocks["vehicle_plus_physio200_curated_ref"] = (
            np.concatenate([vehicle_x.astype(float), physio200[physio_curated].to_numpy(dtype=float)], axis=1),
            [f"vehicle_{i}" for i in range(vehicle_x.shape[1])] + physio_curated,
        )

    feature_blocks: Dict[str, np.ndarray] = {}
    audit_rows: List[Dict[str, object]] = []
    for name, (x, cols) in blocks.items():
        x_keep, keep = clean_train_feature_block(x, train_mask)
        feature_blocks[name] = x_keep
        kept_cols = [c for c, k in zip(cols, keep) if bool(k)]
        audit_rows.append(
            {
                "feature_block": name,
                "raw_dim": int(x.shape[1]),
                "kept_dim": int(x_keep.shape[1]),
                "kept_bio260_columns": int(sum(c.startswith("bio260") for c in kept_cols)),
                "kept_physio200_columns": int(sum(c.startswith("physio200") for c in kept_cols)),
            }
        )
    return feature_blocks, pd.DataFrame(audit_rows)


def safe_auc(y_true: np.ndarray, proba: np.ndarray) -> float:
    """二分类 AUC 安全计算。"""

    if len(np.unique(y_true)) < 2:
        return math.nan
    try:
        return float(roc_auc_score(y_true, proba))
    except Exception:
        return math.nan


def evaluate_classification(feature_blocks: Dict[str, np.ndarray], targets: pd.DataFrame, split: np.ndarray, split_protocol: str) -> pd.DataFrame:
    """分类诊断。"""

    rows: List[Dict[str, object]] = []
    train_mask = split == "train"
    eval_masks = {"val": split == "val", "test": split == "test"}
    for target_col in TARGETS_FOR_REPORT:
        y_raw = targets[target_col].to_numpy()
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        if len(np.unique(y[train_mask])) < 2:
            continue
        for block, x in feature_blocks.items():
            if x.shape[1] == 0:
                continue
            print(f"[v260] {split_protocol} classification target={target_col} block={block} dim={x.shape[1]}", flush=True)
            clf = Pipeline(
                steps=[
                    ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scale", StandardScaler()),
                    (
                        "clf",
                        SGDClassifier(
                            loss="log_loss",
                            max_iter=2500,
                            tol=1e-3,
                            alpha=7e-4,
                            class_weight="balanced",
                            random_state=SEED,
                            n_jobs=1,
                        ),
                    ),
                ]
            )
            clf.fit(x[train_mask], y[train_mask])
            for eval_name, mask in eval_masks.items():
                if int(mask.sum()) == 0:
                    continue
                pred = clf.predict(x[mask])
                row = {
                    "split_protocol": split_protocol,
                    "task_type": "classification",
                    "target": target_col,
                    "feature_block": block,
                    "eval_split": eval_name,
                    "n_eval": int(mask.sum()),
                    "accuracy": float(accuracy_score(y[mask], pred)),
                    "macro_f1": float(f1_score(y[mask], pred, average="macro", zero_division=0)),
                    "auc": math.nan,
                }
                if len(le.classes_) == 2 and hasattr(clf, "predict_proba"):
                    row["auc"] = safe_auc(y[mask], clf.predict_proba(x[mask])[:, 1])
                rows.append(row)
    return add_delta_vs_vehicle(pd.DataFrame(rows), "macro_f1")


def evaluate_regression(feature_blocks: Dict[str, np.ndarray], targets: pd.DataFrame, split: np.ndarray, split_protocol: str) -> pd.DataFrame:
    """未来摘要回归诊断。"""

    rows: List[Dict[str, object]] = []
    train_mask = split == "train"
    eval_masks = {"val": split == "val", "test": split == "test"}
    for target_col in ["future_peak_abs", "future_range", "future_mean_abs", "future_final", "future_slope"]:
        y = pd.to_numeric(targets[target_col], errors="coerce").to_numpy(dtype=float)
        good_train = train_mask & np.isfinite(y)
        if int(good_train.sum()) < 50:
            continue
        for block, x in feature_blocks.items():
            if x.shape[1] == 0:
                continue
            reg = Pipeline(
                steps=[
                    ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scale", StandardScaler()),
                    ("reg", Ridge(alpha=80.0)),
                ]
            )
            reg.fit(x[good_train], y[good_train])
            for eval_name, mask in eval_masks.items():
                good_eval = mask & np.isfinite(y)
                if int(good_eval.sum()) < 10:
                    continue
                pred = reg.predict(x[good_eval])
                rows.append(
                    {
                        "split_protocol": split_protocol,
                        "task_type": "regression",
                        "target": target_col,
                        "feature_block": block,
                        "eval_split": eval_name,
                        "n_eval": int(good_eval.sum()),
                        "r2": float(r2_score(y[good_eval], pred)),
                        "mae": float(mean_absolute_error(y[good_eval], pred)),
                    }
                )
    return add_delta_vs_vehicle(pd.DataFrame(rows), "r2")


def add_delta_vs_vehicle(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    """给每个 block 加相对 vehicle_only 的差值。"""

    if df.empty or metric_col not in df.columns:
        return df
    base = df[df["feature_block"].eq("vehicle_only")][["split_protocol", "target", "eval_split", metric_col]].rename(columns={metric_col: "vehicle_metric"})
    out = df.merge(base, on=["split_protocol", "target", "eval_split"], how="left")
    out[f"delta_{metric_col}_minus_vehicle"] = out[metric_col] - out["vehicle_metric"]
    return out


def eta_squared(feature: np.ndarray, labels: np.ndarray) -> float:
    """离散标签 eta²。"""

    x = np.asarray(feature, dtype=float)
    labels = np.asarray(labels)
    mask = np.isfinite(x) & pd.notna(labels)
    if int(mask.sum()) < 20:
        return math.nan
    x = x[mask]
    labels = labels[mask]
    grand = float(np.mean(x))
    ss_total = float(np.sum((x - grand) ** 2))
    if ss_total <= 1e-12:
        return math.nan
    ss_between = 0.0
    for label in np.unique(labels):
        vals = x[labels == label]
        ss_between += float(len(vals) * (np.mean(vals) - grand) ** 2)
    return float(ss_between / ss_total)


def build_eta(bio260: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """特征对 subject/recording/未来标签的描述性可分性。"""

    cols = numeric_feature_columns(bio260, ("bio260_",))
    eta_targets = {
        "subject": bio260["subject"].astype(str).to_numpy(),
        "recording": bio260["recording"].astype(str).to_numpy(),
        "future_cluster4": targets["future_cluster4"].astype(str).to_numpy(),
        "high_future_abs_q75": targets["high_future_abs_q75"].astype(str).to_numpy(),
        "bad_top10_v250_diagnostic": targets["bad_top10_v250_diagnostic"].astype(str).to_numpy(),
    }
    rows: List[Dict[str, object]] = []
    for target, labels in eta_targets.items():
        for col in cols:
            e = eta_squared(pd.to_numeric(bio260[col], errors="coerce").to_numpy(dtype=float), labels)
            if np.isfinite(e):
                rows.append({"target": target, "feature": col, "eta2": e, "signal": signal_family(col)})
    return pd.DataFrame(rows).sort_values(["target", "eta2"], ascending=[True, False]).reset_index(drop=True)


def signal_family(col: str) -> str:
    """给特征粗分信号族。"""

    for key in ["ecg", "scr", "resp", "emg", "hrv", "hr"]:
        if f"_{key}" in col:
            return key
    return "other"


def summarize_alignment(bio260: pd.DataFrame) -> pd.DataFrame:
    """覆盖与泄漏审计。"""

    rows: List[Dict[str, object]] = []
    for split, g in bio260.groupby("split", dropna=False):
        rows.append(
            {
                "split": split,
                "n": int(len(g)),
                "ok_rate": float(g["bio260_status"].astype(str).eq("ok").mean()),
                "uses_post_observation_rate": float(g["bio260_uses_post_observation"].astype(bool).mean()),
                "baseline_rows_mean": float(pd.to_numeric(g["bio260_baseline_rows"], errors="coerce").mean()),
                "baseline_duration_s_mean": float(pd.to_numeric(g["bio260_baseline_duration_s"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows)


def plot_macro_f1(cls: pd.DataFrame) -> Path:
    """画 subject-disjoint test macro-F1 关键对照。"""

    path = FIGURES / "v260_subject_disjoint_test_macro_f1.png"
    sub = cls[
        cls["split_protocol"].eq("subject_disjoint")
        & cls["eval_split"].eq("test")
        & cls["target"].isin(["future_cluster4", "high_future_abs_q75", "bad_top10_v250_diagnostic"])
        & cls["feature_block"].isin(["vehicle_only", "physio200_curated_ref", "bio260_curated", "vehicle_plus_bio260_curated"])
    ].copy()
    if sub.empty:
        return path
    targets = list(sub["target"].drop_duplicates())
    blocks = ["vehicle_only", "physio200_curated_ref", "bio260_curated", "vehicle_plus_bio260_curated"]
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(targets))
    width = 0.82 / len(blocks)
    for j, block in enumerate(blocks):
        vals = []
        for target in targets:
            r = sub[sub["target"].eq(target) & sub["feature_block"].eq(block)]
            vals.append(float(r["macro_f1"].iloc[0]) if len(r) else np.nan)
        ax.bar(x + (j - (len(blocks) - 1) / 2) * width, vals, width=width, label=block)
    ax.set_xticks(x)
    ax.set_xticklabels(targets, rotation=20, ha="right")
    ax.set_ylabel("macro-F1")
    ax.set_title("v260: subject-disjoint test biomarker diagnostics")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_eta(eta: pd.DataFrame) -> Path:
    """画 eta² top 特征。"""

    path = FIGURES / "v260_eta2_top_features.png"
    sub = eta[eta["target"].isin(["subject", "future_cluster4", "high_future_abs_q75", "bad_top10_v250_diagnostic"])].copy()
    sub = sub.groupby("target", as_index=False).head(10)
    if sub.empty:
        return path
    labels = sub["target"] + " | " + sub["signal"] + " | " + sub["feature"].str.replace("bio260_", "", regex=False).str.slice(0, 42)
    fig, ax = plt.subplots(figsize=(12, max(5, 0.26 * len(sub))))
    ax.barh(np.arange(len(sub)), sub["eta2"].to_numpy())
    ax.set_yticks(np.arange(len(sub)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("eta^2")
    ax.set_title("v260: biomarker eta^2 top features")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_input_hashes() -> None:
    """写输入文件哈希。"""

    rows = []
    for label, path in [
        ("v252_script", V252_SCRIPT),
        ("v254a_script", V254A_SCRIPT),
        ("v254b_script", V254B_SCRIPT),
        ("physio_inventory", PHYSIO_INVENTORY),
        ("v254b_features", V254B_FEATURES),
    ]:
        rows.append({"label": label, "path": str(path), "exists": path.exists(), "sha256": file_sha256(path) if path.exists() else ""})
    write_csv(pd.DataFrame(rows), LOGS / "input_file_hashes.csv")


def write_file_inventory() -> None:
    """写输出清单。"""

    rows = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            rows.append({"relative_path": str(path.relative_to(OUT)), "size_bytes": path.stat().st_size})
    write_csv(pd.DataFrame(rows), LOGS / "file_inventory.csv")


def make_zip() -> bool:
    """打包 v260 输出。"""

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(OUT.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(OUT))
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        return zf.testzip() is None


def write_report(
    alignment: pd.DataFrame,
    block_audit: pd.DataFrame,
    cls: pd.DataFrame,
    reg: pd.DataFrame,
    eta: pd.DataFrame,
    figures: List[Path],
) -> None:
    """写中文报告。"""

    lines: List[str] = []
    lines.append("# v260 事件型生理 biomarker 重构与诊断")
    lines.append("")
    lines.append("## 本轮问题")
    lines.append("")
    lines.append("- v254b-v259 显示：现有生理统计/序列直接拼接或 attention 融合没有形成稳定预测增量。")
    lines.append("- v260 因此改查数据层：从 ECG/EDA/RESP/EMG 连续波形重新派生事件型 biomarker，再评估其可辨识性。")
    lines.append("- 本轮仍不删除样本、不使用 observation_s 之后数据、不使用 test 后验误差做部署输入。")
    lines.append("")
    lines.append("## 生理特征重构")
    lines.append("")
    lines.append("- ECG：按 baseline robust z 后检测局部峰，计算 peak rate、IBI、SDNN、RMSSD、derived BPM。")
    lines.append("- EDA/SCR：使用 EDA_Phasic，若缺失则回退 EDA_filt200，计算正向面积、峰、burst episode。")
    lines.append("- RESP：从 RESP_filt200 重新计算零交叉、周期、BPM 和窗口末端相位。")
    lines.append("- EMG：从 EMG_RMS 计算 burst rate、burst episode、绝对/正向面积和近期变化。")
    lines.append("")
    lines.append("## 对齐覆盖")
    lines.append("")
    lines.append(alignment.to_markdown(index=False))
    lines.append("")
    lines.append("## 特征块")
    lines.append("")
    lines.append(block_audit.to_markdown(index=False))
    lines.append("")
    lines.append("## Subject-disjoint test 分类结果")
    lines.append("")
    focus = cls[
        cls["split_protocol"].eq("subject_disjoint")
        & cls["eval_split"].eq("test")
        & cls["target"].isin(TARGETS_FOR_REPORT)
    ].copy()
    lines.append(
        focus[
            [
                "target",
                "feature_block",
                "n_eval",
                "accuracy",
                "macro_f1",
                "auc",
                "vehicle_metric",
                "delta_macro_f1_minus_vehicle",
            ]
        ].to_markdown(index=False)
    )
    lines.append("")
    lines.append("## Subject-disjoint test 回归结果")
    lines.append("")
    reg_focus = reg[reg["split_protocol"].eq("subject_disjoint") & reg["eval_split"].eq("test")].copy()
    lines.append(
        reg_focus[["target", "feature_block", "n_eval", "r2", "mae", "vehicle_metric", "delta_r2_minus_vehicle"]].to_markdown(index=False)
    )
    lines.append("")
    lines.append("## eta² top")
    lines.append("")
    eta_focus = eta[eta["target"].isin(["subject", "future_cluster4", "high_future_abs_q75", "bad_top10_v250_diagnostic"])].groupby("target", as_index=False).head(12)
    lines.append(eta_focus[["target", "feature", "signal", "eta2"]].to_markdown(index=False))
    lines.append("")
    lines.append("## 判读")
    lines.append("")
    bad = focus[focus["target"].eq("bad_top10_v250_diagnostic")]
    veh = bad[bad["feature_block"].eq("vehicle_only")]
    bio = bad[bad["feature_block"].eq("bio260_curated")]
    vpb = bad[bad["feature_block"].eq("vehicle_plus_bio260_curated")]
    ref = bad[bad["feature_block"].eq("physio200_curated_ref")]
    if len(veh) and len(bio) and len(vpb):
        lines.append(
            f"- bad_top10 subject-disjoint：vehicle macro-F1={float(veh['macro_f1'].iloc[0]):.4f}；"
            f"bio260={float(bio['macro_f1'].iloc[0]):.4f}；"
            f"vehicle+bio260={float(vpb['macro_f1'].iloc[0]):.4f}。"
        )
    if len(ref) and len(bio):
        lines.append(
            f"- 与 v254b 参考相比：physio200_curated_ref bad_top10 macro-F1={float(ref['macro_f1'].iloc[0]):.4f}；"
            f"bio260_curated={float(bio['macro_f1'].iloc[0]):.4f}。"
        )
    lines.append("- 若 bio260 明显超过 physio200_curated_ref，说明数据层重构比旧统计更有价值，可进入 v261 selector/预测实验。")
    lines.append("- 若 vehicle+bio260 仍不超过 vehicle_only，说明即使事件型 biomarker 也没有形成正式跨驾驶员预测增量。")
    lines.append("")
    lines.append("## 关键图")
    lines.append("")
    for fig in figures:
        lines.append(f"- `{fig.relative_to(OUT)}`")
    (REPORTS / "v260_event_biomarker_physio_rebuild_cn.md").write_text("\n".join(lines), encoding="utf-8")


def build_guardrail(split_check: pd.DataFrame, bio260: pd.DataFrame, zip_ok: bool) -> Dict[str, object]:
    """基础 guardrail。"""

    split_ok = True
    if "n_splits" in split_check.columns:
        split_ok = bool((pd.to_numeric(split_check["n_splits"], errors="coerce").fillna(0) <= 1).all())
    return {
        "pass": bool(split_ok and zip_ok and not bool(bio260["bio260_uses_post_observation"].astype(bool).any())),
        "zip_testzip": bool(zip_ok),
        "split_integrity_pass": bool(split_ok),
        "no_post_observation_physio": bool(not bool(bio260["bio260_uses_post_observation"].astype(bool).any())),
        "event_rows": int(len(bio260)),
        "ok_rate": float(bio260["bio260_status"].astype(str).eq("ok").mean()),
    }


def main() -> None:
    print("[v260] event biomarker physio rebuild", flush=True)
    clean_out_dir()
    np.random.seed(SEED)

    loaded = V252.load_fixed_inputs()
    manifest = loaded["data"].manifest.copy()
    sample_metrics = loaded["sample_metrics"].copy()
    split_main = manifest["split"].astype(str).to_numpy()
    split_subject_aware = V254B.make_subject_aware_split(manifest)

    bio260 = build_biomarker_features(manifest)
    write_csv(bio260, TABLES / "v260_event_biomarker_features.csv")
    physio200 = pd.read_csv(V254B_FEATURES, encoding="utf-8-sig") if V254B_FEATURES.exists() else None

    targets = V254A.build_future_targets(loaded["y_true"], sample_metrics, split_main)
    targets, cluster_summary = V254A.add_future_clusters(loaded["y_true"], split_main, targets)
    feature_blocks, block_audit = make_feature_blocks(loaded["x_flat"], bio260, split_main, physio200)
    alignment = summarize_alignment(bio260)
    eta = build_eta(bio260, targets)

    cls_main = evaluate_classification(feature_blocks, targets, split_main, "subject_disjoint")
    reg_main = evaluate_regression(feature_blocks, targets, split_main, "subject_disjoint")
    cls_sa = evaluate_classification(feature_blocks, targets, split_subject_aware, "subject_aware")
    reg_sa = evaluate_regression(feature_blocks, targets, split_subject_aware, "subject_aware")
    cls = pd.concat([cls_main, cls_sa], ignore_index=True)
    reg = pd.concat([reg_main, reg_sa], ignore_index=True)

    split_table = pd.DataFrame(
        {
            "row_index": np.arange(len(manifest)),
            "event_uid": manifest["event_uid"],
            "subject": manifest["subject"],
            "subject_disjoint_split": split_main,
            "subject_aware_split": split_subject_aware,
        }
    )

    write_csv(targets, TABLES / "v260_future_behavior_targets.csv")
    write_csv(cluster_summary, TABLES / "v260_future_cluster_summary.csv")
    write_csv(split_table, TABLES / "v260_split_protocol_table.csv")
    write_csv(alignment, TABLES / "v260_alignment_coverage_summary.csv")
    write_csv(block_audit, TABLES / "v260_feature_block_audit.csv")
    write_csv(eta, TABLES / "v260_biomarker_eta2_by_target_feature.csv")
    write_csv(cls, TABLES / "v260_behavior_classification_diagnostics.csv")
    write_csv(reg, TABLES / "v260_future_summary_regression_diagnostics.csv")

    figures = [plot_macro_f1(cls), plot_eta(eta)]
    write_input_hashes()
    write_file_inventory()
    write_report(alignment, block_audit, cls, reg, eta, figures)
    write_file_inventory()
    zip_ok = make_zip()
    guardrail = build_guardrail(loaded["split_check"], bio260, zip_ok)
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    if not bool(guardrail["pass"]):
        raise AssertionError("v260 guardrail 未通过：\n" + json.dumps(guardrail, ensure_ascii=False, indent=2))
    write_file_inventory()

    focus = cls[
        cls["split_protocol"].eq("subject_disjoint")
        & cls["eval_split"].eq("test")
        & cls["target"].eq("bad_top10_v250_diagnostic")
    ].sort_values("macro_f1", ascending=False)
    print(f"[v260] report={REPORTS / 'v260_event_biomarker_physio_rebuild_cn.md'}", flush=True)
    print(f"[v260] zip={ZIP_PATH}", flush=True)
    if len(focus):
        print(focus[["feature_block", "macro_f1", "delta_macro_f1_minus_vehicle", "auc"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
