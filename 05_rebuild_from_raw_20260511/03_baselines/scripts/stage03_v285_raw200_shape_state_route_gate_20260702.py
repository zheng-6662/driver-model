#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v285 raw-200Hz signal-shape physiology state route gate.

本轮目的：
- v284 已经证明：继续在 v260 派生 biomarker 上做低身份筛选，仍不能通过
  vehicle top40 歧义候选 route gate；
- v285 因此回到底层 cleaned 200Hz 连续信号本身，重新构造事件前状态：
  质量、短窗形态、导数/突变、呼吸相位、跨信号耦合、个体内 causal past percentile；
- 仍然先做 route gate，不直接训练复杂轨迹融合模型。

边界：
- 只使用 observation_s 之前的数据；
- 不读取 v260/v284 已派生特征表作为输入；
- 特征筛选只用 train split；
- 正式结论只看 validation 选择后的 deployable top1 在 test 上是否超过 latest。
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


ROOT = Path(r"F:\data_set_process\data_process")
REBUILD = ROOT / "05_rebuild_from_raw_20260511"
BASELINES = REBUILD / "03_baselines"
SCRIPTS = BASELINES / "scripts"

V252_SCRIPT = SCRIPTS / "stage03_v252_input_similarity_future_divergence_20260701.py"
V254A_SCRIPT = SCRIPTS / "stage03_v254a_physio_deep_signal_audit_20260701.py"
V284_SCRIPT = SCRIPTS / "stage03_v284_dynamic_low_identity_physio_route_gate_20260702.py"
PHYSIO_INVENTORY = (
    REBUILD
    / "06_physio_processing"
    / "physio_subject_collection_v1_20260603"
    / "tables"
    / "physio_recording_inventory.csv"
)
PHYSIO_SIGNAL_AVAIL = (
    REBUILD
    / "06_physio_processing"
    / "physio_subject_collection_v1_20260603"
    / "tables"
    / "physio_signal_availability_summary.csv"
)

OUT = BASELINES / "v285_raw200_shape_state_route_gate_20260702"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
ZIP_PATH = BASELINES / "v285_raw200_shape_state_route_gate_20260702_pack.zip"

SEED = 28502
MIN_GROUP_N = 5

# 只使用已知可用的 cleaned 200Hz 底层/近底层列；HRV_RMSSD、
# RESP_BPM、RESP_Amplitude 在前序审计中不可用或近常数，因此不进入主特征。
PHYSIO_COLS = [
    "t_s",
    "ECG_filt200",
    "ECG_raw200",
    "EMG_RMS",
    "EMG_filt200",
    "EMG_raw200",
    "EDA_Phasic",
    "EDA_Tonic",
    "EDA_filt200",
    "EDA_raw200",
    "RESP_filt200",
    "RESP_raw200",
    "HR_bpm",
]

SIGNAL_SOURCES = {
    "ecg": ["ECG_filt200", "ECG_raw200"],
    "emg": ["EMG_RMS", "EMG_filt200", "EMG_raw200"],
    "eda": ["EDA_Phasic", "EDA_filt200", "EDA_Tonic", "EDA_raw200"],
    "resp": ["RESP_filt200", "RESP_raw200"],
    "hr": ["HR_bpm"],
}

BASELINE_WINDOW = (-60.0, -20.0)
PAST_CONTEXT_WINDOW = (-180.0, -20.0)
EVENT_WINDOWS = {
    "pre30_pre20": (-30.0, -20.0),
    "pre20_pre10": (-20.0, -10.0),
    "pre10_pre5": (-10.0, -5.0),
    "pre5_pre2": (-5.0, -2.0),
    "pre2_0": (-2.0, 0.0),
    "pre1_0": (-1.0, 0.0),
    "pre5_0": (-5.0, 0.0),
    "pre10_0": (-10.0, 0.0),
}

COUPLING_WINDOWS = ["pre10_0", "pre5_0", "pre2_0"]
SCREEN_TARGETS = [
    "future_cluster4",
    "high_future_abs_q75",
    "high_future_range_q75",
    "strong_steer_existing",
    "bad_top10_v250_diagnostic",
    "bad_top10",
    "vehicle_ambiguous",
    "bad_top10_vehicle_ambiguous",
]

mpl.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
mpl.rcParams["axes.unicode_minus"] = False


def import_module_from_path(module_name: str, path: Path):
    """按路径导入前序脚本，只复用已验证的数据口径和 gate 评估逻辑。"""

    if not path.exists():
        raise FileNotFoundError(f"缺少脚本：{path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法导入脚本：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V252 = import_module_from_path("stage03_v252_for_v285", V252_SCRIPT)
V254A = import_module_from_path("stage03_v254a_for_v285", V254A_SCRIPT)
V284 = import_module_from_path("stage03_v284_for_v285", V284_SCRIPT)


def ensure_dirs() -> None:
    """创建 v285 输出目录。"""

    for folder in (TABLES, FIGURES, REPORTS, LOGS):
        folder.mkdir(parents=True, exist_ok=True)


def clean_out_dir() -> None:
    """只清理 v285 自己的输出，避免影响前序版本。"""

    if OUT.exists():
        shutil.rmtree(OUT)
    ensure_dirs()
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """统一以 utf-8-sig 写 CSV，方便 Excel 和中文环境打开。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def file_sha256(path: Path) -> str:
    """计算文件哈希，供后续追溯。"""

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def session_stamp_from_recording(recording: str) -> str:
    """将 recording 名转成 physiology inventory 中的 session_stamp。"""

    return str(recording).replace("Entity_Recording_", "")


def finite(values: Iterable[object]) -> np.ndarray:
    """提取有限浮点值。"""

    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def robust_center_scale(values: Iterable[object]) -> Tuple[float, float]:
    """用 median + robust scale 表示基线；scale 不可用时返回 NaN。"""

    vals = finite(values)
    if vals.size == 0:
        return math.nan, math.nan
    center = float(np.median(vals))
    q25, q75 = np.quantile(vals, [0.25, 0.75])
    iqr = float(q75 - q25)
    mad = float(np.median(np.abs(vals - center)))
    std = float(np.std(vals))
    for scale in [iqr / 1.349 if iqr > 0 else math.nan, mad * 1.4826 if mad > 0 else math.nan, std]:
        if np.isfinite(scale) and scale > 1e-9:
            return center, float(scale)
    return center, math.nan


def robust_z(values: np.ndarray, baseline_values: np.ndarray) -> np.ndarray:
    """按事件自身锚点前 baseline 做因果 z-score。"""

    center, scale = robust_center_scale(baseline_values)
    arr = np.asarray(values, dtype=float)
    if not np.isfinite(center) or not np.isfinite(scale) or scale <= 1e-9:
        return np.full(arr.shape, np.nan, dtype=float)
    z = (arr - center) / scale
    z[~np.isfinite(z)] = np.nan
    return z


def safe_div(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or abs(den) < 1e-12:
        return math.nan
    return float(num / den)


def slope(times: np.ndarray, vals: np.ndarray) -> float:
    """简单首尾斜率，避免小窗口多项式拟合不稳定。"""

    mask = np.isfinite(times) & np.isfinite(vals)
    if int(mask.sum()) < 2:
        return math.nan
    t = times[mask]
    v = vals[mask]
    dt = float(t[-1] - t[0])
    if abs(dt) < 1e-9:
        return math.nan
    return float((v[-1] - v[0]) / dt)


def load_physio_inventory() -> Dict[Tuple[str, str], Path]:
    """读取 cleaned 200Hz 生理 inventory。"""

    inv = pd.read_csv(PHYSIO_INVENTORY, encoding="utf-8-sig")
    out: Dict[Tuple[str, str], Path] = {}
    for _, row in inv.iterrows():
        out[(str(row["subject"]), str(row["session_stamp"]))] = Path(str(row["physio_file"]))
    return out


def read_physio_recording(path: Path) -> pd.DataFrame:
    """读取 v285 需要的 200Hz 生理列。"""

    header = pd.read_csv(path, nrows=0).columns.tolist()
    usecols = [c for c in PHYSIO_COLS if c in header]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    if "t_s" not in df.columns:
        raise RuntimeError(f"physio 文件缺少 t_s：{path}")
    df["t_s"] = pd.to_numeric(df["t_s"], errors="coerce")
    df = df.sort_values("t_s").reset_index(drop=True)
    return df


def slice_range(times: np.ndarray, values: np.ndarray, start: float, end: float) -> Tuple[np.ndarray, np.ndarray]:
    """按时间切片。"""

    left = int(np.searchsorted(times, start, side="left"))
    right = int(np.searchsorted(times, end, side="right"))
    return times[left:right], values[left:right]


def choose_signal_array(arrays: Dict[str, np.ndarray], candidates: List[str], start_idx: int, end_idx: int) -> Tuple[str, np.ndarray]:
    """在候选列中选择当前窗口有效值最多且非近常数的信号。"""

    best_col = candidates[0]
    best_score = -1.0
    best_arr = arrays.get(best_col, np.array([], dtype=float))[start_idx:end_idx]
    for col in candidates:
        arr = arrays.get(col)
        if arr is None:
            continue
        sub = arr[start_idx:end_idx]
        good = finite(sub)
        if good.size < 5:
            score = float(good.size)
        else:
            score = float(good.size) + min(float(np.nanstd(good)), 10.0)
        if score > best_score:
            best_score = score
            best_col = col
            best_arr = sub
    return best_col, np.asarray(best_arr, dtype=float)


def segment_features(vals: np.ndarray, times: np.ndarray, prefix: str) -> Dict[str, float]:
    """将短窗切成 0.5s 小段，提取局部形态变化。"""

    out: Dict[str, float] = {}
    mask = np.isfinite(vals) & np.isfinite(times)
    if int(mask.sum()) < 4:
        for name in ["bin_mean_std", "bin_mean_max_step", "bin_mean_last_minus_first", "bin_absmax"]:
            out[f"{prefix}_{name}"] = math.nan
        return out

    t = times[mask]
    v = vals[mask]
    start = float(t[0])
    end = float(t[-1])
    if end - start < 0.7:
        for name in ["bin_mean_std", "bin_mean_max_step", "bin_mean_last_minus_first", "bin_absmax"]:
            out[f"{prefix}_{name}"] = math.nan
        return out

    bin_edges = np.arange(start, end + 0.5001, 0.5)
    if bin_edges.size < 3:
        bin_edges = np.linspace(start, end, 4)
    means: List[float] = []
    for left, right in zip(bin_edges[:-1], bin_edges[1:]):
        idx = (t >= left) & (t < right)
        if int(idx.sum()) >= 2:
            means.append(float(np.nanmean(v[idx])))
    m = np.asarray(means, dtype=float)
    if finite(m).size < 2:
        for name in ["bin_mean_std", "bin_mean_max_step", "bin_mean_last_minus_first", "bin_absmax"]:
            out[f"{prefix}_{name}"] = math.nan
        return out
    diff = np.diff(m)
    out[f"{prefix}_bin_mean_std"] = float(np.nanstd(m))
    out[f"{prefix}_bin_mean_max_step"] = float(np.nanmax(np.abs(diff))) if diff.size else math.nan
    out[f"{prefix}_bin_mean_last_minus_first"] = float(m[-1] - m[0])
    out[f"{prefix}_bin_absmax"] = float(np.nanmax(np.abs(m)))
    return out


def run_features(vals: np.ndarray, times: np.ndarray, prefix: str, threshold: float) -> Dict[str, float]:
    """提取超过阈值的连续 episode，兼容 EDA/EMG/ECG burst。"""

    out: Dict[str, float] = {}
    mask = np.isfinite(vals) & np.isfinite(times)
    duration = float(times[mask][-1] - times[mask][0]) if int(mask.sum()) >= 2 else math.nan
    if int(mask.sum()) < 3 or not np.isfinite(duration) or duration <= 0:
        out[f"{prefix}_burst_rate"] = math.nan
        out[f"{prefix}_burst_episode_rate"] = math.nan
        out[f"{prefix}_burst_longest_s"] = math.nan
        return out
    t = times[mask]
    v = vals[mask]
    active = v > threshold
    out[f"{prefix}_burst_rate"] = float(np.mean(active))
    episodes = 0
    longest = 0.0
    i = 0
    while i < len(active):
        if not bool(active[i]):
            i += 1
            continue
        j = i
        while j < len(active) and bool(active[j]):
            j += 1
        episodes += 1
        if j - i >= 2:
            longest = max(longest, float(t[j - 1] - t[i]))
        i = j
    out[f"{prefix}_burst_episode_rate"] = safe_div(float(episodes), duration)
    out[f"{prefix}_burst_longest_s"] = float(longest)
    return out


def peak_features(vals: np.ndarray, times: np.ndarray, prefix: str, sample_hz: float, threshold: float = 1.0) -> Dict[str, float]:
    """用局部峰近似 ECG/RESP 等节律，不依赖 scipy。"""

    out: Dict[str, float] = {}
    v = np.asarray(vals, dtype=float)
    t = np.asarray(times, dtype=float)
    good = np.isfinite(v) & np.isfinite(t)
    if int(good.sum()) < 8:
        for name in ["peak_rate_hz", "peak_amp_mean", "peak_ibi_mean_s", "peak_ibi_std_s"]:
            out[f"{prefix}_{name}"] = math.nan
        return out
    v = v[good]
    t = t[good]
    candidate = np.r_[False, (v[1:-1] > v[:-2]) & (v[1:-1] >= v[2:]) & (v[1:-1] > threshold), False]
    idx = np.flatnonzero(candidate)
    min_dist = max(1, int(round(0.28 * sample_hz)))
    kept: List[int] = []
    for i in idx:
        if not kept or i - kept[-1] >= min_dist:
            kept.append(int(i))
        elif v[i] > v[kept[-1]]:
            kept[-1] = int(i)
    peaks = np.asarray(kept, dtype=int)
    duration = float(t[-1] - t[0])
    out[f"{prefix}_peak_rate_hz"] = safe_div(float(len(peaks)), duration)
    out[f"{prefix}_peak_amp_mean"] = float(np.nanmean(v[peaks])) if len(peaks) else math.nan
    if len(peaks) >= 2:
        ibi = np.diff(t[peaks])
        out[f"{prefix}_peak_ibi_mean_s"] = float(np.nanmean(ibi))
        out[f"{prefix}_peak_ibi_std_s"] = float(np.nanstd(ibi))
    else:
        out[f"{prefix}_peak_ibi_mean_s"] = math.nan
        out[f"{prefix}_peak_ibi_std_s"] = math.nan
    return out


def zero_cross_phase(vals: np.ndarray, times: np.ndarray, prefix: str) -> Dict[str, float]:
    """用上升零交叉近似呼吸相位和周期。"""

    out: Dict[str, float] = {}
    mask = np.isfinite(vals) & np.isfinite(times)
    if int(mask.sum()) < 8:
        for name in ["zero_up_rate_hz", "period_mean_s", "period_std_s", "phase_sin_end", "phase_cos_end"]:
            out[f"{prefix}_{name}"] = math.nan
        return out
    v = vals[mask]
    t = times[mask]
    signs = v >= 0
    up = np.flatnonzero((~signs[:-1]) & signs[1:]) + 1
    duration = float(t[-1] - t[0])
    out[f"{prefix}_zero_up_rate_hz"] = safe_div(float(len(up)), duration)
    if len(up) >= 2:
        periods = np.diff(t[up])
        period = float(np.nanmedian(periods))
        out[f"{prefix}_period_mean_s"] = float(np.nanmean(periods))
        out[f"{prefix}_period_std_s"] = float(np.nanstd(periods))
        last_up = float(t[up[-1]])
        phase = 2.0 * math.pi * ((float(t[-1]) - last_up) / max(period, 1e-6))
        out[f"{prefix}_phase_sin_end"] = float(math.sin(phase))
        out[f"{prefix}_phase_cos_end"] = float(math.cos(phase))
    else:
        out[f"{prefix}_period_mean_s"] = math.nan
        out[f"{prefix}_period_std_s"] = math.nan
        out[f"{prefix}_phase_sin_end"] = math.nan
        out[f"{prefix}_phase_cos_end"] = math.nan
    return out


def window_shape_features(times: np.ndarray, raw: np.ndarray, z: np.ndarray, prefix: str, sample_hz: float, signal: str) -> Dict[str, float]:
    """针对一个信号一个窗口提取质量、形态、导数、突变和节律特征。"""

    out: Dict[str, float] = {}
    raw = np.asarray(raw, dtype=float)
    z = np.asarray(z, dtype=float)
    times = np.asarray(times, dtype=float)
    good = np.isfinite(z)
    out[f"{prefix}_n"] = int(len(z))
    out[f"{prefix}_valid_ratio"] = float(np.mean(good)) if len(z) else 0.0
    if int(good.sum()) == 0:
        for name in [
            "z_mean",
            "z_std",
            "z_range",
            "z_abs_mean",
            "z_abs_area_per_s",
            "z_pos_area_per_s",
            "z_neg_area_per_s",
            "z_slope",
            "z_last_minus_first",
            "z_p95_abs",
            "z_outlier3_rate",
            "raw_flat_step_rate",
            "dz_abs_mean",
            "dz_std",
            "dz_p95_abs",
            "dz_sign_change_rate",
        ]:
            out[f"{prefix}_{name}"] = math.nan
        out.update(segment_features(z, times, prefix))
        out.update(run_features(z, times, prefix, threshold=1.0))
        return out

    zg = z[good]
    tg = times[good]
    duration = float(tg[-1] - tg[0]) if len(tg) >= 2 else math.nan
    out[f"{prefix}_z_mean"] = float(np.nanmean(zg))
    out[f"{prefix}_z_std"] = float(np.nanstd(zg))
    out[f"{prefix}_z_range"] = float(np.nanquantile(zg, 0.90) - np.nanquantile(zg, 0.10)) if len(zg) >= 3 else math.nan
    out[f"{prefix}_z_abs_mean"] = float(np.nanmean(np.abs(zg)))
    out[f"{prefix}_z_abs_area_per_s"] = safe_div(float(np.nansum(np.abs(zg))), float(len(zg)))
    out[f"{prefix}_z_pos_area_per_s"] = safe_div(float(np.nansum(np.clip(zg, 0, None))), float(len(zg)))
    out[f"{prefix}_z_neg_area_per_s"] = safe_div(float(np.nansum(np.clip(-zg, 0, None))), float(len(zg)))
    out[f"{prefix}_z_slope"] = slope(tg, zg)
    out[f"{prefix}_z_last_minus_first"] = float(zg[-1] - zg[0]) if len(zg) >= 2 else math.nan
    out[f"{prefix}_z_p95_abs"] = float(np.nanquantile(np.abs(zg), 0.95)) if len(zg) >= 3 else math.nan
    out[f"{prefix}_z_outlier3_rate"] = float(np.nanmean(np.abs(zg) > 3.0))

    rg = raw[good]
    if len(rg) >= 3:
        draw = np.diff(rg)
        out[f"{prefix}_raw_flat_step_rate"] = float(np.nanmean(np.abs(draw) < 1e-9))
    else:
        out[f"{prefix}_raw_flat_step_rate"] = math.nan

    if len(zg) >= 3:
        dt = np.diff(tg)
        dz = np.diff(zg)
        dt = np.where(np.isfinite(dt) & (dt > 1e-9), dt, np.nan)
        deriv = dz / dt
        good_deriv = finite(deriv)
        out[f"{prefix}_dz_abs_mean"] = float(np.nanmean(np.abs(good_deriv))) if good_deriv.size else math.nan
        out[f"{prefix}_dz_std"] = float(np.nanstd(good_deriv)) if good_deriv.size else math.nan
        out[f"{prefix}_dz_p95_abs"] = float(np.nanquantile(np.abs(good_deriv), 0.95)) if good_deriv.size >= 3 else math.nan
        signs = np.sign(dz[np.isfinite(dz)])
        out[f"{prefix}_dz_sign_change_rate"] = float(np.mean(signs[1:] * signs[:-1] < 0)) if len(signs) >= 2 else math.nan
    else:
        for name in ["dz_abs_mean", "dz_std", "dz_p95_abs", "dz_sign_change_rate"]:
            out[f"{prefix}_{name}"] = math.nan

    out.update(segment_features(z, times, prefix))
    if signal in {"emg", "eda"}:
        out.update(run_features(z, times, prefix, threshold=1.0 if signal == "eda" else 2.0))
    elif signal == "ecg":
        out.update(peak_features(z, times, prefix, sample_hz, threshold=1.0))
    elif signal == "resp":
        out.update(zero_cross_phase(z, times, prefix))
    return out


def causal_past_features(past_raw: np.ndarray, win_raw: np.ndarray, prefix: str) -> Dict[str, float]:
    """把短窗状态放到同一 recording 的 causal past 分布里看。"""

    out: Dict[str, float] = {}
    past = finite(past_raw)
    win = finite(win_raw)
    if past.size < 40 or win.size < 3:
        out[f"{prefix}_past_z_mean"] = math.nan
        out[f"{prefix}_past_abs_z_mean"] = math.nan
        out[f"{prefix}_past_percentile_mean"] = math.nan
        return out
    center, scale = robust_center_scale(past)
    wmean = float(np.nanmean(win))
    if np.isfinite(scale) and scale > 1e-9:
        z = float((wmean - center) / scale)
    else:
        z = math.nan
    out[f"{prefix}_past_z_mean"] = z
    out[f"{prefix}_past_abs_z_mean"] = abs(z) if np.isfinite(z) else math.nan
    out[f"{prefix}_past_percentile_mean"] = float(np.mean(past <= wmean))
    return out


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """相关系数安全计算。"""

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = min(len(a), len(b))
    if n < 5:
        return math.nan
    a = a[:n]
    b = b[:n]
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 5:
        return math.nan
    aa = a[mask]
    bb = b[mask]
    if float(np.nanstd(aa)) <= 1e-9 or float(np.nanstd(bb)) <= 1e-9:
        return math.nan
    return float(np.corrcoef(aa, bb)[0, 1])


def coupling_features(window_z: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]], out: Dict[str, object]) -> None:
    """跨信号耦合特征，用来捕捉车辆数据看不出来的生理状态组合。"""

    pairs = [("hr", "emg"), ("hr", "resp"), ("emg", "resp"), ("eda", "resp"), ("eda", "emg"), ("ecg", "resp")]
    for win in COUPLING_WINDOWS:
        for a, b in pairs:
            key_a = (win, a)
            key_b = (win, b)
            if key_a not in window_z or key_b not in window_z:
                out[f"bio285_{win}_{a}_{b}_corr"] = math.nan
                continue
            za = window_z[key_a][1]
            zb = window_z[key_b][1]
            c = safe_corr(za, zb)
            out[f"bio285_{win}_{a}_{b}_corr"] = c
            out[f"bio285_{win}_{a}_{b}_abs_corr"] = abs(c) if np.isfinite(c) else math.nan


def extract_recording_shape_state(recording_df: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    """对同一 recording 的 delay=0 事件提取 v285 底层状态特征。"""

    times = pd.to_numeric(recording_df["t_s"], errors="coerce").to_numpy(dtype=float)
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
            "bio285_status": "ok",
            "bio285_sample_hz": sample_hz,
            "bio285_uses_post_observation": False,
        }

        b_start = max(0.0, obs + BASELINE_WINDOW[0])
        b_end = max(0.0, obs + BASELINE_WINDOW[1])
        b_left = int(np.searchsorted(times, b_start, side="left"))
        b_right = int(np.searchsorted(times, b_end, side="right"))
        out["bio285_baseline_rows"] = int(max(0, b_right - b_left))
        out["bio285_baseline_duration_s"] = float(times[b_right - 1] - times[b_left]) if b_right - b_left >= 2 else 0.0

        past_start = max(0.0, obs + PAST_CONTEXT_WINDOW[0])
        past_end = max(0.0, obs + PAST_CONTEXT_WINDOW[1])
        p_left = int(np.searchsorted(times, past_start, side="left"))
        p_right = int(np.searchsorted(times, past_end, side="right"))
        out["bio285_past_context_rows"] = int(max(0, p_right - p_left))

        signal_arrays: Dict[str, Dict[str, object]] = {}
        for sig, candidates in SIGNAL_SOURCES.items():
            chosen_col, baseline = choose_signal_array(arrays, candidates, b_left, b_right)
            signal_arrays[sig] = {"column": chosen_col, "baseline": baseline, "full": arrays.get(chosen_col, np.full(len(times), np.nan))}
            out[f"bio285_{sig}_chosen_col_code"] = float(candidates.index(chosen_col)) if chosen_col in candidates else math.nan
            out[f"bio285_{sig}_baseline_valid_ratio"] = float(np.isfinite(baseline).mean()) if len(baseline) else 0.0

        window_z: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}
        for win_name, (offset_start, offset_end) in EVENT_WINDOWS.items():
            start = max(0.0, obs + offset_start)
            end = max(0.0, obs + offset_end)
            if end > obs + 1e-9:
                out["bio285_uses_post_observation"] = True
            left = int(np.searchsorted(times, start, side="left"))
            right = int(np.searchsorted(times, end, side="right"))
            win_t = times[left:right]
            out[f"bio285_{win_name}_rows"] = int(max(0, right - left))
            out[f"bio285_{win_name}_duration_s"] = float(win_t[-1] - win_t[0]) if len(win_t) >= 2 else 0.0

            for sig, info in signal_arrays.items():
                full = np.asarray(info["full"], dtype=float)
                baseline = np.asarray(info["baseline"], dtype=float)
                raw = full[left:right]
                z = robust_z(raw, baseline)
                prefix = f"bio285_{win_name}_{sig}"
                out.update(window_shape_features(win_t, raw, z, prefix, sample_hz, sig))
                window_z[(win_name, sig)] = (win_t, z)

                # 个体内 causal past 特征只在最近窗口做，避免把 long recording identity 放大。
                if win_name in {"pre10_0", "pre5_0", "pre2_0"}:
                    past_raw = full[p_left:p_right]
                    out.update(causal_past_features(past_raw, raw, prefix))

        for sig in SIGNAL_SOURCES:
            # 最近状态相对更早状态的显式变化，专门针对“事件前几秒很像但未来分叉”的问题。
            for metric in ["z_mean", "z_std", "z_range", "z_slope", "z_abs_mean", "dz_abs_mean", "bin_mean_max_step"]:
                recent = f"bio285_pre2_0_{sig}_{metric}"
                for ref in ["pre5_pre2", "pre10_pre5", "pre20_pre10", "pre30_pre20"]:
                    old = f"bio285_{ref}_{sig}_{metric}"
                    key = f"bio285_delta_pre2_0_minus_{ref}_{sig}_{metric}"
                    rv = out.get(recent, math.nan)
                    ov = out.get(old, math.nan)
                    out[key] = float(rv - ov) if np.isfinite(rv) and np.isfinite(ov) else math.nan

        coupling_features(window_z, out)
        rows.append(out)
    return pd.DataFrame(rows)


def build_raw200_shape_features(manifest: pd.DataFrame) -> pd.DataFrame:
    """从 200Hz 连续层直接构造 delay=0 事件状态特征。"""

    inventory = load_physio_inventory()
    samples = manifest[manifest["delay_ms"].eq(0)][["event_uid", "subject", "recording", "split", "delay_ms", "observation_s"]].copy()
    samples = samples.reset_index(names="row_index")
    samples["session_stamp"] = samples["recording"].map(session_stamp_from_recording)

    parts: List[pd.DataFrame] = []
    missing: List[Dict[str, object]] = []
    grouped = samples.groupby(["subject", "session_stamp"], sort=False)
    for group_i, ((subject, session), g) in enumerate(grouped, start=1):
        path = inventory.get((str(subject), str(session)))
        if path is None or not path.exists():
            print(f"[v285] missing 200Hz physio {group_i}/{len(grouped)} subject={subject} session={session}", flush=True)
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
                        "bio285_status": "missing_recording",
                        "bio285_uses_post_observation": False,
                    }
                )
            continue
        print(f"[v285] extract raw200 shape {group_i}/{len(grouped)} subject={subject} session={session} events={len(g)}", flush=True)
        rec = read_physio_recording(path)
        parts.append(extract_recording_shape_state(rec, g))
    if missing:
        parts.append(pd.DataFrame(missing))
    out = pd.concat(parts, ignore_index=True).sort_values("row_index").reset_index(drop=True)
    return out


def eta_squared(feature: np.ndarray, labels: np.ndarray) -> float:
    """离散标签 eta²，作为 train-only 行为/身份可分性度量。"""

    x = np.asarray(feature, dtype=float)
    lab = np.asarray(labels)
    mask = np.isfinite(x) & pd.notna(lab)
    if int(mask.sum()) < 20:
        return math.nan
    x = x[mask]
    lab = lab[mask]
    grand = float(np.mean(x))
    total = float(np.sum((x - grand) ** 2))
    if total <= 1e-12:
        return math.nan
    between = 0.0
    for one in np.unique(lab):
        vals = x[lab == one]
        between += float(len(vals) * (np.mean(vals) - grand) ** 2)
    return float(between / total)


def numeric_feature_columns(events: pd.DataFrame) -> List[str]:
    """选择可进入 route gate 的 v285 数值特征列。"""

    excluded_substrings = [
        "_n",
        "_rows",
        "_duration_s",
        "sample_hz",
        "uses_post_observation",
        "baseline_rows",
        "baseline_duration",
        "past_context_rows",
        "chosen_col_code",
    ]
    cols: List[str] = []
    for col in events.columns:
        if not col.startswith("bio285_"):
            continue
        if any(s in col for s in excluded_substrings):
            continue
        if pd.api.types.is_numeric_dtype(events[col]):
            cols.append(col)
    return cols


def feature_category(col: str) -> str:
    """按特征名粗分类型，便于解释和构造 feature set。"""

    low = col.lower()
    if "_corr" in low:
        return "coupling"
    if "past_" in low:
        return "causal_past"
    if "valid_ratio" in low or "flat_step" in low or "outlier" in low:
        return "quality"
    if "phase" in low or "zero_up" in low or "period" in low or "peak_" in low or "burst_" in low:
        return "rhythm"
    if "delta_" in low or "dz_" in low or "bin_" in low or "slope" in low or "last_minus" in low:
        return "shape_dynamic"
    return "level_dynamic"


def signal_family(col: str) -> str:
    """识别特征所属信号族。"""

    for sig in ["ecg", "emg", "eda", "resp", "hr"]:
        if f"_{sig}_" in col:
            return sig
    if "_corr" in col:
        return "coupling"
    return "other"


def finite_rate(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return 0.0
    return float(np.isfinite(x).mean())


def feature_screening(events: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """只用 train split 计算行为相关性与 subject/recording 身份惩罚。"""

    train = events[events["split"].astype(str).eq("train")].copy()
    rows: List[Dict[str, object]] = []
    for col in cols:
        x = pd.to_numeric(train[col], errors="coerce").to_numpy(dtype=float)
        behavior_scores = {}
        for target in SCREEN_TARGETS:
            if target in train.columns:
                behavior_scores[target] = eta_squared(x, train[target])
        identity_subject = eta_squared(x, train["subject"])
        identity_recording = eta_squared(x, train["recording"])
        identity_max = max(identity_subject, identity_recording)
        behavior_max = max([v for v in behavior_scores.values() if np.isfinite(v)] or [0.0])
        bad_score = max(
            [
                behavior_scores.get("bad_top10_v250_diagnostic", 0.0),
                behavior_scores.get("bad_top10", 0.0),
                behavior_scores.get("bad_top10_vehicle_ambiguous", 0.0),
            ]
        )
        cat = feature_category(col)
        rows.append(
            {
                "feature": col,
                "feature_category": cat,
                "signal_family": signal_family(col),
                "finite_rate_train": finite_rate(x),
                "behavior_eta_max": float(behavior_max),
                "bad_eta_max": float(bad_score),
                "identity_eta_subject": float(identity_subject) if np.isfinite(identity_subject) else math.nan,
                "identity_eta_recording": float(identity_recording) if np.isfinite(identity_recording) else math.nan,
                "identity_eta_max": float(identity_max) if np.isfinite(identity_max) else math.nan,
                "identity_to_behavior_ratio": float(identity_max / max(behavior_max, 1e-6)) if np.isfinite(identity_max) else math.nan,
                "behavior_identity_score": float(behavior_max / (identity_max + 0.01)) if np.isfinite(identity_max) else 0.0,
                "bad_identity_score": float(bad_score / (identity_max + 0.01)) if np.isfinite(identity_max) else 0.0,
                **{f"eta_{k}": float(v) if np.isfinite(v) else math.nan for k, v in behavior_scores.items()},
            }
        )
    screen = pd.DataFrame(rows)
    return screen.sort_values(["behavior_identity_score", "behavior_eta_max"], ascending=False).reset_index(drop=True)


def choose_feature_sets(screen: pd.DataFrame) -> Dict[str, List[str]]:
    """构造几组互补的 v285 特征集合，用同一 route gate 评估。"""

    usable = screen[screen["finite_rate_train"].ge(0.78)].copy()
    if usable.empty:
        return {}

    def top(df: pd.DataFrame, sort_cols: List[str], n: int) -> List[str]:
        if df.empty:
            return []
        return (
            df.sort_values(sort_cols, ascending=[False] * len(sort_cols))["feature"]
            .drop_duplicates()
            .head(n)
            .astype(str)
            .tolist()
        )

    non_quality = usable[~usable["feature_category"].eq("quality")].copy()
    shape = usable[usable["feature_category"].isin(["shape_dynamic", "rhythm", "level_dynamic"])].copy()
    coupling = usable[usable["feature_category"].eq("coupling")].copy()
    past = usable[usable["feature_category"].eq("causal_past")].copy()
    quality_mix = usable[usable["feature_category"].isin(["quality", "shape_dynamic", "rhythm"])].copy()
    low_identity = non_quality[non_quality["identity_eta_max"].le(0.10)].copy()
    if len(low_identity) < 32:
        low_identity = non_quality.sort_values("identity_eta_max", ascending=True).head(max(32, min(96, len(non_quality))))

    feature_sets = {
        "raw_shape_behavior_top64": top(shape, ["behavior_identity_score", "behavior_eta_max"], 64),
        "raw_shape_bad_top64": top(shape, ["bad_identity_score", "bad_eta_max"], 64),
        "raw_low_identity_top64": top(low_identity, ["behavior_identity_score", "behavior_eta_max"], 64),
        "raw_quality_shape_top64": top(quality_mix, ["behavior_identity_score", "behavior_eta_max"], 64),
        "raw_coupling_top48": top(coupling, ["behavior_identity_score", "behavior_eta_max"], 48),
        "raw_causal_past_top48": top(past, ["behavior_identity_score", "behavior_eta_max"], 48),
    }
    return {k: v for k, v in feature_sets.items() if len(v) >= 8}


def add_targets_and_labels(features: pd.DataFrame, loaded: Dict[str, object], context: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """合并未来行为标签和 v272/v284 统一差样本标签。"""

    manifest = loaded["data"].manifest.copy()
    split_main = manifest["split"].astype(str).to_numpy()
    targets = V254A.build_future_targets(loaded["y_true"], loaded["sample_metrics"], split_main)
    targets, _ = V254A.add_future_clusters(loaded["y_true"], split_main, targets)
    delay0_targets = targets.loc[features["row_index"].to_numpy()].reset_index(drop=True)
    merged = pd.concat([features.reset_index(drop=True), delay0_targets.reset_index(drop=True)], axis=1)
    label_cols = [
        "event_uid",
        "bad_top10",
        "very_bad_top5",
        "vehicle_ambiguous",
        "bad_top10_vehicle_ambiguous",
        "candidate_rmse_std",
        "unique_delay_n",
    ]
    merged = merged.merge(context[label_cols], on="event_uid", how="left")
    for col in ["bad_top10", "very_bad_top5", "vehicle_ambiguous", "bad_top10_vehicle_ambiguous"]:
        merged[col] = merged[col].fillna(False).astype(bool)
    return merged, targets


def summarize_feature_screen(screen: pd.DataFrame) -> pd.DataFrame:
    """按信号族和特征类型汇总 train-only 筛选结果。"""

    if screen.empty:
        return pd.DataFrame()
    return (
        screen.groupby(["feature_category", "signal_family"], as_index=False)
        .agg(
            feature_n=("feature", "count"),
            finite_rate_train_median=("finite_rate_train", "median"),
            behavior_eta_max=("behavior_eta_max", "max"),
            bad_eta_max=("bad_eta_max", "max"),
            identity_eta_median=("identity_eta_max", "median"),
            behavior_identity_score_max=("behavior_identity_score", "max"),
        )
        .sort_values(["behavior_identity_score_max", "behavior_eta_max"], ascending=False)
    )


def table_to_md(df: pd.DataFrame, cols: List[str] | None = None, max_rows: int = 40) -> str:
    """DataFrame 转 markdown，避免空表报错。"""

    if df is None or df.empty:
        return "_空表_"
    show = df.copy()
    if cols is not None:
        show = show[[c for c in cols if c in show.columns]]
    return show.head(max_rows).to_markdown(index=False)


def plot_val_test_delta(summary: pd.DataFrame) -> Path:
    """画 bad_top10 上 val/test top1 delta。"""

    path = FIGURES / "v285_badtop10_val_test_delta.png"
    data = summary[summary["event_group"].eq("bad_top10") & summary["split"].isin(["val", "test"])].copy()
    if data.empty:
        return path
    feature_sets = list(data["feature_set"].drop_duplicates())
    x = np.arange(len(feature_sets))
    width = 0.22
    fig, ax = plt.subplots(figsize=(13, 5))
    for i, (split, col, label) in enumerate(
        [
            ("val", "bio_top1_minus_latest_mean", "val top1"),
            ("test", "bio_top1_minus_latest_mean", "test top1"),
            ("test", "bio_top3_minus_latest_mean", "test top3 oracle"),
        ]
    ):
        vals = []
        for fs in feature_sets:
            sub = data[data["feature_set"].eq(fs) & data["split"].eq(split)]
            vals.append(float(sub[col].iloc[0]) if len(sub) else np.nan)
        ax.bar(x + (i - 1) * width, vals, width=width, label=label)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s).replace("_", "\n") for s in feature_sets], fontsize=8)
    ax.set_ylabel("RMSE delta vs latest, lower is better")
    ax.set_title("v285: raw 200Hz shape-state route gate on bad_top10")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_screen(screen_summary: pd.DataFrame) -> Path:
    """画不同信号/特征类型的行为-身份分离得分。"""

    path = FIGURES / "v285_feature_screen_by_family.png"
    if screen_summary.empty:
        return path
    data = screen_summary.head(30).copy()
    labels = data["feature_category"].astype(str) + "\n" + data["signal_family"].astype(str)
    fig, ax = plt.subplots(figsize=(12, max(5, 0.28 * len(data))))
    ax.barh(np.arange(len(data)), data["behavior_identity_score_max"].astype(float))
    ax.set_yticks(np.arange(len(data)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("max behavior / identity score, train only")
    ax.set_title("v285: raw 200Hz feature screening")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_corr(summary: pd.DataFrame) -> Path:
    """画歧义差样本的生理距离-真实误差排序相关。"""

    path = FIGURES / "v285_bad_ambiguous_corr.png"
    data = summary[
        summary["event_group"].eq("bad_top10_vehicle_ambiguous")
        & summary["split"].isin(["train", "val", "test"])
    ].copy()
    if data.empty:
        return path
    feature_sets = list(data["feature_set"].drop_duplicates())
    x = np.arange(len(feature_sets))
    width = 0.24
    fig, ax = plt.subplots(figsize=(13, 5))
    for i, split in enumerate(["train", "val", "test"]):
        vals = []
        for fs in feature_sets:
            sub = data[data["feature_set"].eq(fs) & data["split"].eq(split)]
            vals.append(float(sub["bio_corr_mean"].iloc[0]) if len(sub) else np.nan)
        ax.bar(x + (i - 1) * width, vals, width=width, label=split)
    ax.axhline(0, color="black", linewidth=1)
    ax.axhline(0.05, color="tab:red", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s).replace("_", "\n") for s in feature_sets], fontsize=8)
    ax.set_ylabel("rank corr: physiology distance vs candidate RMSE")
    ax.set_title("v285: bad_top10 + vehicle_ambiguous rank correlation")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_input_hashes() -> None:
    """写输入哈希；注意这里不把 v260/v284 特征表列为输入。"""

    rows = []
    for label, path in [
        ("v252_script", V252_SCRIPT),
        ("v254a_script", V254A_SCRIPT),
        ("v284_script_for_gate_logic", V284_SCRIPT),
        ("physio_inventory", PHYSIO_INVENTORY),
        ("physio_signal_availability", PHYSIO_SIGNAL_AVAIL),
        ("v278_candidates", V284.V278_CANDIDATES),
        ("v272_diag", V284.V272_DIAG),
        ("v283_guardrail", V284.V283_GUARDRAIL),
    ]:
        rows.append(
            {
                "label": label,
                "path": str(path),
                "exists": bool(path.exists()),
                "sha256": file_sha256(path) if path.exists() else "",
            }
        )
    write_csv(pd.DataFrame(rows), LOGS / "input_hashes.csv")


def write_file_inventory() -> None:
    """写输出文件清单。"""

    rows = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            rows.append({"relative_path": str(path.relative_to(OUT)), "size_bytes": int(path.stat().st_size)})
    write_csv(pd.DataFrame(rows), LOGS / "file_inventory.csv")


def make_zip() -> bool:
    """打包 v285 输出并自检。"""

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(__file__, arcname=f"scripts/{Path(__file__).name}")
        for folder in [TABLES, FIGURES, REPORTS, LOGS]:
            for path in sorted(folder.rglob("*")):
                if path.is_file():
                    zf.write(path, arcname=str(path.relative_to(OUT)))
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        return zf.testzip() is None


def write_report(
    feature_audit: pd.DataFrame,
    screen: pd.DataFrame,
    screen_summary: pd.DataFrame,
    summary: pd.DataFrame,
    val_test: pd.DataFrame,
    decision: pd.DataFrame,
    guardrail: Dict[str, object],
    figures: List[Path],
) -> Path:
    """写中文报告。"""

    path = REPORTS / "v285_raw200_shape_state_route_gate_cn.md"
    bad = summary[summary["event_group"].eq("bad_top10") & summary["split"].isin(["val", "test"])].sort_values(
        ["split", "bio_top1_minus_latest_mean"]
    )
    amb = summary[
        summary["event_group"].eq("bad_top10_vehicle_ambiguous")
        & summary["split"].isin(["val", "test"])
    ].sort_values(["split", "bio_top1_minus_latest_mean"])

    lines: List[str] = []
    lines.append("# v285 raw-200Hz signal-shape physiology route gate")
    lines.append("")
    lines.append("## 本轮目的")
    lines.append("")
    lines.append("- 承接 v284：不再复用 v260 biomarker 做筛选，而是直接从 cleaned 200Hz 连续信号重算事件前状态。")
    lines.append("- 重点特征包括质量、短窗形态、导数/突变、节律/相位、跨信号耦合、个体内 causal past percentile。")
    lines.append("- 仍然先在 v278 vehicle top40 候选池中过 route gate，未通过则不进入复杂融合轨迹模型。")
    lines.append("")
    lines.append("## route gate 判定")
    lines.append("")
    lines.append(table_to_md(decision, ["check", "requirement", "pass", "evidence", "deployable", "route_viable_now"]))
    lines.append("")
    lines.append("## feature set 审计")
    lines.append("")
    lines.append(table_to_md(feature_audit, ["feature_set", "feature_n"]))
    lines.append("")
    lines.append("## validation 选择后的 test 泛化")
    lines.append("")
    lines.append(
        table_to_md(
            val_test,
            [
                "event_group",
                "method",
                "deployable",
                "val_chosen_feature_set",
                "val_delta_vs_latest_mean",
                "test_delta_vs_latest_mean",
                "test_corr_mean",
                "test_passes_latest",
                "val_and_test_same_direction_gain",
            ],
        )
    )
    lines.append("")
    lines.append("## bad_top10 分层")
    lines.append("")
    lines.append(
        table_to_md(
            bad,
            [
                "feature_set",
                "split",
                "n",
                "latest_rmse_mean",
                "bio_top1_rmse_mean",
                "bio_top1_minus_latest_mean",
                "bio_top3_oracle_rmse_mean",
                "bio_top3_minus_latest_mean",
                "bio_corr_mean",
                "bio_best_in_top3_rate",
            ],
            max_rows=80,
        )
    )
    lines.append("")
    lines.append("## bad_top10 + vehicle_ambiguous 分层")
    lines.append("")
    lines.append(
        table_to_md(
            amb,
            [
                "feature_set",
                "split",
                "n",
                "latest_rmse_mean",
                "bio_top1_rmse_mean",
                "bio_top1_minus_latest_mean",
                "bio_top3_oracle_rmse_mean",
                "bio_top3_minus_latest_mean",
                "bio_corr_mean",
                "bio_best_in_top3_rate",
            ],
            max_rows=80,
        )
    )
    lines.append("")
    lines.append("## train-only 特征类型筛选摘要")
    lines.append("")
    lines.append(table_to_md(screen_summary, max_rows=40))
    lines.append("")
    lines.append("## train-only top20 特征")
    lines.append("")
    lines.append(
        table_to_md(
            screen.head(20),
            [
                "feature",
                "feature_category",
                "signal_family",
                "finite_rate_train",
                "behavior_eta_max",
                "bad_eta_max",
                "identity_eta_max",
                "identity_to_behavior_ratio",
                "behavior_identity_score",
            ],
            max_rows=20,
        )
    )
    lines.append("")
    lines.append("## 关键判读")
    lines.append("")
    route_viable = bool(decision["route_viable_now"].iloc[0]) if len(decision) else False
    if route_viable:
        lines.append("- route gate 通过：底层 200Hz shape-state 生理表示已经具备进入轨迹模型的最低证据。")
    else:
        lines.append("- route gate 未通过：即使回到底层 200Hz 信号形态，当前生理状态仍未形成可部署候选选择收益。")
    lines.append("- deployable 结论只看 validation 选择后的 top1；top3/top5 oracle 只作为上限诊断。")
    lines.append("- 如果本轮仍未通过，继续做更复杂融合模型的收益很低，应考虑更底层信号清洗/事件定义，或转为 subject-aware 个体校准任务。")
    lines.append("")
    lines.append("## 关键图")
    lines.append("")
    for fig in figures:
        lines.append(f"- `{fig.relative_to(OUT)}`")
    lines.append("")
    lines.append("## guardrail")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(guardrail, ensure_ascii=False, indent=2))
    lines.append("```")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    print("[v285] 目的：从 cleaned 200Hz 连续信号重算 shape-state，并验证 route gate。", flush=True)
    clean_out_dir()
    np.random.seed(SEED)

    loaded = V252.load_fixed_inputs()
    manifest = loaded["data"].manifest.copy()
    cand = V284.load_candidate_table()
    context = V284.build_event_context(cand)

    features = build_raw200_shape_features(manifest)
    events, _targets = add_targets_and_labels(features, loaded, context)
    cols = numeric_feature_columns(events)
    screen = feature_screening(events, cols)
    screen_summary = summarize_feature_screen(screen)
    feature_sets = choose_feature_sets(screen)
    if not feature_sets:
        raise RuntimeError("v285 没有可用 feature set，无法 route gate")

    per_event_parts = []
    scaler_parts = []
    feature_audit_parts = []
    for name, fs_cols in feature_sets.items():
        print(f"[v285] evaluate feature_set={name} feature_n={len(fs_cols)}", flush=True)
        per_event, scaler, audit = V284.evaluate_feature_set(name, fs_cols, events, cand, context)
        per_event_parts.append(per_event)
        scaler_parts.append(scaler)
        feature_audit_parts.append(audit)

    per_event_all = pd.concat(per_event_parts, ignore_index=True)
    scaler_all = pd.concat(scaler_parts, ignore_index=True)
    feature_audit = pd.concat(feature_audit_parts, ignore_index=True)
    expanded = V284.expand_groups(per_event_all)
    summary = V284.summarize_groups(expanded)
    val_test = V284.val_chosen_generalization(summary)
    decision = V284.route_gate_decision(summary, val_test)

    write_csv(features, TABLES / "v285_raw200_shape_state_features.csv")
    write_csv(events, TABLES / "v285_raw200_shape_state_features_with_targets.csv")
    write_csv(screen, TABLES / "v285_train_only_feature_screen.csv")
    write_csv(screen_summary, TABLES / "v285_feature_screen_summary.csv")
    write_csv(feature_audit, TABLES / "v285_feature_set_audit.csv")
    write_csv(scaler_all, TABLES / "v285_train_scaler_audit.csv")
    write_csv(per_event_all, TABLES / "v285_route_gate_per_event.csv")
    write_csv(summary, TABLES / "v285_route_group_summary.csv")
    write_csv(val_test, TABLES / "v285_val_chosen_generalization.csv")
    write_csv(decision, TABLES / "v285_route_gate_decision.csv")
    write_input_hashes()

    figures = [plot_val_test_delta(summary), plot_screen(screen_summary), plot_corr(summary)]
    v283_guard = json.loads(V284.V283_GUARDRAIL.read_text(encoding="utf-8")) if V284.V283_GUARDRAIL.exists() else {}
    fixed_latest = summary[
        summary["split"].eq("test")
        & summary["event_group"].eq("bad_top10")
        & summary["feature_set"].eq(feature_audit["feature_set"].iloc[0])
    ]["latest_rmse_mean"]
    guardrail: Dict[str, object] = {
        "pass": True,
        "zip_testzip": False,
        "event_n": int(events["event_uid"].nunique()),
        "candidate_rows": int(len(cand)),
        "raw200_feature_n": int(len(cols)),
        "feature_set_n": int(len(feature_sets)),
        "uses_post_observation_any": bool(events["bio285_uses_post_observation"].astype(bool).any()),
        "ok_rate": float(events["bio285_status"].astype(str).eq("ok").mean()),
        "fixed_wait_latest_badtop10": float(fixed_latest.iloc[0]) if len(fixed_latest) else math.nan,
        "route_viable_now": bool(decision["route_viable_now"].iloc[0]),
        "deployable_top1_badtop10_pass": bool(
            decision.loc[decision["check"].eq("deployable_top1_val_chosen_bad_top10"), "pass"].iloc[0]
        ),
        "deployable_top1_bad_ambiguous_pass": bool(
            decision.loc[decision["check"].eq("deployable_top1_val_chosen_bad_ambiguous"), "pass"].iloc[0]
        ),
        "test_best_top1_diagnostic_pass": bool(
            decision.loc[decision["check"].eq("test_best_top1_diagnostic_beats_latest"), "pass"].iloc[0]
        ),
        "reused_v260_feature_table": False,
        "test_used_for_feature_selection": False,
        "v283_old_route_closed": bool(v283_guard.get("old_feature_selector_route_closed", False)),
    }
    guardrail["pass"] = bool(
        guardrail["event_n"] > 0
        and guardrail["candidate_rows"] > 0
        and guardrail["raw200_feature_n"] >= 50
        and guardrail["feature_set_n"] >= 3
        and not guardrail["uses_post_observation_any"]
        and not guardrail["reused_v260_feature_table"]
        and not guardrail["test_used_for_feature_selection"]
    )
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    report = write_report(feature_audit, screen, screen_summary, summary, val_test, decision, guardrail, figures)
    write_file_inventory()
    first_zip_ok = make_zip()
    guardrail["zip_testzip"] = bool(first_zip_ok)
    guardrail["pass"] = bool(guardrail["pass"] and first_zip_ok)
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    report = write_report(feature_audit, screen, screen_summary, summary, val_test, decision, guardrail, figures)
    write_file_inventory()
    second_zip_ok = make_zip()
    guardrail["zip_testzip"] = bool(second_zip_ok)
    guardrail["pass"] = bool(guardrail["pass"] and second_zip_ok)
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    write_file_inventory()
    if not bool(guardrail["pass"]):
        raise AssertionError("v285 guardrail 未通过：\n" + json.dumps(guardrail, ensure_ascii=False, indent=2))

    print(f"[v285] report={report}", flush=True)
    print(f"[v285] zip={ZIP_PATH}", flush=True)
    print(json.dumps(guardrail, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
