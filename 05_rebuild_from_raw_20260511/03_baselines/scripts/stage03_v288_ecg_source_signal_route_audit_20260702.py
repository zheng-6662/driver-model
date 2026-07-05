#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v288 ECG source-signal route audit.

本轮目标：
- v287 只在 v285 已抽好的 shape-state 特征里拆窗口，发现 ECG 最近 1-2 秒有弱苗头，
  但没有形成可部署 top1 改善；
- v288 不继续换 fusion / gate / residual 模型，而是回到 cleaned 200Hz ECG 源信号，
  单独重建 ECG 形态、R 峰/RR、质量和因果同步偏移特征；
- 仍然先过 v284/v285 同一个 vehicle top40 candidate route gate。只有 route gate 通过，
  才说明 ECG 源信号值得进入更复杂轨迹预测模型。

边界：
- 只使用 observation_s 之前的 ECG 数据；
- 当前实验选择 ECG 这个方向来自 v287 的弱诊断苗头，但 v288 内部 feature set 和
  validation 选择不使用 v288 test 结果；
- 不读取 v260/v284 派生生理特征表作为输入；
- 不训练轨迹融合模型，只验证 ECG 源信号是否能稳定帮助候选选择。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"F:\data_set_process\data_process")
REBUILD = ROOT / "05_rebuild_from_raw_20260511"
BASELINES = REBUILD / "03_baselines"
SCRIPTS = BASELINES / "scripts"

V284_SCRIPT = SCRIPTS / "stage03_v284_dynamic_low_identity_physio_route_gate_20260702.py"
V285_SCRIPT = SCRIPTS / "stage03_v285_raw200_shape_state_route_gate_20260702.py"
V287_GUARDRAIL = BASELINES / "v287_physio_temporal_window_route_audit_20260702" / "logs" / "guardrail_check.json"
V287_WINNERS = (
    BASELINES
    / "v287_physio_temporal_window_route_audit_20260702"
    / "tables"
    / "v287_group_winner_summary.csv"
)

OUT = BASELINES / "v288_ecg_source_signal_route_audit_20260702"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
ZIP_PATH = BASELINES / "v288_ecg_source_signal_route_audit_20260702_pack.zip"

SEED = 28802
MIN_FEATURES = 8
MAX_FEATURES_PER_SET = 48

# ECG 基线只取 observation 前 60 到 20 秒，避免用到接近事件的响应片段。
BASELINE_WINDOW = (-60.0, -20.0)

# 每个窗口全部在 observation_s 之前。endm0p5/endm1/endm2 用来检查轻微同步偏移或生理提前量。
WINDOW_SPECS: Dict[str, Tuple[float, float]] = {
    "dur1_end0": (-1.0, 0.0),
    "dur2_end0": (-2.0, 0.0),
    "dur3_end0": (-3.0, 0.0),
    "dur5_end0": (-5.0, 0.0),
    "dur1_endm0p5": (-1.5, -0.5),
    "dur2_endm0p5": (-2.5, -0.5),
    "dur3_endm0p5": (-3.5, -0.5),
    "dur1_endm1": (-2.0, -1.0),
    "dur2_endm1": (-3.0, -1.0),
    "dur3_endm1": (-4.0, -1.0),
    "dur2_endm2": (-4.0, -2.0),
    "dur5_endm2": (-7.0, -2.0),
    "pre10_pre5": (-10.0, -5.0),
    "pre20_pre10": (-20.0, -10.0),
}

DELTA_PAIRS = [
    ("dur1_end0", "dur1_endm1"),
    ("dur2_end0", "dur2_endm1"),
    ("dur2_end0", "dur2_endm2"),
    ("dur3_end0", "dur3_endm1"),
    ("dur5_end0", "dur5_endm2"),
    ("dur5_end0", "pre10_pre5"),
]

DELTA_METRICS = [
    "z_mean",
    "z_std",
    "z_abs_mean",
    "z_abs_p95",
    "dz_abs_mean",
    "dz_abs_p95",
    "noise_ratio",
    "line_length_per_s",
    "peak_rate_per_s",
    "peak_score_median",
    "bpm",
    "rr_mean",
    "rr_std",
    "rr_rmssd",
    "rr_cv",
    "last_peak_age_to_obs",
]

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

matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False


def import_module_from_path(module_name: str, path: Path):
    """按路径导入前序脚本，只复用已验证的数据读取和 route gate 工具。"""

    if not path.exists():
        raise FileNotFoundError(f"缺少脚本：{path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法导入脚本：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V284 = import_module_from_path("stage03_v284_for_v288", V284_SCRIPT)
V285 = import_module_from_path("stage03_v285_for_v288", V285_SCRIPT)


def ensure_dirs() -> None:
    """创建 v288 输出目录。"""

    for folder in (TABLES, FIGURES, REPORTS, LOGS):
        folder.mkdir(parents=True, exist_ok=True)


def clean_out_dir() -> None:
    """只清理 v288 自己的输出，避免影响前序版本。"""

    if OUT.exists():
        shutil.rmtree(OUT)
    ensure_dirs()
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """统一写 utf-8-sig CSV，方便 Excel 直接打开中文列名和内容。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def file_sha256(path: Path) -> str:
    """记录输入文件哈希，保证后续能追溯本轮实验依赖。"""

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finite(values: Iterable[object]) -> np.ndarray:
    """提取有限浮点值。"""

    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def robust_center_scale(values: Iterable[object]) -> Tuple[float, float]:
    """用 median + IQR/MAD/std 构造稳健基线。"""

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
    """按事件自身 observation 前 baseline 做 ECG z-score，减少 subject/recording 绝对幅值混淆。"""

    center, scale = robust_center_scale(baseline_values)
    arr = np.asarray(values, dtype=float)
    if not np.isfinite(center) or not np.isfinite(scale) or scale <= 1e-9:
        return np.full(arr.shape, np.nan, dtype=float)
    z = (arr - center) / scale
    z[~np.isfinite(z)] = np.nan
    return z


def nan_quantile(values: np.ndarray, q: float) -> float:
    vals = finite(values)
    if vals.size == 0:
        return math.nan
    return float(np.quantile(vals, q))


def nan_mean(values: np.ndarray) -> float:
    vals = finite(values)
    if vals.size == 0:
        return math.nan
    return float(np.mean(vals))


def nan_std(values: np.ndarray) -> float:
    vals = finite(values)
    if vals.size == 0:
        return math.nan
    return float(np.std(vals))


def safe_div(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or abs(den) < 1e-12:
        return math.nan
    return float(num / den)


def slope(times: np.ndarray, vals: np.ndarray) -> float:
    """首尾斜率，比小窗线性回归更稳，也更不容易过拟合噪声。"""

    mask = np.isfinite(times) & np.isfinite(vals)
    if int(mask.sum()) < 2:
        return math.nan
    t = times[mask]
    v = vals[mask]
    dt = float(t[-1] - t[0])
    if abs(dt) < 1e-9:
        return math.nan
    return float((v[-1] - v[0]) / dt)


def choose_ecg_source(arrays: Dict[str, np.ndarray], b_left: int, b_right: int) -> Tuple[str, np.ndarray]:
    """在 ECG_filt200 / ECG_raw200 之间选一个基线可用且方差不是 0 的通道。"""

    best_col = "ECG_filt200" if "ECG_filt200" in arrays else "ECG_raw200"
    best_score = -math.inf
    best_baseline = np.array([], dtype=float)
    for col in ["ECG_filt200", "ECG_raw200"]:
        if col not in arrays:
            continue
        baseline = np.asarray(arrays[col][b_left:b_right], dtype=float)
        vals = finite(baseline)
        finite_rate = float(np.isfinite(baseline).mean()) if len(baseline) else 0.0
        std = float(np.std(vals)) if vals.size else 0.0
        score = finite_rate + min(1.0, std / 1e-3)
        if score > best_score:
            best_col = col
            best_score = score
            best_baseline = baseline
    return best_col, best_baseline


def detect_local_peaks(score: np.ndarray, sample_hz: float, threshold: float) -> np.ndarray:
    """用局部极大值 + refractory period 提取 ECG 候选 R 峰，避免 scipy 依赖。"""

    s = np.asarray(score, dtype=float)
    if s.size < 3:
        return np.array([], dtype=int)
    finite_mask = np.isfinite(s)
    s2 = np.where(finite_mask, s, -np.inf)
    candidate = np.r_[False, (s2[1:-1] > s2[:-2]) & (s2[1:-1] >= s2[2:]) & (s2[1:-1] > threshold), False]
    idx = np.flatnonzero(candidate)
    if idx.size == 0:
        return np.array([], dtype=int)

    min_dist = max(1, int(round(0.28 * max(sample_hz, 1.0))))
    selected: List[int] = []
    for i in idx[np.argsort(s2[idx])[::-1]]:
        if all(abs(int(i) - int(j)) >= min_dist for j in selected):
            selected.append(int(i))
    selected = sorted(selected)
    return np.asarray(selected, dtype=int)


def peak_quality_score(peak_times: np.ndarray, duration_s: float) -> float:
    """给极性选择使用的简单质量分数：峰数适中且 RR 多数落在合理心率范围。"""

    if duration_s <= 0:
        return -math.inf
    n = int(len(peak_times))
    if n == 0:
        return -10.0
    rr = np.diff(peak_times)
    plausible = (rr >= 0.32) & (rr <= 1.50)
    plausible_rate = float(plausible.mean()) if rr.size else 0.0
    density = n / duration_s
    density_penalty = abs(density - 1.25)
    return float(n + 2.0 * plausible_rate - density_penalty)


def choose_ecg_polarity(z_context: np.ndarray, t_context: np.ndarray, sample_hz: float) -> int:
    """ECG 极性可能因设备或预处理变化而反向，因此在事件内因果上下文里自适应选择正/负 R 峰。"""

    vals = finite(z_context)
    if vals.size == 0:
        return 1
    threshold = max(0.8, float(np.quantile(np.abs(vals), 0.80)))
    best_polarity = 1
    best_score = -math.inf
    duration = float(t_context[-1] - t_context[0]) if len(t_context) >= 2 else 0.0
    for polarity in [1, -1]:
        peaks = detect_local_peaks(polarity * z_context, sample_hz, threshold)
        score = peak_quality_score(t_context[peaks] if len(peaks) else np.array([], dtype=float), duration)
        if score > best_score:
            best_polarity = polarity
            best_score = score
    return int(best_polarity)


def morphology_features(times: np.ndarray, raw: np.ndarray, z: np.ndarray, prefix: str, sample_hz: float) -> Dict[str, float]:
    """ECG 短窗形态和噪声特征，主要服务于“事件前状态是否变化”。"""

    out: Dict[str, float] = {}
    duration = float(times[-1] - times[0]) if len(times) >= 2 else 0.0
    zvals = np.asarray(z, dtype=float)
    raw_vals = np.asarray(raw, dtype=float)
    finite_z = finite(zvals)
    out[f"{prefix}_rows"] = int(len(zvals))
    out[f"{prefix}_duration_s"] = duration
    out[f"{prefix}_valid_ratio"] = float(np.isfinite(zvals).mean()) if len(zvals) else 0.0
    out[f"{prefix}_raw_valid_ratio"] = float(np.isfinite(raw_vals).mean()) if len(raw_vals) else 0.0
    if finite_z.size == 0:
        for metric in [
            "z_mean",
            "z_std",
            "z_min",
            "z_max",
            "z_range",
            "z_abs_mean",
            "z_abs_p95",
            "z_slope",
            "z_last_minus_first",
            "dz_abs_mean",
            "dz_abs_p95",
            "noise_ratio",
            "line_length_per_s",
            "flat_step_rate",
            "outlier_rate",
        ]:
            out[f"{prefix}_{metric}"] = math.nan
        return out

    out[f"{prefix}_z_mean"] = float(np.mean(finite_z))
    out[f"{prefix}_z_std"] = float(np.std(finite_z))
    out[f"{prefix}_z_min"] = float(np.min(finite_z))
    out[f"{prefix}_z_max"] = float(np.max(finite_z))
    out[f"{prefix}_z_range"] = float(np.max(finite_z) - np.min(finite_z))
    out[f"{prefix}_z_abs_mean"] = float(np.mean(np.abs(finite_z)))
    out[f"{prefix}_z_abs_p95"] = nan_quantile(np.abs(zvals), 0.95)
    out[f"{prefix}_z_slope"] = slope(times, zvals)
    valid_idx = np.flatnonzero(np.isfinite(zvals))
    out[f"{prefix}_z_last_minus_first"] = (
        float(zvals[valid_idx[-1]] - zvals[valid_idx[0]]) if len(valid_idx) >= 2 else math.nan
    )

    dz = np.diff(zvals)
    dz = dz[np.isfinite(dz)]
    out[f"{prefix}_dz_abs_mean"] = float(np.mean(np.abs(dz))) if dz.size else math.nan
    out[f"{prefix}_dz_abs_p95"] = float(np.quantile(np.abs(dz), 0.95)) if dz.size else math.nan
    out[f"{prefix}_noise_ratio"] = safe_div(float(np.std(dz)) if dz.size else math.nan, float(np.std(finite_z)))
    out[f"{prefix}_line_length_per_s"] = safe_div(float(np.sum(np.abs(dz))) if dz.size else math.nan, duration)
    out[f"{prefix}_flat_step_rate"] = float(np.mean(np.abs(dz) < 1e-6)) if dz.size else math.nan
    out[f"{prefix}_outlier_rate"] = float(np.mean(np.abs(finite_z) > 8.0)) if finite_z.size else math.nan
    return out


def peak_rr_features(
    times: np.ndarray,
    z: np.ndarray,
    prefix: str,
    sample_hz: float,
    polarity: int,
    observation_s: float,
) -> Dict[str, float]:
    """从 ECG z-score 中提取 R 峰和 RR 变化，全部只使用窗口内历史点。"""

    out: Dict[str, float] = {}
    if len(times) < 3:
        for metric in [
            "peak_n",
            "peak_rate_per_s",
            "peak_score_median",
            "peak_score_p90",
            "bpm",
            "rr_mean",
            "rr_std",
            "rr_rmssd",
            "rr_cv",
            "rr_slope",
            "rr_plausible_rate",
            "last_rr",
            "last_peak_age_to_obs",
        ]:
            out[f"{prefix}_{metric}"] = math.nan
        return out

    vals = finite(z)
    if vals.size == 0:
        threshold = math.nan
    else:
        threshold = max(0.8, float(np.quantile(np.abs(vals), 0.80)))
    score = polarity * np.asarray(z, dtype=float)
    peak_idx = detect_local_peaks(score, sample_hz, threshold if np.isfinite(threshold) else 0.8)
    peak_times = np.asarray(times[peak_idx], dtype=float) if len(peak_idx) else np.array([], dtype=float)
    peak_scores = np.asarray(score[peak_idx], dtype=float) if len(peak_idx) else np.array([], dtype=float)
    duration = float(times[-1] - times[0]) if len(times) >= 2 else math.nan
    rr = np.diff(peak_times)
    plausible = (rr >= 0.32) & (rr <= 1.50) if rr.size else np.array([], dtype=bool)
    rr_good = rr[plausible] if rr.size else np.array([], dtype=float)

    out[f"{prefix}_peak_n"] = int(len(peak_times))
    out[f"{prefix}_peak_rate_per_s"] = safe_div(float(len(peak_times)), duration)
    out[f"{prefix}_peak_score_median"] = nan_mean(peak_scores) if len(peak_scores) else math.nan
    out[f"{prefix}_peak_score_p90"] = nan_quantile(peak_scores, 0.90) if len(peak_scores) else math.nan
    out[f"{prefix}_bpm"] = 60.0 / float(np.mean(rr_good)) if rr_good.size else math.nan
    out[f"{prefix}_rr_mean"] = float(np.mean(rr_good)) if rr_good.size else math.nan
    out[f"{prefix}_rr_std"] = float(np.std(rr_good)) if rr_good.size else math.nan
    out[f"{prefix}_rr_rmssd"] = float(np.sqrt(np.mean(np.diff(rr_good) ** 2))) if rr_good.size >= 3 else math.nan
    out[f"{prefix}_rr_cv"] = safe_div(float(np.std(rr_good)) if rr_good.size else math.nan, float(np.mean(rr_good)) if rr_good.size else math.nan)
    out[f"{prefix}_rr_slope"] = slope(np.arange(len(rr_good), dtype=float), rr_good) if rr_good.size >= 2 else math.nan
    out[f"{prefix}_rr_plausible_rate"] = float(plausible.mean()) if rr.size else math.nan
    out[f"{prefix}_last_rr"] = float(rr_good[-1]) if rr_good.size else math.nan
    out[f"{prefix}_last_peak_age_to_obs"] = float(observation_s - peak_times[-1]) if len(peak_times) else math.nan
    return out


def parse_window_end_group(window_name: str) -> str:
    if "_end0" in window_name:
        return "end0"
    if "_endm0p5" in window_name:
        return "endm0p5"
    if "_endm1" in window_name:
        return "endm1"
    if "_endm2" in window_name:
        return "endm2"
    if window_name == "pre10_pre5":
        return "endm5"
    if window_name == "pre20_pre10":
        return "endm10"
    return "other"


def parse_window_duration_group(window_name: str) -> str:
    m = re.match(r"dur(\d+)_", window_name)
    if m:
        return f"dur{m.group(1)}"
    return window_name


def extract_recording_ecg_features(recording_df: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    """对同一 recording 内所有 delay=0 事件提取 ECG 源信号特征。"""

    times = pd.to_numeric(recording_df["t_s"], errors="coerce").to_numpy(dtype=float)
    sample_hz = 200.0
    if len(times) >= 5:
        dt = np.diff(times)
        good_dt = dt[np.isfinite(dt) & (dt > 0)]
        if good_dt.size:
            sample_hz = float(1.0 / np.median(good_dt))

    arrays: Dict[str, np.ndarray] = {}
    for col in ["ECG_filt200", "ECG_raw200"]:
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
            "bio288_status": "ok",
            "bio288_sample_hz": sample_hz,
            "bio288_uses_post_observation": False,
        }

        b_start = max(0.0, obs + BASELINE_WINDOW[0])
        b_end = max(0.0, obs + BASELINE_WINDOW[1])
        b_left = int(np.searchsorted(times, b_start, side="left"))
        b_right = int(np.searchsorted(times, b_end, side="right"))
        chosen_col, baseline = choose_ecg_source(arrays, b_left, b_right)
        full = np.asarray(arrays.get(chosen_col, np.full(len(times), np.nan)), dtype=float)
        out["bio288_ecg_chosen_col_code"] = 0 if chosen_col == "ECG_filt200" else 1
        out["bio288_baseline_rows"] = int(max(0, b_right - b_left))
        out["bio288_baseline_valid_ratio"] = float(np.isfinite(baseline).mean()) if len(baseline) else 0.0
        out["bio288_baseline_std"] = nan_std(baseline)

        # 用 observation 前 20 秒上下文选择 ECG 极性，仍然不越过 observation。
        c_left = int(np.searchsorted(times, max(0.0, obs - 20.0), side="left"))
        c_right = int(np.searchsorted(times, obs, side="right"))
        z_context = robust_z(full[c_left:c_right], baseline)
        t_context = times[c_left:c_right]
        polarity = choose_ecg_polarity(z_context, t_context, sample_hz)
        out["bio288_ecg_polarity_code"] = float(polarity)

        for win_name, (offset_start, offset_end) in WINDOW_SPECS.items():
            if offset_end > 1e-9:
                out["bio288_uses_post_observation"] = True
            start = max(0.0, obs + offset_start)
            end = max(0.0, obs + offset_end)
            left = int(np.searchsorted(times, start, side="left"))
            right = int(np.searchsorted(times, end, side="right"))
            win_t = times[left:right]
            raw = full[left:right]
            z = robust_z(raw, baseline)
            prefix = f"bio288_w_{win_name}_ecg"
            out.update(morphology_features(win_t, raw, z, prefix, sample_hz))
            out.update(peak_rr_features(win_t, z, prefix, sample_hz, polarity, obs))

        for recent, ref in DELTA_PAIRS:
            for metric in DELTA_METRICS:
                a = out.get(f"bio288_w_{recent}_ecg_{metric}", math.nan)
                b = out.get(f"bio288_w_{ref}_ecg_{metric}", math.nan)
                out[f"bio288_delta_{recent}_minus_{ref}_ecg_{metric}"] = (
                    float(a - b) if np.isfinite(a) and np.isfinite(b) else math.nan
                )
        rows.append(out)
    return pd.DataFrame(rows)


def build_ecg_source_features(manifest: pd.DataFrame) -> pd.DataFrame:
    """从 cleaned 200Hz ECG 连续层重建 delay=0 事件级特征。"""

    inventory = V285.load_physio_inventory()
    samples = manifest[manifest["delay_ms"].eq(0)][["event_uid", "subject", "recording", "split", "delay_ms", "observation_s"]].copy()
    samples = samples.reset_index(names="row_index")
    samples["session_stamp"] = samples["recording"].map(V285.session_stamp_from_recording)

    parts: List[pd.DataFrame] = []
    missing: List[Dict[str, object]] = []
    grouped = samples.groupby(["subject", "session_stamp"], sort=False)
    for group_i, ((subject, session), g) in enumerate(grouped, start=1):
        path = inventory.get((str(subject), str(session)))
        if path is None or not path.exists():
            print(f"[v288] missing 200Hz physio {group_i}/{len(grouped)} subject={subject} session={session}", flush=True)
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
                        "bio288_status": "missing_recording",
                        "bio288_uses_post_observation": False,
                    }
                )
            continue
        print(f"[v288] extract ECG source {group_i}/{len(grouped)} subject={subject} session={session} events={len(g)}", flush=True)
        rec = V285.read_physio_recording(path)
        parts.append(extract_recording_ecg_features(rec, g))
    if missing:
        parts.append(pd.DataFrame(missing))
    out = pd.concat(parts, ignore_index=True).sort_values("row_index").reset_index(drop=True)
    return out


def eta_squared(feature: np.ndarray, labels: Iterable[object]) -> float:
    """train-only 单特征 eta²，用于行为相关性和 subject/recording 身份惩罚。"""

    x = np.asarray(feature, dtype=float)
    y = pd.Series(labels).astype(str).to_numpy()
    mask = np.isfinite(x) & pd.notna(y)
    x = x[mask]
    y = y[mask]
    if x.size < 20 or np.nanstd(x) < 1e-12:
        return math.nan
    grand = float(np.mean(x))
    total = float(np.sum((x - grand) ** 2))
    if total <= 1e-12:
        return math.nan
    between = 0.0
    for label in pd.unique(y):
        sub = x[y == label]
        if sub.size:
            between += float(sub.size) * float((np.mean(sub) - grand) ** 2)
    return float(max(0.0, min(1.0, between / total)))


def finite_rate(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return 0.0
    return float(np.isfinite(x).mean())


def infer_window_group(feature: str) -> str:
    name = str(feature)
    for win in WINDOW_SPECS:
        if f"bio288_w_{win}_ecg_" in name:
            return win
    if name.startswith("bio288_delta_"):
        return "delta"
    return "global"


def infer_offset_group(feature: str) -> str:
    win = infer_window_group(feature)
    if win in WINDOW_SPECS:
        return parse_window_end_group(win)
    if win == "delta":
        return "delta"
    return "global"


def infer_duration_group(feature: str) -> str:
    win = infer_window_group(feature)
    if win in WINDOW_SPECS:
        return parse_window_duration_group(win)
    if win == "delta":
        return "delta"
    return "global"


def feature_category(col: str) -> str:
    low = col.lower()
    if low.startswith("bio288_delta_"):
        return "temporal_delta"
    if any(k in low for k in ["rr_", "bpm", "peak_n", "peak_rate", "last_peak"]):
        return "rr_peak"
    if any(k in low for k in ["valid_ratio", "baseline", "flat", "outlier", "noise_ratio", "chosen_col", "polarity"]):
        return "quality"
    if any(k in low for k in ["dz_", "line_length", "z_slope", "last_minus", "z_abs", "z_range"]):
        return "morph_dynamic"
    return "morph_level"


def numeric_feature_columns(events: pd.DataFrame) -> List[str]:
    """选择可以进入 route gate 的 ECG 数值特征，排除行数、持续时间和显式元数据。"""

    excluded = [
        "_rows",
        "_duration_s",
        "sample_hz",
        "uses_post_observation",
        "baseline_rows",
        "ecg_chosen_col_code",
        "ecg_polarity_code",
    ]
    cols: List[str] = []
    for col in events.columns:
        if not col.startswith("bio288_"):
            continue
        if any(s in col for s in excluded):
            continue
        if pd.api.types.is_numeric_dtype(events[col]):
            cols.append(col)
    return cols


def feature_screening(events: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """只在 train split 上筛选 ECG 特征，test 仅用于最终报告。"""

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
        identity_values = [v for v in [identity_subject, identity_recording] if np.isfinite(v)]
        identity_max = max(identity_values) if identity_values else 0.0
        behavior_values = [v for v in behavior_scores.values() if np.isfinite(v)]
        behavior_max = max(behavior_values) if behavior_values else 0.0
        bad_score = max(
            [
                behavior_scores.get("bad_top10_v250_diagnostic", 0.0),
                behavior_scores.get("bad_top10", 0.0),
                behavior_scores.get("bad_top10_vehicle_ambiguous", 0.0),
            ]
        )
        rows.append(
            {
                "feature": col,
                "feature_category": feature_category(col),
                "signal_family": "ecg",
                "window_group": infer_window_group(col),
                "offset_group": infer_offset_group(col),
                "duration_group": infer_duration_group(col),
                "finite_rate_train": finite_rate(x),
                "behavior_eta_max": float(behavior_max),
                "bad_eta_max": float(bad_score),
                "identity_eta_subject": float(identity_subject) if np.isfinite(identity_subject) else math.nan,
                "identity_eta_recording": float(identity_recording) if np.isfinite(identity_recording) else math.nan,
                "identity_eta_max": float(identity_max),
                "identity_to_behavior_ratio": float(identity_max / max(behavior_max, 1e-6)),
                "behavior_identity_score": float(behavior_max / (identity_max + 0.01)),
                "bad_identity_score": float(bad_score / (identity_max + 0.01)),
                **{f"eta_{k}": float(v) if np.isfinite(v) else math.nan for k, v in behavior_scores.items()},
            }
        )
    return pd.DataFrame(rows).sort_values(["behavior_identity_score", "behavior_eta_max"], ascending=False).reset_index(drop=True)


def top_features(df: pd.DataFrame, n: int = MAX_FEATURES_PER_SET) -> List[str]:
    if df.empty:
        return []
    return (
        df.sort_values(["rank_score", "behavior_identity_score", "bad_identity_score"], ascending=False)["feature"]
        .drop_duplicates()
        .head(n)
        .astype(str)
        .tolist()
    )


def build_feature_sets(screen: pd.DataFrame) -> Tuple[Dict[str, List[str]], pd.DataFrame]:
    """构造 ECG 专用 feature set；组内排序仍只来自 train split。"""

    usable = screen[screen["finite_rate_train"].ge(0.70)].copy()
    usable["rank_score"] = (
        usable["behavior_identity_score"].fillna(0.0)
        + 0.7 * usable["bad_identity_score"].fillna(0.0)
        + 0.1 * usable["behavior_eta_max"].fillna(0.0)
    )

    sets: Dict[str, List[str]] = {}
    audit_rows: List[Dict[str, object]] = []

    def add_set(name: str, group_type: str, group_value: str, df: pd.DataFrame, n: int = MAX_FEATURES_PER_SET) -> None:
        cols = top_features(df, n=n)
        if len(cols) < MIN_FEATURES:
            return
        sets[name] = cols
        audit_rows.append(
            {
                "feature_set": name,
                "group_type": group_type,
                "group_value": group_value,
                "candidate_feature_n": int(len(df)),
                "feature_n": int(len(cols)),
                "rank_score_max": float(df["rank_score"].max()),
                "behavior_eta_max": float(df["behavior_eta_max"].max()),
                "bad_eta_max": float(df["bad_eta_max"].max()),
                "identity_eta_median": float(df["identity_eta_max"].median()),
            }
        )

    add_set("ecg_all_top64", "all", "all", usable, n=64)
    low_identity = usable[usable["identity_eta_max"].le(0.10)].copy()
    if len(low_identity) < 32:
        low_identity = usable.sort_values("identity_eta_max", ascending=True).head(max(32, min(96, len(usable))))
    add_set("ecg_low_identity_top48", "identity", "low_identity", low_identity, n=48)

    for cat in ["rr_peak", "morph_dynamic", "morph_level", "quality", "temporal_delta"]:
        add_set(f"ecg_category_{cat}_top48", "category", cat, usable[usable["feature_category"].eq(cat)], n=48)

    for offset in ["end0", "endm0p5", "endm1", "endm2", "endm5", "endm10", "delta"]:
        add_set(f"ecg_offset_{offset}_top32", "offset", offset, usable[usable["offset_group"].eq(offset)], n=32)

    for dur in ["dur1", "dur2", "dur3", "dur5", "pre10_pre5", "pre20_pre10", "delta"]:
        add_set(f"ecg_duration_{dur}_top32", "duration", dur, usable[usable["duration_group"].eq(dur)], n=32)

    # 对 v287 暴露出的最近窗口苗头做预注册复核：只按 train screen 排序，不使用 v288 test。
    for win in ["dur1_end0", "dur2_end0", "dur1_endm0p5", "dur2_endm0p5", "dur1_endm1", "dur2_endm1"]:
        add_set(f"ecg_window_{win}_top24", "window", win, usable[usable["window_group"].eq(win)], n=24)

    return sets, pd.DataFrame(audit_rows)


def summarize_feature_screen(screen: pd.DataFrame) -> pd.DataFrame:
    if screen.empty:
        return pd.DataFrame()
    return (
        screen.groupby(["feature_category", "offset_group", "duration_group"], as_index=False)
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


def summarize_ecg_quality(features: pd.DataFrame) -> pd.DataFrame:
    """按 recording 汇总 ECG 源信号可用性和最近 2 秒 R 峰检测质量。"""

    cols = [
        "bio288_baseline_valid_ratio",
        "bio288_baseline_std",
        "bio288_w_dur2_end0_ecg_valid_ratio",
        "bio288_w_dur2_end0_ecg_peak_n",
        "bio288_w_dur2_end0_ecg_peak_rate_per_s",
        "bio288_w_dur2_end0_ecg_rr_plausible_rate",
        "bio288_w_dur2_end0_ecg_noise_ratio",
    ]
    use_cols = [c for c in cols if c in features.columns]
    return (
        features.groupby(["subject", "recording", "split"], as_index=False)
        .agg(
            event_n=("event_uid", "nunique"),
            ok_rate=("bio288_status", lambda s: float(pd.Series(s).astype(str).eq("ok").mean())),
            **{f"{c}_median": (c, "median") for c in use_cols},
        )
        .sort_values(["split", "subject", "recording"])
    )


def table_to_md(df: pd.DataFrame, cols: List[str] | None = None, max_rows: int = 80) -> str:
    if df is None or df.empty:
        return "_空表_"
    show = df.copy()
    if cols is not None:
        show = show[[c for c in cols if c in show.columns]]
    return show.head(max_rows).to_markdown(index=False)


def plot_badtop10_delta(summary: pd.DataFrame) -> Path:
    path = FIGURES / "v288_badtop10_val_test_delta.png"
    data = summary[summary["event_group"].eq("bad_top10") & summary["split"].isin(["val", "test"])].copy()
    if data.empty:
        return path
    test_order = (
        data[data["split"].eq("test")]
        .sort_values("bio_top1_minus_latest_mean")
        .head(20)["feature_set"]
        .astype(str)
        .tolist()
    )
    x = np.arange(len(test_order))
    width = 0.35
    fig, ax = plt.subplots(figsize=(13, 5))
    for i, split in enumerate(["val", "test"]):
        vals = []
        for fs in test_order:
            sub = data[data["feature_set"].astype(str).eq(fs) & data["split"].eq(split)]
            vals.append(float(sub["bio_top1_minus_latest_mean"].iloc[0]) if not sub.empty else np.nan)
        ax.bar(x + (i - 0.5) * width, vals, width=width, label=f"{split} top1")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in test_order], fontsize=8)
    ax.set_ylabel("RMSE delta vs latest, lower is better")
    ax.set_title("v288 ECG source route gate: bad_top10 top1")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_offset_summary(summary: pd.DataFrame, feature_audit: pd.DataFrame) -> Path:
    path = FIGURES / "v288_ecg_offset_group_summary.png"
    data = summary[summary["split"].eq("test") & summary["event_group"].eq("bad_top10_vehicle_ambiguous")].copy()
    data = data.merge(feature_audit[["feature_set", "group_type", "group_value"]], on="feature_set", how="left")
    data = data[data["group_type"].eq("offset")].copy()
    if data.empty:
        return path
    data = data.sort_values("bio_corr_mean", ascending=False)
    x = np.arange(len(data))
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(x - 0.18, data["bio_corr_mean"], width=0.36, label="rank corr", color="tab:blue")
    ax1.axhline(0.05, color="tab:blue", linestyle="--", linewidth=1)
    ax1.set_ylabel("test bad ambiguous rank corr")
    ax2 = ax1.twinx()
    ax2.bar(x + 0.18, data["bio_top1_minus_latest_mean"], width=0.36, label="top1 delta", color="tab:orange")
    ax2.axhline(0, color="tab:orange", linestyle="--", linewidth=1)
    ax2.set_ylabel("top1 delta vs latest")
    ax1.set_xticks(x)
    ax1.set_xticklabels(data["group_value"].astype(str), rotation=30, ha="right")
    ax1.set_title("v288 ECG causal offset groups on bad_top10_vehicle_ambiguous")
    ax1.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_feature_screen(screen_summary: pd.DataFrame) -> Path:
    path = FIGURES / "v288_ecg_feature_screen_summary.png"
    if screen_summary.empty:
        return path
    data = screen_summary.sort_values("behavior_identity_score_max", ascending=False).head(24)
    labels = [
        f"{r.feature_category}\n{r.offset_group}/{r.duration_group}"
        for r in data.itertuples(index=False)
    ]
    x = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x, data["behavior_identity_score_max"], label="behavior/identity score")
    ax.plot(x, data["identity_eta_median"], color="tab:red", marker="o", linewidth=1, label="identity eta median")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("train-only score")
    ax.set_title("v288 ECG source feature screen")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_input_hashes() -> None:
    rows = []
    paths = [
        ("v284_script", V284_SCRIPT),
        ("v285_script", V285_SCRIPT),
        ("physio_inventory", V285.PHYSIO_INVENTORY),
        ("v278_candidates", V284.V278_CANDIDATES),
        ("v287_guardrail", V287_GUARDRAIL),
        ("v287_winners", V287_WINNERS),
    ]
    for name, path in paths:
        path = Path(path)
        rows.append(
            {
                "name": name,
                "path": str(path),
                "exists": bool(path.exists()),
                "sha256": file_sha256(path) if path.exists() and path.is_file() else "",
            }
        )
    write_csv(pd.DataFrame(rows), LOGS / "input_hashes.csv")


def write_file_inventory() -> None:
    rows = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "relative_path": str(path.relative_to(OUT)),
                    "bytes": int(path.stat().st_size),
                    "sha256": file_sha256(path),
                }
            )
    write_csv(pd.DataFrame(rows), LOGS / "file_inventory.csv")


def make_zip() -> bool:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(OUT.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=str(path.relative_to(OUT.parent)))
        zf.write(Path(__file__), arcname=f"scripts/{Path(__file__).name}")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        return zf.testzip() is None


def write_report(
    feature_audit: pd.DataFrame,
    screen: pd.DataFrame,
    screen_summary: pd.DataFrame,
    quality: pd.DataFrame,
    summary: pd.DataFrame,
    val_test: pd.DataFrame,
    decision: pd.DataFrame,
    guardrail: Dict[str, object],
    figures: List[Path],
) -> Path:
    path = REPORTS / "v288_ecg_source_signal_route_audit_cn.md"
    bad = summary[summary["event_group"].eq("bad_top10") & summary["split"].isin(["val", "test"])].sort_values(
        ["split", "bio_top1_minus_latest_mean"]
    )
    amb = summary[
        summary["event_group"].eq("bad_top10_vehicle_ambiguous") & summary["split"].isin(["val", "test"])
    ].sort_values(["split", "bio_top1_minus_latest_mean"])
    offset = (
        summary[summary["split"].eq("test") & summary["event_group"].eq("bad_top10_vehicle_ambiguous")]
        .merge(feature_audit[["feature_set", "group_type", "group_value"]], on="feature_set", how="left")
        .query("group_type == 'offset'")
        .sort_values("bio_corr_mean", ascending=False)
    )
    best_bad = (
        summary[summary["split"].eq("test") & summary["event_group"].eq("bad_top10")]
        .sort_values("bio_top1_minus_latest_mean")
        .head(8)
    )
    best_corr = (
        summary[summary["split"].eq("test") & summary["event_group"].eq("bad_top10")]
        .sort_values("bio_corr_mean", ascending=False)
        .head(8)
    )

    lines: List[str] = []
    lines.append("# v288 ECG source-signal route audit")
    lines.append("")
    lines.append("## 本轮目的")
    lines.append("")
    lines.append("- 承接 v287：ECG 最近 1-2 秒有弱诊断苗头，但 v287 的现成 shape-state 特征没有形成可部署改善。")
    lines.append("- 本轮回到 cleaned 200Hz ECG 源信号，重新提取 R 峰/RR、形态、质量和因果同步偏移特征。")
    lines.append("- 仍然使用 v284/v285 同一个 vehicle top40 route gate；本轮不训练轨迹融合模型。")
    lines.append("")
    lines.append("## route gate 判定")
    lines.append("")
    lines.append(table_to_md(decision, ["check", "requirement", "pass", "evidence", "deployable", "route_viable_now"]))
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
    lines.append("## test bad_top10 的最佳诊断结果")
    lines.append("")
    lines.append(
        table_to_md(
            best_bad,
            [
                "feature_set",
                "n",
                "latest_rmse_mean",
                "bio_top1_rmse_mean",
                "bio_top1_minus_latest_mean",
                "bio_top3_oracle_rmse_mean",
                "bio_top3_minus_latest_mean",
                "bio_corr_mean",
            ],
        )
    )
    lines.append("")
    lines.append("## test bad_top10 排序相关最高的 ECG 特征集")
    lines.append("")
    lines.append(
        table_to_md(
            best_corr,
            [
                "feature_set",
                "n",
                "bio_top1_minus_latest_mean",
                "bio_top3_minus_latest_mean",
                "bio_corr_mean",
                "bio_corr_positive_rate",
            ],
        )
    )
    lines.append("")
    lines.append("## 因果同步偏移组")
    lines.append("")
    lines.append(
        table_to_md(
            offset,
            [
                "feature_set",
                "group_value",
                "n",
                "bio_top1_minus_latest_mean",
                "bio_top3_minus_latest_mean",
                "bio_corr_mean",
                "bio_corr_positive_rate",
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
            ],
            max_rows=40,
        )
    )
    lines.append("")
    lines.append("## bad_top10_vehicle_ambiguous 分层")
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
            ],
            max_rows=40,
        )
    )
    lines.append("")
    lines.append("## feature set 审计")
    lines.append("")
    lines.append(table_to_md(feature_audit, ["feature_set", "group_type", "group_value", "candidate_feature_n", "feature_n", "behavior_eta_max", "bad_eta_max", "identity_eta_median"]))
    lines.append("")
    lines.append("## train-only ECG feature screen 摘要")
    lines.append("")
    lines.append(
        table_to_md(
            screen_summary,
            [
                "feature_category",
                "offset_group",
                "duration_group",
                "feature_n",
                "behavior_eta_max",
                "bad_eta_max",
                "identity_eta_median",
                "behavior_identity_score_max",
            ],
            max_rows=50,
        )
    )
    lines.append("")
    lines.append("## ECG 质量摘要")
    lines.append("")
    lines.append(
        table_to_md(
            quality,
            [
                "subject",
                "recording",
                "split",
                "event_n",
                "ok_rate",
                "bio288_baseline_valid_ratio_median",
                "bio288_w_dur2_end0_ecg_valid_ratio_median",
                "bio288_w_dur2_end0_ecg_peak_n_median",
                "bio288_w_dur2_end0_ecg_rr_plausible_rate_median",
            ],
            max_rows=60,
        )
    )
    lines.append("")
    lines.append("## 图表")
    lines.append("")
    for fig in figures:
        lines.append(f"- `{fig}`")
    lines.append("")
    lines.append("## 解释")
    lines.append("")
    route_viable = bool(decision["route_viable_now"].iloc[0]) if len(decision) else False
    if route_viable:
        lines.append("- route gate 通过：ECG 源信号已经形成进入下一步轨迹模型的最低证据。")
    else:
        lines.append("- route gate 未通过：即使回到 ECG 源信号和 R 峰/RR 层，当前 ECG 仍没有形成可部署候选选择收益。")
    lines.append("- 如果 test-best 仍只有弱排序相关，而 validation 选择后的 top1 不赢 latest，则不能把 ECG 解释为已解决差样本问题。")
    lines.append("- 本轮使用的是因果同步偏移窗口；若这些窗口都不通过，后续不应继续靠同类 ECG 特征微调。")
    lines.append("")
    lines.append("## guardrail")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(guardrail, ensure_ascii=False, indent=2))
    lines.append("```")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    print("[v288] 目的：从 cleaned 200Hz ECG 源信号重建 R 峰/RR/短窗形态/同步偏移特征，并验证 route gate。", flush=True)
    np.random.seed(SEED)
    clean_out_dir()

    loaded = V285.V252.load_fixed_inputs()
    manifest = loaded["data"].manifest.copy()
    cand = V284.load_candidate_table()
    context = V284.build_event_context(cand)

    features = build_ecg_source_features(manifest)
    events, _targets = V285.add_targets_and_labels(features, loaded, context)
    cols = numeric_feature_columns(events)
    if len(cols) < 50:
        raise RuntimeError(f"v288 ECG 可用数值特征太少：{len(cols)}")

    screen = feature_screening(events, cols)
    screen_summary = summarize_feature_screen(screen)
    feature_sets, feature_audit = build_feature_sets(screen)
    if len(feature_sets) < 8:
        raise RuntimeError(f"v288 可用 ECG feature set 太少：{len(feature_sets)}")

    quality = summarize_ecg_quality(features)

    per_event_parts = []
    scaler_parts = []
    eval_audit_parts = []
    for name, fs_cols in feature_sets.items():
        print(f"[v288] evaluate feature_set={name} feature_n={len(fs_cols)}", flush=True)
        per_event, scaler, audit = V284.evaluate_feature_set(name, fs_cols, events, cand, context)
        per_event_parts.append(per_event)
        scaler_parts.append(scaler)
        eval_audit_parts.append(audit)

    per_event_all = pd.concat(per_event_parts, ignore_index=True)
    scaler_all = pd.concat(scaler_parts, ignore_index=True)
    eval_audit = pd.concat(eval_audit_parts, ignore_index=True)
    feature_audit = feature_audit.merge(eval_audit, on="feature_set", how="left", suffixes=("", "_eval"))
    expanded = V284.expand_groups(per_event_all)
    summary = V284.summarize_groups(expanded)
    val_test = V284.val_chosen_generalization(summary)
    decision = V284.route_gate_decision(summary, val_test)

    write_csv(features, TABLES / "v288_ecg_source_features.csv")
    write_csv(events, TABLES / "v288_ecg_source_features_with_targets.csv")
    write_csv(screen, TABLES / "v288_train_only_feature_screen.csv")
    write_csv(screen_summary, TABLES / "v288_feature_screen_summary.csv")
    write_csv(quality, TABLES / "v288_ecg_quality_by_recording.csv")
    write_csv(feature_audit, TABLES / "v288_feature_set_audit.csv")
    write_csv(scaler_all, TABLES / "v288_train_scaler_audit.csv")
    write_csv(per_event_all, TABLES / "v288_route_gate_per_event.csv")
    write_csv(summary, TABLES / "v288_route_group_summary.csv")
    write_csv(val_test, TABLES / "v288_val_chosen_generalization.csv")
    write_csv(decision, TABLES / "v288_route_gate_decision.csv")
    write_input_hashes()

    figures = [
        plot_badtop10_delta(summary),
        plot_offset_summary(summary, feature_audit),
        plot_feature_screen(screen_summary),
    ]

    v287_guard = json.loads(V287_GUARDRAIL.read_text(encoding="utf-8")) if V287_GUARDRAIL.exists() else {}
    fixed_latest = summary[
        summary["split"].eq("test")
        & summary["event_group"].eq("bad_top10")
        & summary["feature_set"].eq(feature_audit["feature_set"].iloc[0])
    ]["latest_rmse_mean"]
    test_bad = summary[summary["split"].eq("test") & summary["event_group"].eq("bad_top10")]
    best_top1_delta = float(test_bad["bio_top1_minus_latest_mean"].min()) if not test_bad.empty else math.nan
    best_corr = float(test_bad["bio_corr_mean"].max()) if not test_bad.empty else math.nan

    guardrail: Dict[str, object] = {
        "pass": True,
        "zip_testzip": False,
        "event_n": int(events["event_uid"].nunique()),
        "candidate_rows": int(len(cand)),
        "ecg_source_feature_n": int(len(cols)),
        "feature_set_n": int(len(feature_sets)),
        "uses_post_observation_any": bool(events["bio288_uses_post_observation"].astype(bool).any()),
        "ok_rate": float(events["bio288_status"].astype(str).eq("ok").mean()),
        "baseline_valid_ratio_median": float(pd.to_numeric(events["bio288_baseline_valid_ratio"], errors="coerce").median()),
        "dur2_end0_valid_ratio_median": float(pd.to_numeric(events["bio288_w_dur2_end0_ecg_valid_ratio"], errors="coerce").median()),
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
        "best_test_badtop10_top1_delta": best_top1_delta,
        "best_test_badtop10_corr": best_corr,
        "reused_v260_feature_table": False,
        "test_used_for_current_feature_selection": False,
        "ecg_direction_seeded_by_prior_v287_test_diagnostic": True,
        "v287_source_guardrail_pass": bool(v287_guard.get("pass", False)),
        "v287_source_route_viable_now": bool(v287_guard.get("route_viable_now", False)),
    }
    guardrail["pass"] = bool(
        guardrail["event_n"] > 0
        and guardrail["candidate_rows"] > 0
        and guardrail["ecg_source_feature_n"] >= 50
        and guardrail["feature_set_n"] >= 8
        and not guardrail["uses_post_observation_any"]
        and not guardrail["reused_v260_feature_table"]
        and not guardrail["test_used_for_current_feature_selection"]
        and guardrail["v287_source_guardrail_pass"]
    )
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    report = write_report(feature_audit, screen, screen_summary, quality, summary, val_test, decision, guardrail, figures)
    write_file_inventory()

    first_zip_ok = make_zip()
    guardrail["zip_testzip"] = bool(first_zip_ok)
    guardrail["pass"] = bool(guardrail["pass"] and first_zip_ok)
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    report = write_report(feature_audit, screen, screen_summary, quality, summary, val_test, decision, guardrail, figures)
    write_file_inventory()

    second_zip_ok = make_zip()
    guardrail["zip_testzip"] = bool(second_zip_ok)
    guardrail["pass"] = bool(guardrail["pass"] and second_zip_ok)
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    write_file_inventory()

    if not bool(guardrail["pass"]):
        raise AssertionError("v288 guardrail 未通过：\n" + json.dumps(guardrail, ensure_ascii=False, indent=2))

    print(f"[v288] report={report}", flush=True)
    print(f"[v288] zip={ZIP_PATH}", flush=True)
    print(json.dumps(guardrail, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
