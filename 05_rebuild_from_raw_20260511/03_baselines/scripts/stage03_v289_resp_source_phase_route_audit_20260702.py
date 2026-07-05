#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v289 RESP source phase route audit.

本轮目标：
- v288 已经排除了“ECG 源信号/R 峰/RR/短窗同步偏移能明显解决差样本”的解释；
- v289 转到另一个底层源信号：RESP_filt200 / RESP_raw200；
- 不使用已知弱的 RESP_BPM / RESP_Amplitude 记录级派生列，而是从 cleaned 200Hz
  连续呼吸波形中因果重建呼吸周期、相位、幅值、质量和同步偏移窗口；
- 仍然先进入 v278 vehicle top40 candidate route gate。只有 route gate 通过，
  才能说明 RESP 源信号值得进入更复杂轨迹预测模型。

边界：
- 只使用 observation_s 之前的 RESP 数据；
- feature screening 只用 train split，validation 选择，test 只报告；
- 不读取 v260/v284 派生生理特征表作为输入；
- 不训练轨迹融合模型，只做源信号可部署性门控。
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
V288_SCRIPT = SCRIPTS / "stage03_v288_ecg_source_signal_route_audit_20260702.py"
V288_GUARDRAIL = BASELINES / "v288_ecg_source_signal_route_audit_20260702" / "logs" / "guardrail_check.json"

OUT = BASELINES / "v289_resp_source_phase_route_audit_20260702"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
ZIP_PATH = BASELINES / "v289_resp_source_phase_route_audit_20260702_pack.zip"

SEED = 28902
MIN_FEATURES = 8
MAX_FEATURES_PER_SET = 48

BASELINE_WINDOW = (-60.0, -20.0)
CONTEXT_WINDOW = (-30.0, 0.0)

WINDOW_SPECS: Dict[str, Tuple[float, float]] = {
    "dur2_end0": (-2.0, 0.0),
    "dur3_end0": (-3.0, 0.0),
    "dur5_end0": (-5.0, 0.0),
    "dur10_end0": (-10.0, 0.0),
    "dur2_endm0p5": (-2.5, -0.5),
    "dur3_endm0p5": (-3.5, -0.5),
    "dur5_endm0p5": (-5.5, -0.5),
    "dur2_endm1": (-3.0, -1.0),
    "dur3_endm1": (-4.0, -1.0),
    "dur5_endm1": (-6.0, -1.0),
    "dur3_endm2": (-5.0, -2.0),
    "dur5_endm2": (-7.0, -2.0),
    "pre10_pre5": (-10.0, -5.0),
    "pre20_pre10": (-20.0, -10.0),
}

DELTA_PAIRS = [
    ("dur2_end0", "dur2_endm1"),
    ("dur3_end0", "dur3_endm1"),
    ("dur5_end0", "dur5_endm1"),
    ("dur5_end0", "dur5_endm2"),
    ("dur10_end0", "pre10_pre5"),
]

DELTA_METRICS = [
    "z_mean",
    "z_std",
    "z_range",
    "z_abs_mean",
    "z_slope",
    "dz_abs_mean",
    "line_length_per_s",
    "zero_up_rate",
    "zero_down_rate",
    "breath_bpm",
    "period_mean",
    "period_std",
    "period_cv",
    "amp_median",
    "phase_sin",
    "phase_cos",
    "last_zero_up_age_to_obs",
    "last_peak_age_to_obs",
    "direction_slope",
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
    """按路径导入本项目前序脚本，只复用已验证的数据入口和 route gate 工具。"""

    if not path.exists():
        raise FileNotFoundError(f"缺少脚本：{path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法导入脚本：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V284 = import_module_from_path("stage03_v284_for_v289", V284_SCRIPT)
V285 = import_module_from_path("stage03_v285_for_v289", V285_SCRIPT)
V288 = import_module_from_path("stage03_v288_for_v289", V288_SCRIPT)


def ensure_dirs() -> None:
    for folder in (TABLES, FIGURES, REPORTS, LOGS):
        folder.mkdir(parents=True, exist_ok=True)


def clean_out_dir() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    ensure_dirs()
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finite(values: Iterable[object]) -> np.ndarray:
    return V288.finite(values)


def robust_z(values: np.ndarray, baseline_values: np.ndarray) -> np.ndarray:
    return V288.robust_z(values, baseline_values)


def safe_div(num: float, den: float) -> float:
    return V288.safe_div(num, den)


def slope(times: np.ndarray, vals: np.ndarray) -> float:
    return V288.slope(times, vals)


def nan_quantile(values: np.ndarray, q: float) -> float:
    return V288.nan_quantile(values, q)


def moving_average(values: np.ndarray, sample_hz: float, seconds: float = 0.40) -> np.ndarray:
    """对 RESP 做轻量平滑，降低 200Hz 噪声对零交叉和极值检测的影响。"""

    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    width = max(3, int(round(sample_hz * seconds)))
    if width % 2 == 0:
        width += 1
    vals = finite(arr)
    fill = float(np.median(vals)) if vals.size else 0.0
    filled = np.where(np.isfinite(arr), arr, fill)
    kernel = np.ones(width, dtype=float) / float(width)
    smooth = np.convolve(filled, kernel, mode="same")
    smooth[~np.isfinite(arr)] = np.nan
    return smooth


def downsample_signal(times: np.ndarray, values: np.ndarray, sample_hz: float, target_hz: float = 20.0) -> Tuple[np.ndarray, np.ndarray]:
    """呼吸是低频信号，下采样后做周期/相位检测更稳。"""

    if len(times) == 0:
        return times, values
    stride = max(1, int(round(sample_hz / target_hz)))
    return np.asarray(times[::stride], dtype=float), np.asarray(values[::stride], dtype=float)


def zero_cross_times(times: np.ndarray, values: np.ndarray, direction: str) -> np.ndarray:
    """线性插值零交叉时间；direction 为 up 或 down。"""

    t = np.asarray(times, dtype=float)
    v = np.asarray(values, dtype=float)
    if len(t) < 2:
        return np.array([], dtype=float)
    rows: List[float] = []
    for i in range(len(v) - 1):
        a = v[i]
        b = v[i + 1]
        if not np.isfinite(a) or not np.isfinite(b):
            continue
        ok = (a <= 0 < b) if direction == "up" else (a >= 0 > b)
        if not ok:
            continue
        den = b - a
        frac = 0.0 if abs(den) < 1e-12 else float((0.0 - a) / den)
        rows.append(float(t[i] + frac * (t[i + 1] - t[i])))
    return np.asarray(rows, dtype=float)


def refractory_extrema(times: np.ndarray, values: np.ndarray, kind: str, sample_hz: float) -> np.ndarray:
    """检测 RESP 平滑波形的峰/谷，使用 refractory period 抑制噪声极值。"""

    v = np.asarray(values, dtype=float)
    if len(v) < 3:
        return np.array([], dtype=int)
    vv = np.where(np.isfinite(v), v, np.nan)
    if kind == "peak":
        candidate = np.r_[False, (vv[1:-1] > vv[:-2]) & (vv[1:-1] >= vv[2:]), False]
        score = vv
    else:
        candidate = np.r_[False, (vv[1:-1] < vv[:-2]) & (vv[1:-1] <= vv[2:]), False]
        score = -vv
    idx = np.flatnonzero(candidate & np.isfinite(score))
    if idx.size == 0:
        return np.array([], dtype=int)
    min_dist = max(1, int(round(sample_hz * 1.0)))
    selected: List[int] = []
    for i in idx[np.argsort(score[idx])[::-1]]:
        if all(abs(int(i) - int(j)) >= min_dist for j in selected):
            selected.append(int(i))
    return np.asarray(sorted(selected), dtype=int)


def choose_resp_source(arrays: Dict[str, np.ndarray], b_left: int, b_right: int) -> Tuple[str, np.ndarray]:
    """在 RESP_filt200 / RESP_raw200 间选择基线可用且方差足够的通道。"""

    best_col = "RESP_filt200" if "RESP_filt200" in arrays else "RESP_raw200"
    best_score = -math.inf
    best_baseline = np.array([], dtype=float)
    for col in ["RESP_filt200", "RESP_raw200"]:
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


def context_period(times: np.ndarray, z: np.ndarray, sample_hz: float) -> float:
    """只用 observation 前上下文估计当前事件的呼吸周期。"""

    smooth = moving_average(z, sample_hz, seconds=0.55)
    tt, ss = downsample_signal(times, smooth, sample_hz, target_hz=20.0)
    up = zero_cross_times(tt, ss, "up")
    period = np.diff(up)
    good = period[(period >= 2.0) & (period <= 10.0)]
    if good.size:
        return float(np.median(good))
    # 零交叉不够时，用峰间距做兜底。
    idx = refractory_extrema(tt, ss, "peak", 20.0)
    peak_times = tt[idx] if len(idx) else np.array([], dtype=float)
    p2 = np.diff(peak_times)
    good2 = p2[(p2 >= 2.0) & (p2 <= 10.0)]
    return float(np.median(good2)) if good2.size else math.nan


def morphology_features(times: np.ndarray, raw: np.ndarray, z: np.ndarray, prefix: str, sample_hz: float) -> Dict[str, float]:
    """RESP 窗口形态特征。"""

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
    out[f"{prefix}_line_length_per_s"] = safe_div(float(np.sum(np.abs(dz))) if dz.size else math.nan, duration)
    out[f"{prefix}_flat_step_rate"] = float(np.mean(np.abs(dz) < 1e-6)) if dz.size else math.nan
    out[f"{prefix}_outlier_rate"] = float(np.mean(np.abs(finite_z) > 8.0)) if finite_z.size else math.nan
    return out


def phase_cycle_features(
    times: np.ndarray,
    z: np.ndarray,
    prefix: str,
    sample_hz: float,
    observation_s: float,
    fallback_period: float,
) -> Dict[str, float]:
    """从 RESP 波形重建呼吸周期和因果相位。"""

    out: Dict[str, float] = {}
    duration = float(times[-1] - times[0]) if len(times) >= 2 else 0.0
    if len(times) < 3:
        for metric in [
            "zero_up_n",
            "zero_down_n",
            "zero_up_rate",
            "zero_down_rate",
            "period_mean",
            "period_std",
            "period_cv",
            "breath_bpm",
            "period_plausible_rate",
            "amp_median",
            "peak_n",
            "trough_n",
            "phase",
            "phase_sin",
            "phase_cos",
            "last_zero_up_age_to_obs",
            "last_peak_age_to_obs",
            "direction_slope",
        ]:
            out[f"{prefix}_{metric}"] = math.nan
        return out

    smooth = moving_average(z, sample_hz, seconds=0.55)
    tt, ss = downsample_signal(times, smooth, sample_hz, target_hz=20.0)
    up = zero_cross_times(tt, ss, "up")
    down = zero_cross_times(tt, ss, "down")
    period_all = np.diff(up)
    plausible = (period_all >= 2.0) & (period_all <= 10.0) if period_all.size else np.array([], dtype=bool)
    period = period_all[plausible] if period_all.size else np.array([], dtype=float)
    peak_idx = refractory_extrema(tt, ss, "peak", 20.0)
    trough_idx = refractory_extrema(tt, ss, "trough", 20.0)
    peak_times = tt[peak_idx] if len(peak_idx) else np.array([], dtype=float)
    trough_times = tt[trough_idx] if len(trough_idx) else np.array([], dtype=float)
    peak_vals = ss[peak_idx] if len(peak_idx) else np.array([], dtype=float)
    trough_vals = ss[trough_idx] if len(trough_idx) else np.array([], dtype=float)

    # 粗略幅值：窗口内峰值和谷值中位数之差，不依赖未来点。
    amp = math.nan
    if len(peak_vals) and len(trough_vals):
        amp = float(np.median(peak_vals) - np.median(trough_vals))

    out[f"{prefix}_zero_up_n"] = int(len(up))
    out[f"{prefix}_zero_down_n"] = int(len(down))
    out[f"{prefix}_zero_up_rate"] = safe_div(float(len(up)), duration)
    out[f"{prefix}_zero_down_rate"] = safe_div(float(len(down)), duration)
    out[f"{prefix}_period_mean"] = float(np.mean(period)) if period.size else math.nan
    out[f"{prefix}_period_std"] = float(np.std(period)) if period.size else math.nan
    out[f"{prefix}_period_cv"] = safe_div(float(np.std(period)) if period.size else math.nan, float(np.mean(period)) if period.size else math.nan)
    out[f"{prefix}_breath_bpm"] = 60.0 / float(np.mean(period)) if period.size else math.nan
    out[f"{prefix}_period_plausible_rate"] = float(plausible.mean()) if period_all.size else math.nan
    out[f"{prefix}_amp_median"] = amp
    out[f"{prefix}_peak_n"] = int(len(peak_times))
    out[f"{prefix}_trough_n"] = int(len(trough_times))

    end_time = float(times[-1]) if len(times) else observation_s
    last_up = up[up <= end_time]
    period_for_phase = float(np.median(period)) if period.size else fallback_period
    if len(last_up) and np.isfinite(period_for_phase) and period_for_phase > 1e-6:
        age = float(end_time - last_up[-1])
        phase = float((age % period_for_phase) / period_for_phase)
        out[f"{prefix}_phase"] = phase
        out[f"{prefix}_phase_sin"] = float(np.sin(2.0 * np.pi * phase))
        out[f"{prefix}_phase_cos"] = float(np.cos(2.0 * np.pi * phase))
        out[f"{prefix}_last_zero_up_age_to_obs"] = float(observation_s - last_up[-1])
    else:
        out[f"{prefix}_phase"] = math.nan
        out[f"{prefix}_phase_sin"] = math.nan
        out[f"{prefix}_phase_cos"] = math.nan
        out[f"{prefix}_last_zero_up_age_to_obs"] = math.nan
    out[f"{prefix}_last_peak_age_to_obs"] = float(observation_s - peak_times[-1]) if len(peak_times) else math.nan

    # 最近 0.8 秒方向斜率：近似判断当前处于吸气/呼气上升还是下降段。
    recent_mask = tt >= (end_time - 0.8)
    out[f"{prefix}_direction_slope"] = slope(tt[recent_mask], ss[recent_mask]) if int(recent_mask.sum()) >= 2 else math.nan
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


def extract_recording_resp_features(recording_df: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    """对同一 recording 的 delay=0 事件提取 RESP 源信号周期/相位特征。"""

    times = pd.to_numeric(recording_df["t_s"], errors="coerce").to_numpy(dtype=float)
    sample_hz = 200.0
    if len(times) >= 5:
        dt = np.diff(times)
        good_dt = dt[np.isfinite(dt) & (dt > 0)]
        if good_dt.size:
            sample_hz = float(1.0 / np.median(good_dt))

    arrays: Dict[str, np.ndarray] = {}
    for col in ["RESP_filt200", "RESP_raw200"]:
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
            "bio289_status": "ok",
            "bio289_sample_hz": sample_hz,
            "bio289_uses_post_observation": False,
        }

        b_start = max(0.0, obs + BASELINE_WINDOW[0])
        b_end = max(0.0, obs + BASELINE_WINDOW[1])
        b_left = int(np.searchsorted(times, b_start, side="left"))
        b_right = int(np.searchsorted(times, b_end, side="right"))
        chosen_col, baseline = choose_resp_source(arrays, b_left, b_right)
        full = np.asarray(arrays.get(chosen_col, np.full(len(times), np.nan)), dtype=float)
        out["bio289_resp_chosen_col_code"] = 0 if chosen_col == "RESP_filt200" else 1
        out["bio289_baseline_rows"] = int(max(0, b_right - b_left))
        out["bio289_baseline_valid_ratio"] = float(np.isfinite(baseline).mean()) if len(baseline) else 0.0
        out["bio289_baseline_std"] = float(np.std(finite(baseline))) if finite(baseline).size else math.nan

        c_start = max(0.0, obs + CONTEXT_WINDOW[0])
        c_end = max(0.0, obs + CONTEXT_WINDOW[1])
        c_left = int(np.searchsorted(times, c_start, side="left"))
        c_right = int(np.searchsorted(times, c_end, side="right"))
        context_z = robust_z(full[c_left:c_right], baseline)
        context_t = times[c_left:c_right]
        period_ctx = context_period(context_t, context_z, sample_hz)
        out["bio289_context_period_s"] = period_ctx
        out["bio289_context_bpm"] = 60.0 / period_ctx if np.isfinite(period_ctx) and period_ctx > 1e-6 else math.nan

        for win_name, (offset_start, offset_end) in WINDOW_SPECS.items():
            if offset_end > 1e-9:
                out["bio289_uses_post_observation"] = True
            start = max(0.0, obs + offset_start)
            end = max(0.0, obs + offset_end)
            left = int(np.searchsorted(times, start, side="left"))
            right = int(np.searchsorted(times, end, side="right"))
            win_t = times[left:right]
            raw = full[left:right]
            z = robust_z(raw, baseline)
            prefix = f"bio289_w_{win_name}_resp"
            out.update(morphology_features(win_t, raw, z, prefix, sample_hz))
            out.update(phase_cycle_features(win_t, z, prefix, sample_hz, obs, period_ctx))

        for recent, ref in DELTA_PAIRS:
            for metric in DELTA_METRICS:
                a = out.get(f"bio289_w_{recent}_resp_{metric}", math.nan)
                b = out.get(f"bio289_w_{ref}_resp_{metric}", math.nan)
                out[f"bio289_delta_{recent}_minus_{ref}_resp_{metric}"] = (
                    float(a - b) if np.isfinite(a) and np.isfinite(b) else math.nan
                )
        rows.append(out)
    return pd.DataFrame(rows)


def build_resp_source_features(manifest: pd.DataFrame) -> pd.DataFrame:
    """从 cleaned 200Hz RESP 连续层构造事件级源信号特征。"""

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
            print(f"[v289] missing 200Hz physio {group_i}/{len(grouped)} subject={subject} session={session}", flush=True)
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
                        "bio289_status": "missing_recording",
                        "bio289_uses_post_observation": False,
                    }
                )
            continue
        print(f"[v289] extract RESP source {group_i}/{len(grouped)} subject={subject} session={session} events={len(g)}", flush=True)
        rec = V285.read_physio_recording(path)
        parts.append(extract_recording_resp_features(rec, g))
    if missing:
        parts.append(pd.DataFrame(missing))
    return pd.concat(parts, ignore_index=True).sort_values("row_index").reset_index(drop=True)


def eta_squared(feature: np.ndarray, labels: Iterable[object]) -> float:
    return V288.eta_squared(feature, labels)


def finite_rate(values: np.ndarray) -> float:
    return V288.finite_rate(values)


def infer_window_group(feature: str) -> str:
    name = str(feature)
    for win in WINDOW_SPECS:
        if f"bio289_w_{win}_resp_" in name:
            return win
    if name.startswith("bio289_delta_"):
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
    if low.startswith("bio289_delta_"):
        return "temporal_delta"
    if any(k in low for k in ["phase", "zero_", "period", "breath_bpm", "peak_", "trough_", "amp_", "last_zero", "last_peak"]):
        return "phase_cycle"
    if any(k in low for k in ["valid_ratio", "baseline", "flat", "outlier", "chosen_col", "context_period", "context_bpm"]):
        return "quality"
    if any(k in low for k in ["dz_", "line_length", "z_slope", "last_minus", "z_abs", "z_range", "direction_slope"]):
        return "morph_dynamic"
    return "morph_level"


def numeric_feature_columns(events: pd.DataFrame) -> List[str]:
    """选择进入 route gate 的 RESP 数值特征，排除行数和显式元数据。"""

    excluded = [
        "_rows",
        "_duration_s",
        "sample_hz",
        "uses_post_observation",
        "baseline_rows",
        "resp_chosen_col_code",
    ]
    cols: List[str] = []
    for col in events.columns:
        if not col.startswith("bio289_"):
            continue
        if any(s in col for s in excluded):
            continue
        if pd.api.types.is_numeric_dtype(events[col]):
            cols.append(col)
    return cols


def feature_screening(events: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """只用 train split 筛选 RESP 特征。"""

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
                "signal_family": "resp",
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

    add_set("resp_all_top64", "all", "all", usable, n=64)
    low_identity = usable[usable["identity_eta_max"].le(0.10)].copy()
    if len(low_identity) < 32:
        low_identity = usable.sort_values("identity_eta_max", ascending=True).head(max(32, min(96, len(usable))))
    add_set("resp_low_identity_top48", "identity", "low_identity", low_identity, n=48)

    for cat in ["phase_cycle", "morph_dynamic", "morph_level", "quality", "temporal_delta"]:
        add_set(f"resp_category_{cat}_top48", "category", cat, usable[usable["feature_category"].eq(cat)], n=48)

    for offset in ["end0", "endm0p5", "endm1", "endm2", "endm5", "endm10", "delta"]:
        add_set(f"resp_offset_{offset}_top32", "offset", offset, usable[usable["offset_group"].eq(offset)], n=32)

    for dur in ["dur2", "dur3", "dur5", "dur10", "pre10_pre5", "pre20_pre10", "delta"]:
        add_set(f"resp_duration_{dur}_top32", "duration", dur, usable[usable["duration_group"].eq(dur)], n=32)

    for win in ["dur2_end0", "dur3_end0", "dur5_end0", "dur2_endm1", "dur3_endm1", "dur5_endm1"]:
        add_set(f"resp_window_{win}_top24", "window", win, usable[usable["window_group"].eq(win)], n=24)

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


def summarize_resp_quality(features: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "bio289_baseline_valid_ratio",
        "bio289_baseline_std",
        "bio289_context_period_s",
        "bio289_context_bpm",
        "bio289_w_dur5_end0_resp_valid_ratio",
        "bio289_w_dur5_end0_resp_zero_up_n",
        "bio289_w_dur5_end0_resp_period_plausible_rate",
        "bio289_w_dur5_end0_resp_amp_median",
    ]
    use_cols = [c for c in cols if c in features.columns]
    return (
        features.groupby(["subject", "recording", "split"], as_index=False)
        .agg(
            event_n=("event_uid", "nunique"),
            ok_rate=("bio289_status", lambda s: float(pd.Series(s).astype(str).eq("ok").mean())),
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
    path = FIGURES / "v289_badtop10_val_test_delta.png"
    data = summary[summary["event_group"].eq("bad_top10") & summary["split"].isin(["val", "test"])].copy()
    if data.empty:
        return path
    order = (
        data[data["split"].eq("test")]
        .sort_values("bio_top1_minus_latest_mean")
        .head(20)["feature_set"]
        .astype(str)
        .tolist()
    )
    x = np.arange(len(order))
    width = 0.35
    fig, ax = plt.subplots(figsize=(13, 5))
    for i, split in enumerate(["val", "test"]):
        vals = []
        for fs in order:
            sub = data[data["feature_set"].astype(str).eq(fs) & data["split"].eq(split)]
            vals.append(float(sub["bio_top1_minus_latest_mean"].iloc[0]) if not sub.empty else np.nan)
        ax.bar(x + (i - 0.5) * width, vals, width=width, label=f"{split} top1")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in order], fontsize=8)
    ax.set_ylabel("RMSE delta vs latest, lower is better")
    ax.set_title("v289 RESP source phase route gate: bad_top10 top1")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_offset_summary(summary: pd.DataFrame, feature_audit: pd.DataFrame) -> Path:
    path = FIGURES / "v289_resp_offset_group_summary.png"
    data = summary[summary["split"].eq("test") & summary["event_group"].eq("bad_top10_vehicle_ambiguous")].copy()
    data = data.merge(feature_audit[["feature_set", "group_type", "group_value"]], on="feature_set", how="left")
    data = data[data["group_type"].eq("offset")].sort_values("bio_corr_mean", ascending=False)
    if data.empty:
        return path
    x = np.arange(len(data))
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(x - 0.18, data["bio_corr_mean"], width=0.36, color="tab:blue", label="rank corr")
    ax1.axhline(0.05, color="tab:blue", linestyle="--", linewidth=1)
    ax1.set_ylabel("test bad ambiguous rank corr")
    ax2 = ax1.twinx()
    ax2.bar(x + 0.18, data["bio_top1_minus_latest_mean"], width=0.36, color="tab:orange", label="top1 delta")
    ax2.axhline(0, color="tab:orange", linestyle="--", linewidth=1)
    ax2.set_ylabel("top1 delta vs latest")
    ax1.set_xticks(x)
    ax1.set_xticklabels(data["group_value"].astype(str), rotation=30, ha="right")
    ax1.set_title("v289 RESP causal offset groups on bad_top10_vehicle_ambiguous")
    ax1.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_feature_screen(screen_summary: pd.DataFrame) -> Path:
    path = FIGURES / "v289_resp_feature_screen_summary.png"
    if screen_summary.empty:
        return path
    data = screen_summary.sort_values("behavior_identity_score_max", ascending=False).head(24)
    labels = [f"{r.feature_category}\n{r.offset_group}/{r.duration_group}" for r in data.itertuples(index=False)]
    x = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x, data["behavior_identity_score_max"], label="behavior/identity score")
    ax.plot(x, data["identity_eta_median"], color="tab:red", marker="o", linewidth=1, label="identity eta median")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("train-only score")
    ax.set_title("v289 RESP source feature screen")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_input_hashes() -> None:
    rows = []
    for name, path in [
        ("v284_script", V284_SCRIPT),
        ("v285_script", V285_SCRIPT),
        ("v288_script", V288_SCRIPT),
        ("physio_inventory", V285.PHYSIO_INVENTORY),
        ("v278_candidates", V284.V278_CANDIDATES),
        ("v288_guardrail", V288_GUARDRAIL),
    ]:
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
    screen_summary: pd.DataFrame,
    quality: pd.DataFrame,
    summary: pd.DataFrame,
    val_test: pd.DataFrame,
    decision: pd.DataFrame,
    guardrail: Dict[str, object],
    figures: List[Path],
) -> Path:
    path = REPORTS / "v289_resp_source_phase_route_audit_cn.md"
    bad = summary[summary["event_group"].eq("bad_top10") & summary["split"].isin(["val", "test"])].sort_values(
        ["split", "bio_top1_minus_latest_mean"]
    )
    amb = summary[
        summary["event_group"].eq("bad_top10_vehicle_ambiguous") & summary["split"].isin(["val", "test"])
    ].sort_values(["split", "bio_top1_minus_latest_mean"])
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
    offset = (
        summary[summary["split"].eq("test") & summary["event_group"].eq("bad_top10_vehicle_ambiguous")]
        .merge(feature_audit[["feature_set", "group_type", "group_value"]], on="feature_set", how="left")
        .query("group_type == 'offset'")
        .sort_values("bio_corr_mean", ascending=False)
    )

    lines: List[str] = []
    lines.append("# v289 RESP source phase route audit")
    lines.append("")
    lines.append("## 本轮目的")
    lines.append("")
    lines.append("- 承接 v288：ECG 源信号未形成可部署 top1 改善。")
    lines.append("- 本轮回到 cleaned 200Hz RESP 源信号，重建呼吸周期、相位、幅值、质量和因果同步偏移特征。")
    lines.append("- 不使用 RESP_BPM / RESP_Amplitude 这类已知弱派生列；仍然只做 route gate，不训练轨迹融合模型。")
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
    lines.append("## test bad_top10 最佳 top1 诊断")
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
    lines.append("## test bad_top10 排序相关最高特征集")
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
    lines.append(
        table_to_md(
            feature_audit,
            ["feature_set", "group_type", "group_value", "candidate_feature_n", "feature_n", "behavior_eta_max", "bad_eta_max", "identity_eta_median"],
        )
    )
    lines.append("")
    lines.append("## train-only RESP feature screen 摘要")
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
    lines.append("## RESP 质量摘要")
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
                "bio289_baseline_valid_ratio_median",
                "bio289_context_period_s_median",
                "bio289_context_bpm_median",
                "bio289_w_dur5_end0_resp_zero_up_n_median",
                "bio289_w_dur5_end0_resp_period_plausible_rate_median",
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
        lines.append("- route gate 通过：RESP 源信号已经形成进入下一步轨迹模型的最低证据。")
    else:
        lines.append("- route gate 未通过：即使重建 RESP 周期/相位，当前呼吸源信号仍没有形成可部署候选选择收益。")
    lines.append("- 若只有 top3 oracle 或 corr 有弱苗头，而 validation 选出的 top1 不赢 latest，不能写成差样本本质改善。")
    lines.append("")
    lines.append("## guardrail")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(guardrail, ensure_ascii=False, indent=2))
    lines.append("```")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    print("[v289] 目的：从 cleaned 200Hz RESP 源信号重建周期/相位/幅值/同步偏移特征，并验证 route gate。", flush=True)
    np.random.seed(SEED)
    clean_out_dir()

    loaded = V285.V252.load_fixed_inputs()
    manifest = loaded["data"].manifest.copy()
    cand = V284.load_candidate_table()
    context = V284.build_event_context(cand)

    features = build_resp_source_features(manifest)
    events, _targets = V285.add_targets_and_labels(features, loaded, context)
    cols = numeric_feature_columns(events)
    if len(cols) < 50:
        raise RuntimeError(f"v289 RESP 可用数值特征太少：{len(cols)}")

    screen = feature_screening(events, cols)
    screen_summary = summarize_feature_screen(screen)
    feature_sets, feature_audit = build_feature_sets(screen)
    if len(feature_sets) < 8:
        raise RuntimeError(f"v289 可用 RESP feature set 太少：{len(feature_sets)}")

    quality = summarize_resp_quality(features)

    per_event_parts = []
    scaler_parts = []
    eval_audit_parts = []
    for name, fs_cols in feature_sets.items():
        print(f"[v289] evaluate feature_set={name} feature_n={len(fs_cols)}", flush=True)
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

    write_csv(features, TABLES / "v289_resp_source_features.csv")
    write_csv(events, TABLES / "v289_resp_source_features_with_targets.csv")
    write_csv(screen, TABLES / "v289_train_only_feature_screen.csv")
    write_csv(screen_summary, TABLES / "v289_feature_screen_summary.csv")
    write_csv(quality, TABLES / "v289_resp_quality_by_recording.csv")
    write_csv(feature_audit, TABLES / "v289_feature_set_audit.csv")
    write_csv(scaler_all, TABLES / "v289_train_scaler_audit.csv")
    write_csv(per_event_all, TABLES / "v289_route_gate_per_event.csv")
    write_csv(summary, TABLES / "v289_route_group_summary.csv")
    write_csv(val_test, TABLES / "v289_val_chosen_generalization.csv")
    write_csv(decision, TABLES / "v289_route_gate_decision.csv")
    write_input_hashes()

    figures = [
        plot_badtop10_delta(summary),
        plot_offset_summary(summary, feature_audit),
        plot_feature_screen(screen_summary),
    ]

    v288_guard = json.loads(V288_GUARDRAIL.read_text(encoding="utf-8")) if V288_GUARDRAIL.exists() else {}
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
        "resp_source_feature_n": int(len(cols)),
        "feature_set_n": int(len(feature_sets)),
        "uses_post_observation_any": bool(events["bio289_uses_post_observation"].astype(bool).any()),
        "ok_rate": float(events["bio289_status"].astype(str).eq("ok").mean()),
        "baseline_valid_ratio_median": float(pd.to_numeric(events["bio289_baseline_valid_ratio"], errors="coerce").median()),
        "context_period_s_median": float(pd.to_numeric(events["bio289_context_period_s"], errors="coerce").median()),
        "context_bpm_median": float(pd.to_numeric(events["bio289_context_bpm"], errors="coerce").median()),
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
        "v288_source_guardrail_pass": bool(v288_guard.get("pass", False)),
        "v288_source_route_viable_now": bool(v288_guard.get("route_viable_now", False)),
    }
    guardrail["pass"] = bool(
        guardrail["event_n"] > 0
        and guardrail["candidate_rows"] > 0
        and guardrail["resp_source_feature_n"] >= 50
        and guardrail["feature_set_n"] >= 8
        and not guardrail["uses_post_observation_any"]
        and not guardrail["reused_v260_feature_table"]
        and not guardrail["test_used_for_current_feature_selection"]
        and guardrail["v288_source_guardrail_pass"]
    )
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    report = write_report(feature_audit, screen_summary, quality, summary, val_test, decision, guardrail, figures)
    write_file_inventory()

    first_zip_ok = make_zip()
    guardrail["zip_testzip"] = bool(first_zip_ok)
    guardrail["pass"] = bool(guardrail["pass"] and first_zip_ok)
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    report = write_report(feature_audit, screen_summary, quality, summary, val_test, decision, guardrail, figures)
    write_file_inventory()

    second_zip_ok = make_zip()
    guardrail["zip_testzip"] = bool(second_zip_ok)
    guardrail["pass"] = bool(guardrail["pass"] and second_zip_ok)
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    write_file_inventory()

    if not bool(guardrail["pass"]):
        raise AssertionError("v289 guardrail 未通过：\n" + json.dumps(guardrail, ensure_ascii=False, indent=2))

    print(f"[v289] report={report}", flush=True)
    print(f"[v289] zip={ZIP_PATH}", flush=True)
    print(json.dumps(guardrail, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
