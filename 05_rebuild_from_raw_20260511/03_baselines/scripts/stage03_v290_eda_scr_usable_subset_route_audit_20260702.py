#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v290 EDA/SCR usable-subset source route audit.

本轮目标：
- v288/v289 已分别检查 ECG 和 RESP 源信号，仍未形成可部署 top1 改善；
- v290 检查最后一个较可能有事件前状态含义的源信号：EDA/SCR；
- 不把 9 个 near-constant / missing EDA recording 混进同一个质量结论里，而是同时报告：
  1) 全体事件 route gate；
  2) EDA query event 可用子集 route gate；
- 仍然使用 v278 vehicle top40 candidate route gate。只有 validation 选择后的
  deployable top1 在 test 上低于 latest，才算可部署路线。

边界：
- 只使用 observation_s 之前的 EDA 数据；
- feature screening 只用 train split，validation 选择，test 只报告；
- 不读取 v260/v284 派生生理特征表作为输入；
- 不训练轨迹融合模型，只做 EDA/SCR 源信号可部署性门控。
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
V289_GUARDRAIL = BASELINES / "v289_resp_source_phase_route_audit_20260702" / "logs" / "guardrail_check.json"

OUT = BASELINES / "v290_eda_scr_usable_subset_route_audit_20260702"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
ZIP_PATH = BASELINES / "v290_eda_scr_usable_subset_route_audit_20260702_pack.zip"

SEED = 29002
MIN_FEATURES = 8
MAX_FEATURES_PER_SET = 48

BASELINE_WINDOW = (-90.0, -20.0)
CONTEXT_WINDOW = (-120.0, 0.0)

# EDA/SCR 变化慢，所以保留更长窗口；所有窗口仍严格在 observation_s 之前。
WINDOW_SPECS: Dict[str, Tuple[float, float]] = {
    "pre60_pre30": (-60.0, -30.0),
    "pre30_pre20": (-30.0, -20.0),
    "pre20_pre10": (-20.0, -10.0),
    "pre10_pre5": (-10.0, -5.0),
    "pre5_pre2": (-5.0, -2.0),
    "pre2_0": (-2.0, 0.0),
    "pre5_0": (-5.0, 0.0),
    "pre10_0": (-10.0, 0.0),
    "pre20_0": (-20.0, 0.0),
    "dur5_endm1": (-6.0, -1.0),
    "dur10_endm1": (-11.0, -1.0),
    "dur20_endm2": (-22.0, -2.0),
}

DELTA_PAIRS = [
    ("pre2_0", "pre5_pre2"),
    ("pre5_0", "pre10_pre5"),
    ("pre10_0", "pre20_pre10"),
    ("pre20_0", "pre60_pre30"),
    ("dur5_endm1", "pre10_pre5"),
    ("dur10_endm1", "pre20_pre10"),
]

DELTA_METRICS = [
    "tonic_z_mean",
    "tonic_z_std",
    "tonic_z_slope",
    "tonic_z_last_minus_first",
    "phasic_z_mean",
    "phasic_z_abs_mean",
    "phasic_z_pos_area_per_s",
    "phasic_peak_rate",
    "phasic_peak_amp_mean",
    "phasic_peak_amp_max",
    "phasic_last_peak_age_to_obs",
    "phasic_scr_burst_area_per_s",
    "raw_line_length_per_s",
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
    """按路径导入前序脚本，只复用已验证的数据入口和 route gate 工具。"""

    if not path.exists():
        raise FileNotFoundError(f"缺少脚本：{path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法导入脚本：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V284 = import_module_from_path("stage03_v284_for_v290", V284_SCRIPT)
V285 = import_module_from_path("stage03_v285_for_v290", V285_SCRIPT)
V288 = import_module_from_path("stage03_v288_for_v290", V288_SCRIPT)


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


def moving_average(values: np.ndarray, sample_hz: float, seconds: float) -> np.ndarray:
    """EDA 慢信号平滑；用于从 raw/filt 中构造 tonic 和 fallback phasic。"""

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
    if len(times) == 0:
        return times, values
    stride = max(1, int(round(sample_hz / target_hz)))
    return np.asarray(times[::stride], dtype=float), np.asarray(values[::stride], dtype=float)


def choose_eda_source(arrays: Dict[str, np.ndarray], b_left: int, b_right: int, candidates: List[str]) -> Tuple[str, np.ndarray]:
    """在多个 EDA 列中选基线有效且不是近常数的通道。"""

    best_col = candidates[0]
    best_score = -math.inf
    best_baseline = np.array([], dtype=float)
    for col in candidates:
        if col not in arrays:
            continue
        baseline = np.asarray(arrays[col][b_left:b_right], dtype=float)
        vals = finite(baseline)
        finite_rate = float(np.isfinite(baseline).mean()) if len(baseline) else 0.0
        std = float(np.std(vals)) if vals.size else 0.0
        spread = float(np.quantile(vals, 0.95) - np.quantile(vals, 0.05)) if vals.size >= 20 else 0.0
        score = finite_rate + min(1.0, std / 1e-4) + min(1.0, spread / 1e-4)
        if score > best_score:
            best_col = col
            best_score = score
            best_baseline = baseline
    return best_col, best_baseline


def detect_scr_peaks(times: np.ndarray, values: np.ndarray, sample_hz: float, threshold: float = 0.8) -> np.ndarray:
    """从 phasic EDA z-score 中检测 SCR-like 峰，使用 1 秒 refractory。"""

    v = np.asarray(values, dtype=float)
    if len(v) < 3:
        return np.array([], dtype=int)
    vv = np.where(np.isfinite(v), v, -np.inf)
    candidate = np.r_[False, (vv[1:-1] > vv[:-2]) & (vv[1:-1] >= vv[2:]) & (vv[1:-1] > threshold), False]
    idx = np.flatnonzero(candidate)
    if idx.size == 0:
        return np.array([], dtype=int)
    min_dist = max(1, int(round(sample_hz * 1.0)))
    selected: List[int] = []
    for i in idx[np.argsort(vv[idx])[::-1]]:
        if all(abs(int(i) - int(j)) >= min_dist for j in selected):
            selected.append(int(i))
    return np.asarray(sorted(selected), dtype=int)


def signal_quality(values: np.ndarray) -> Tuple[float, float, float]:
    vals = finite(values)
    if vals.size == 0:
        return 0.0, math.nan, math.nan
    valid_rate = float(np.isfinite(values).mean()) if len(values) else 0.0
    std = float(np.std(vals))
    spread = float(np.quantile(vals, 0.95) - np.quantile(vals, 0.05)) if vals.size >= 20 else math.nan
    return valid_rate, std, spread


def eda_window_features(
    times: np.ndarray,
    raw: np.ndarray,
    tonic: np.ndarray,
    phasic: np.ndarray,
    raw_baseline: np.ndarray,
    tonic_baseline: np.ndarray,
    phasic_baseline: np.ndarray,
    prefix: str,
    sample_hz: float,
    observation_s: float,
) -> Dict[str, float]:
    """提取 EDA tonic / phasic / SCR 窗口特征。"""

    out: Dict[str, float] = {}
    duration = float(times[-1] - times[0]) if len(times) >= 2 else 0.0
    raw_z = robust_z(raw, raw_baseline)
    tonic_z = robust_z(tonic, tonic_baseline)
    phasic_z = robust_z(phasic, phasic_baseline)

    out[f"{prefix}_rows"] = int(len(times))
    out[f"{prefix}_duration_s"] = duration
    out[f"{prefix}_raw_valid_ratio"] = float(np.isfinite(raw_z).mean()) if len(raw_z) else 0.0
    out[f"{prefix}_tonic_valid_ratio"] = float(np.isfinite(tonic_z).mean()) if len(tonic_z) else 0.0
    out[f"{prefix}_phasic_valid_ratio"] = float(np.isfinite(phasic_z).mean()) if len(phasic_z) else 0.0

    for name, arr in [("raw", raw_z), ("tonic", tonic_z), ("phasic", phasic_z)]:
        vals = finite(arr)
        if vals.size == 0:
            for metric in ["z_mean", "z_std", "z_range", "z_abs_mean", "z_abs_p95", "z_slope", "z_last_minus_first"]:
                out[f"{prefix}_{name}_{metric}"] = math.nan
            continue
        out[f"{prefix}_{name}_z_mean"] = float(np.mean(vals))
        out[f"{prefix}_{name}_z_std"] = float(np.std(vals))
        out[f"{prefix}_{name}_z_range"] = float(np.max(vals) - np.min(vals))
        out[f"{prefix}_{name}_z_abs_mean"] = float(np.mean(np.abs(vals)))
        out[f"{prefix}_{name}_z_abs_p95"] = nan_quantile(np.abs(arr), 0.95)
        out[f"{prefix}_{name}_z_slope"] = slope(times, arr)
        valid_idx = np.flatnonzero(np.isfinite(arr))
        out[f"{prefix}_{name}_z_last_minus_first"] = (
            float(arr[valid_idx[-1]] - arr[valid_idx[0]]) if len(valid_idx) >= 2 else math.nan
        )

    dz = np.diff(raw_z)
    dz = dz[np.isfinite(dz)]
    out[f"{prefix}_raw_line_length_per_s"] = safe_div(float(np.sum(np.abs(dz))) if dz.size else math.nan, duration)

    phasic_vals = np.asarray(phasic_z, dtype=float)
    pos = np.where(np.isfinite(phasic_vals) & (phasic_vals > 0), phasic_vals, 0.0)
    out[f"{prefix}_phasic_z_pos_area_per_s"] = safe_div(float(np.sum(pos)) / max(sample_hz, 1.0), duration)
    peaks = detect_scr_peaks(times, phasic_vals, sample_hz, threshold=0.8)
    peak_times = times[peaks] if len(peaks) else np.array([], dtype=float)
    peak_amp = phasic_vals[peaks] if len(peaks) else np.array([], dtype=float)
    out[f"{prefix}_phasic_peak_n"] = int(len(peaks))
    out[f"{prefix}_phasic_peak_rate"] = safe_div(float(len(peaks)), duration)
    out[f"{prefix}_phasic_peak_amp_mean"] = float(np.mean(peak_amp)) if len(peak_amp) else math.nan
    out[f"{prefix}_phasic_peak_amp_max"] = float(np.max(peak_amp)) if len(peak_amp) else math.nan
    out[f"{prefix}_phasic_last_peak_age_to_obs"] = float(observation_s - peak_times[-1]) if len(peak_times) else math.nan
    out[f"{prefix}_phasic_scr_burst_area_per_s"] = safe_div(float(np.sum(np.maximum(peak_amp - 0.8, 0.0))) if len(peak_amp) else 0.0, duration)
    return out


def parse_window_group(feature: str) -> str:
    name = str(feature)
    for win in WINDOW_SPECS:
        if f"bio290_w_{win}_eda_" in name:
            return win
    if name.startswith("bio290_delta_"):
        return "delta"
    return "global"


def parse_offset_group(window_name: str) -> str:
    if window_name in {"pre2_0", "pre5_0", "pre10_0", "pre20_0"}:
        return "end0"
    if "endm1" in window_name:
        return "endm1"
    if "endm2" in window_name:
        return "endm2"
    if window_name == "pre5_pre2":
        return "endm2"
    if window_name == "pre10_pre5":
        return "endm5"
    if window_name == "pre20_pre10":
        return "endm10"
    if window_name == "pre30_pre20":
        return "endm20"
    if window_name == "pre60_pre30":
        return "endm30"
    if window_name == "delta":
        return "delta"
    return "other"


def parse_duration_group(window_name: str) -> str:
    if window_name.startswith("pre"):
        parts = window_name.replace("pre", "").split("_")
        try:
            a = float(parts[0])
            b = 0.0 if parts[1] == "0" else float(parts[1].replace("pre", ""))
            return f"dur{int(abs(a - b))}"
        except Exception:
            return window_name
    m = re.match(r"dur(\d+)_", window_name)
    if m:
        return f"dur{m.group(1)}"
    return window_name


def extract_recording_eda_features(recording_df: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    """对同一 recording 的 delay=0 事件提取 EDA/SCR 源信号特征。"""

    times = pd.to_numeric(recording_df["t_s"], errors="coerce").to_numpy(dtype=float)
    sample_hz = 200.0
    if len(times) >= 5:
        dt = np.diff(times)
        good_dt = dt[np.isfinite(dt) & (dt > 0)]
        if good_dt.size:
            sample_hz = float(1.0 / np.median(good_dt))

    arrays: Dict[str, np.ndarray] = {}
    for col in ["EDA_raw200", "EDA_filt200", "EDA_Tonic", "EDA_Phasic"]:
        if col in recording_df.columns:
            arrays[col] = pd.to_numeric(recording_df[col], errors="coerce").to_numpy(dtype=float)
        else:
            arrays[col] = np.full(len(times), np.nan, dtype=float)

    # recording 级质量：用于后续 EDA 可用子集审计，不用于训练标签。
    raw_valid, raw_std, raw_spread = signal_quality(arrays.get("EDA_filt200", arrays.get("EDA_raw200", np.array([]))))
    phasic_valid, phasic_std, phasic_spread = signal_quality(arrays.get("EDA_Phasic", np.array([])))
    recording_usable = bool(raw_valid >= 0.80 and np.isfinite(raw_spread) and raw_spread > 1e-6)

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
            "bio290_status": "ok",
            "bio290_sample_hz": sample_hz,
            "bio290_uses_post_observation": False,
            "bio290_eda_recording_usable": bool(recording_usable),
            "bio290_recording_raw_valid_ratio": raw_valid,
            "bio290_recording_raw_std": raw_std,
            "bio290_recording_raw_spread": raw_spread,
            "bio290_recording_phasic_valid_ratio": phasic_valid,
            "bio290_recording_phasic_std": phasic_std,
            "bio290_recording_phasic_spread": phasic_spread,
        }

        b_start = max(0.0, obs + BASELINE_WINDOW[0])
        b_end = max(0.0, obs + BASELINE_WINDOW[1])
        b_left = int(np.searchsorted(times, b_start, side="left"))
        b_right = int(np.searchsorted(times, b_end, side="right"))

        raw_col, raw_baseline = choose_eda_source(arrays, b_left, b_right, ["EDA_filt200", "EDA_raw200"])
        tonic_col, tonic_baseline = choose_eda_source(arrays, b_left, b_right, ["EDA_Tonic", "EDA_filt200", "EDA_raw200"])
        phasic_col, phasic_baseline = choose_eda_source(arrays, b_left, b_right, ["EDA_Phasic", "EDA_filt200", "EDA_raw200"])

        raw_full = np.asarray(arrays.get(raw_col, np.full(len(times), np.nan)), dtype=float)
        tonic_full = np.asarray(arrays.get(tonic_col, np.full(len(times), np.nan)), dtype=float)
        phasic_full = np.asarray(arrays.get(phasic_col, np.full(len(times), np.nan)), dtype=float)

        # 若 phasic 派生列缺失或近常数，用 raw - slow(raw) 构造一个因果前窗口内的 fallback phasic。
        ph_valid, ph_std, ph_spread = signal_quality(phasic_baseline)
        if (not np.isfinite(ph_spread)) or ph_spread <= 1e-7 or ph_valid < 0.50:
            slow = moving_average(raw_full, sample_hz, seconds=8.0)
            phasic_full = raw_full - slow
            phasic_baseline = phasic_full[b_left:b_right]
            phasic_col = "raw_minus_slow_fallback"

        out["bio290_raw_chosen_col_code"] = 0 if raw_col == "EDA_filt200" else 1
        out["bio290_tonic_chosen_col_code"] = {"EDA_Tonic": 0, "EDA_filt200": 1, "EDA_raw200": 2}.get(tonic_col, 9)
        out["bio290_phasic_chosen_col_code"] = {"EDA_Phasic": 0, "EDA_filt200": 1, "EDA_raw200": 2, "raw_minus_slow_fallback": 3}.get(phasic_col, 9)
        out["bio290_baseline_rows"] = int(max(0, b_right - b_left))
        out["bio290_baseline_raw_valid_ratio"] = float(np.isfinite(raw_baseline).mean()) if len(raw_baseline) else 0.0
        out["bio290_baseline_tonic_valid_ratio"] = float(np.isfinite(tonic_baseline).mean()) if len(tonic_baseline) else 0.0
        out["bio290_baseline_phasic_valid_ratio"] = float(np.isfinite(phasic_baseline).mean()) if len(phasic_baseline) else 0.0
        out["bio290_baseline_raw_std"] = float(np.std(finite(raw_baseline))) if finite(raw_baseline).size else math.nan
        out["bio290_baseline_raw_spread"] = (
            float(np.quantile(finite(raw_baseline), 0.95) - np.quantile(finite(raw_baseline), 0.05))
            if finite(raw_baseline).size >= 20
            else math.nan
        )
        event_usable = bool(
            recording_usable
            and out["bio290_baseline_raw_valid_ratio"] >= 0.80
            and np.isfinite(out["bio290_baseline_raw_spread"])
            and out["bio290_baseline_raw_spread"] > 1e-6
        )
        out["bio290_eda_event_usable"] = bool(event_usable)

        for win_name, (offset_start, offset_end) in WINDOW_SPECS.items():
            if offset_end > 1e-9:
                out["bio290_uses_post_observation"] = True
            start = max(0.0, obs + offset_start)
            end = max(0.0, obs + offset_end)
            left = int(np.searchsorted(times, start, side="left"))
            right = int(np.searchsorted(times, end, side="right"))
            win_t = times[left:right]
            prefix = f"bio290_w_{win_name}_eda"
            out.update(
                eda_window_features(
                    win_t,
                    raw_full[left:right],
                    tonic_full[left:right],
                    phasic_full[left:right],
                    raw_baseline,
                    tonic_baseline,
                    phasic_baseline,
                    prefix,
                    sample_hz,
                    obs,
                )
            )

        for recent, ref in DELTA_PAIRS:
            for metric in DELTA_METRICS:
                a = out.get(f"bio290_w_{recent}_eda_{metric}", math.nan)
                b = out.get(f"bio290_w_{ref}_eda_{metric}", math.nan)
                out[f"bio290_delta_{recent}_minus_{ref}_eda_{metric}"] = (
                    float(a - b) if np.isfinite(a) and np.isfinite(b) else math.nan
                )
        rows.append(out)
    return pd.DataFrame(rows)


def build_eda_source_features(manifest: pd.DataFrame) -> pd.DataFrame:
    """从 cleaned 200Hz EDA 连续层构造事件级源信号特征。"""

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
            print(f"[v290] missing 200Hz physio {group_i}/{len(grouped)} subject={subject} session={session}", flush=True)
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
                        "bio290_status": "missing_recording",
                        "bio290_uses_post_observation": False,
                        "bio290_eda_recording_usable": False,
                        "bio290_eda_event_usable": False,
                    }
                )
            continue
        print(f"[v290] extract EDA source {group_i}/{len(grouped)} subject={subject} session={session} events={len(g)}", flush=True)
        rec = V285.read_physio_recording(path)
        parts.append(extract_recording_eda_features(rec, g))
    if missing:
        parts.append(pd.DataFrame(missing))
    return pd.concat(parts, ignore_index=True).sort_values("row_index").reset_index(drop=True)


def eta_squared(feature: np.ndarray, labels: Iterable[object]) -> float:
    return V288.eta_squared(feature, labels)


def finite_rate(values: np.ndarray) -> float:
    return V288.finite_rate(values)


def feature_category(col: str) -> str:
    low = col.lower()
    if low.startswith("bio290_delta_"):
        return "temporal_delta"
    if "phasic_peak" in low or "scr" in low or "phasic_z_pos_area" in low:
        return "scr_phasic"
    if "tonic" in low:
        return "tonic"
    if any(k in low for k in ["valid_ratio", "baseline", "recording_", "chosen_col", "usable"]):
        return "quality"
    if any(k in low for k in ["slope", "last_minus", "line_length", "z_range", "z_abs"]):
        return "morph_dynamic"
    return "level"


def infer_window_group(feature: str) -> str:
    name = str(feature)
    for win in WINDOW_SPECS:
        if f"bio290_w_{win}_eda_" in name:
            return win
    if name.startswith("bio290_delta_"):
        return "delta"
    return "global"


def infer_offset_group(feature: str) -> str:
    win = infer_window_group(feature)
    return parse_offset_group(win)


def infer_duration_group(feature: str) -> str:
    win = infer_window_group(feature)
    if win == "delta":
        return "delta"
    return parse_duration_group(win)


def numeric_feature_columns(events: pd.DataFrame) -> List[str]:
    excluded = [
        "_rows",
        "_duration_s",
        "sample_hz",
        "uses_post_observation",
        "baseline_rows",
        "chosen_col_code",
        "_usable",
    ]
    cols: List[str] = []
    for col in events.columns:
        if not col.startswith("bio290_"):
            continue
        if any(s in col for s in excluded):
            continue
        if pd.api.types.is_numeric_dtype(events[col]):
            cols.append(col)
    return cols


def feature_screening(events: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """只用 train split 筛选 EDA/SCR 特征。"""

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
                "signal_family": "eda",
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
    usable = screen[screen["finite_rate_train"].ge(0.60)].copy()
    usable["rank_score"] = (
        usable["behavior_identity_score"].fillna(0.0)
        + 0.8 * usable["bad_identity_score"].fillna(0.0)
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

    add_set("eda_all_top64", "all", "all", usable, n=64)
    low_identity = usable[usable["identity_eta_max"].le(0.10)].copy()
    if len(low_identity) < 32:
        low_identity = usable.sort_values("identity_eta_max", ascending=True).head(max(32, min(96, len(usable))))
    add_set("eda_low_identity_top48", "identity", "low_identity", low_identity, n=48)

    for cat in ["scr_phasic", "tonic", "morph_dynamic", "level", "quality", "temporal_delta"]:
        add_set(f"eda_category_{cat}_top48", "category", cat, usable[usable["feature_category"].eq(cat)], n=48)

    for offset in ["end0", "endm1", "endm2", "endm5", "endm10", "endm20", "endm30", "delta"]:
        add_set(f"eda_offset_{offset}_top32", "offset", offset, usable[usable["offset_group"].eq(offset)], n=32)

    for dur in ["dur2", "dur3", "dur5", "dur10", "dur20", "dur30", "delta"]:
        add_set(f"eda_duration_{dur}_top32", "duration", dur, usable[usable["duration_group"].eq(dur)], n=32)

    for win in ["pre2_0", "pre5_0", "pre10_0", "pre20_0", "dur5_endm1", "dur10_endm1"]:
        add_set(f"eda_window_{win}_top24", "window", win, usable[usable["window_group"].eq(win)], n=24)

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


def summarize_eda_quality(features: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "bio290_recording_raw_valid_ratio",
        "bio290_recording_raw_std",
        "bio290_recording_raw_spread",
        "bio290_recording_phasic_valid_ratio",
        "bio290_recording_phasic_std",
        "bio290_recording_phasic_spread",
        "bio290_baseline_raw_valid_ratio",
        "bio290_baseline_raw_spread",
        "bio290_w_pre10_0_eda_phasic_peak_n",
        "bio290_w_pre10_0_eda_phasic_peak_rate",
    ]
    use_cols = [c for c in cols if c in features.columns]
    out = (
        features.groupby(["subject", "recording", "split"], as_index=False)
        .agg(
            event_n=("event_uid", "nunique"),
            ok_rate=("bio290_status", lambda s: float(pd.Series(s).astype(str).eq("ok").mean())),
            recording_usable=("bio290_eda_recording_usable", "max"),
            event_usable_rate=("bio290_eda_event_usable", "mean"),
            **{f"{c}_median": (c, "median") for c in use_cols},
        )
        .sort_values(["split", "subject", "recording"])
    )
    return out


def expand_groups_with_eda(per_event: pd.DataFrame) -> pd.DataFrame:
    """扩展标准 route group，同时加入 EDA 可用子集。"""

    rows: List[Dict[str, object]] = []
    for _, row in per_event.iterrows():
        groups = ["all"]
        vehicle_ambiguous = bool(row.get("vehicle_ambiguous", False))
        bad_top10 = bool(row.get("bad_top10", False))
        very_bad_top5 = bool(row.get("very_bad_top5", False))
        bad_amb = bool(row.get("bad_top10_vehicle_ambiguous", False))
        eda_usable = bool(row.get("bio290_eda_event_usable", False))
        if vehicle_ambiguous:
            groups.append("vehicle_ambiguous")
        if bad_top10:
            groups.append("bad_top10")
        if very_bad_top5:
            groups.append("very_bad_top5")
        if bad_amb:
            groups.append("bad_top10_vehicle_ambiguous")
        if eda_usable:
            groups.append("eda_usable")
            if vehicle_ambiguous:
                groups.append("vehicle_ambiguous_eda_usable")
            if bad_top10:
                groups.append("bad_top10_eda_usable")
            if bad_amb:
                groups.append("bad_top10_vehicle_ambiguous_eda_usable")
        for group in groups:
            item = row.to_dict()
            item["event_group"] = group
            rows.append(item)
    return pd.DataFrame(rows)


def val_chosen_generalization(summary: pd.DataFrame) -> pd.DataFrame:
    """和 v284 相同的 validation 选择逻辑，但额外报告 EDA 可用子集。"""

    rows: List[Dict[str, object]] = []
    event_groups = [
        "all",
        "vehicle_ambiguous",
        "bad_top10",
        "bad_top10_vehicle_ambiguous",
        "eda_usable",
        "vehicle_ambiguous_eda_usable",
        "bad_top10_eda_usable",
        "bad_top10_vehicle_ambiguous_eda_usable",
    ]
    methods = [
        ("bio_top1", "bio_top1_rmse_mean", "bio_top1_minus_latest_mean", True),
        ("bio_top3_oracle", "bio_top3_oracle_rmse_mean", "bio_top3_minus_latest_mean", False),
        ("bio_top5_oracle", "bio_top5_oracle_rmse_mean", "bio_top5_minus_latest_mean", False),
    ]
    for group in event_groups:
        for method, rmse_col, delta_col, deployable in methods:
            val = summary[summary["split"].eq("val") & summary["event_group"].eq(group) & summary["n"].ge(5)].copy()
            if val.empty:
                continue
            val = val.sort_values([delta_col, "bio_corr_mean"], ascending=[True, False]).reset_index(drop=True)
            chosen = val.iloc[0]
            test = summary[
                summary["split"].eq("test")
                & summary["event_group"].eq(group)
                & summary["feature_set"].astype(str).eq(str(chosen["feature_set"]))
            ]
            if test.empty:
                continue
            t = test.iloc[0]
            rows.append(
                {
                    "event_group": group,
                    "method": method,
                    "deployable": bool(deployable),
                    "val_chosen_feature_set": str(chosen["feature_set"]),
                    "val_n": int(chosen["n"]),
                    "test_n": int(t["n"]),
                    "val_latest_rmse_mean": float(chosen["latest_rmse_mean"]),
                    "val_method_rmse_mean": float(chosen[rmse_col]),
                    "val_delta_vs_latest_mean": float(chosen[delta_col]),
                    "val_corr_mean": float(chosen["bio_corr_mean"]),
                    "test_latest_rmse_mean": float(t["latest_rmse_mean"]),
                    "test_method_rmse_mean": float(t[rmse_col]),
                    "test_delta_vs_latest_mean": float(t[delta_col]),
                    "test_corr_mean": float(t["bio_corr_mean"]),
                    "test_corr_positive_rate": float(t["bio_corr_positive_rate"]),
                    "test_passes_latest": bool(float(t[delta_col]) < -1e-9),
                    "val_and_test_same_direction_gain": bool(float(chosen[delta_col]) < -1e-9 and float(t[delta_col]) < -1e-9),
                }
            )
    return pd.DataFrame(rows)


def subset_route_decision(val_test: pd.DataFrame) -> pd.DataFrame:
    """专门判断 EDA 可用子集是否形成受限可部署路线。"""

    rows = []
    for group in ["eda_usable", "bad_top10_eda_usable", "bad_top10_vehicle_ambiguous_eda_usable"]:
        sub = val_test[val_test["event_group"].eq(group) & val_test["method"].eq("bio_top1")]
        if sub.empty:
            rows.append({"check": f"{group}_top1", "pass": False, "evidence": "missing", "deployable": True})
            continue
        row = sub.iloc[0]
        rows.append(
            {
                "check": f"{group}_top1",
                "pass": bool(float(row["test_delta_vs_latest_mean"]) < -1e-9),
                "evidence": float(row["test_delta_vs_latest_mean"]),
                "deployable": True,
                "val_chosen_feature_set": str(row["val_chosen_feature_set"]),
                "test_n": int(row["test_n"]),
            }
        )
    out = pd.DataFrame(rows)
    out["eda_subset_route_viable_now"] = bool(out["pass"].all())
    return out


def table_to_md(df: pd.DataFrame, cols: List[str] | None = None, max_rows: int = 80) -> str:
    if df is None or df.empty:
        return "_空表_"
    show = df.copy()
    if cols is not None:
        show = show[[c for c in cols if c in show.columns]]
    return show.head(max_rows).to_markdown(index=False)


def plot_badtop10_delta(summary: pd.DataFrame) -> Path:
    path = FIGURES / "v290_badtop10_val_test_delta.png"
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
    ax.set_title("v290 EDA/SCR route gate: bad_top10 top1")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_eda_subset(summary: pd.DataFrame) -> Path:
    path = FIGURES / "v290_eda_usable_subset_delta.png"
    data = summary[
        summary["split"].eq("test")
        & summary["event_group"].isin(["bad_top10", "bad_top10_eda_usable", "bad_top10_vehicle_ambiguous", "bad_top10_vehicle_ambiguous_eda_usable"])
    ].copy()
    if data.empty:
        return path
    best = data.sort_values("bio_top1_minus_latest_mean").groupby("event_group", as_index=False).head(6)
    labels = best["event_group"].astype(str) + "\n" + best["feature_set"].astype(str)
    x = np.arange(len(best))
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x, best["bio_top1_minus_latest_mean"], color="tab:purple")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("test top1 delta vs latest")
    ax.set_title("v290: EDA usable subset vs full bad groups")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_feature_screen(screen_summary: pd.DataFrame) -> Path:
    path = FIGURES / "v290_eda_feature_screen_summary.png"
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
    ax.set_title("v290 EDA/SCR feature screen")
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
        ("physio_signal_availability", V285.PHYSIO_SIGNAL_AVAIL),
        ("v278_candidates", V284.V278_CANDIDATES),
        ("v289_guardrail", V289_GUARDRAIL),
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
    subset_decision: pd.DataFrame,
    guardrail: Dict[str, object],
    figures: List[Path],
) -> Path:
    path = REPORTS / "v290_eda_scr_usable_subset_route_audit_cn.md"
    bad = summary[summary["event_group"].eq("bad_top10") & summary["split"].isin(["val", "test"])].sort_values(
        ["split", "bio_top1_minus_latest_mean"]
    )
    bad_usable = summary[summary["event_group"].eq("bad_top10_eda_usable") & summary["split"].isin(["val", "test"])].sort_values(
        ["split", "bio_top1_minus_latest_mean"]
    )
    amb_usable = summary[
        summary["event_group"].eq("bad_top10_vehicle_ambiguous_eda_usable") & summary["split"].isin(["val", "test"])
    ].sort_values(["split", "bio_top1_minus_latest_mean"])
    best_bad = (
        summary[summary["split"].eq("test") & summary["event_group"].eq("bad_top10")]
        .sort_values("bio_top1_minus_latest_mean")
        .head(8)
    )
    best_usable = (
        summary[summary["split"].eq("test") & summary["event_group"].eq("bad_top10_eda_usable")]
        .sort_values("bio_top1_minus_latest_mean")
        .head(8)
    )
    lines: List[str] = []
    lines.append("# v290 EDA/SCR usable-subset source route audit")
    lines.append("")
    lines.append("## 本轮目的")
    lines.append("")
    lines.append("- 承接 v288/v289：ECG 与 RESP 源信号都没有形成可部署 top1 改善。")
    lines.append("- 本轮回到 cleaned 200Hz EDA 源信号，重建 tonic/phasic/SCR 特征，并显式区分 EDA 可用子集。")
    lines.append("- 仍然只做 vehicle top40 route gate，不训练轨迹融合模型。")
    lines.append("")
    lines.append("## 标准 route gate 判定")
    lines.append("")
    lines.append(table_to_md(decision, ["check", "requirement", "pass", "evidence", "deployable", "route_viable_now"]))
    lines.append("")
    lines.append("## EDA 可用子集 route gate 判定")
    lines.append("")
    lines.append(table_to_md(subset_decision, ["check", "pass", "evidence", "deployable", "val_chosen_feature_set", "test_n", "eda_subset_route_viable_now"]))
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
            max_rows=60,
        )
    )
    lines.append("")
    lines.append("## test bad_top10 最佳 top1 诊断")
    lines.append("")
    lines.append(
        table_to_md(
            best_bad,
            ["feature_set", "n", "latest_rmse_mean", "bio_top1_rmse_mean", "bio_top1_minus_latest_mean", "bio_top3_oracle_rmse_mean", "bio_top3_minus_latest_mean", "bio_corr_mean"],
        )
    )
    lines.append("")
    lines.append("## test bad_top10_eda_usable 最佳 top1 诊断")
    lines.append("")
    lines.append(
        table_to_md(
            best_usable,
            ["feature_set", "n", "latest_rmse_mean", "bio_top1_rmse_mean", "bio_top1_minus_latest_mean", "bio_top3_oracle_rmse_mean", "bio_top3_minus_latest_mean", "bio_corr_mean"],
        )
    )
    lines.append("")
    lines.append("## bad_top10 全体分层")
    lines.append("")
    lines.append(
        table_to_md(
            bad,
            ["feature_set", "split", "n", "latest_rmse_mean", "bio_top1_rmse_mean", "bio_top1_minus_latest_mean", "bio_top3_oracle_rmse_mean", "bio_top3_minus_latest_mean", "bio_corr_mean"],
            max_rows=40,
        )
    )
    lines.append("")
    lines.append("## bad_top10 EDA 可用子集")
    lines.append("")
    lines.append(
        table_to_md(
            bad_usable,
            ["feature_set", "split", "n", "latest_rmse_mean", "bio_top1_rmse_mean", "bio_top1_minus_latest_mean", "bio_top3_oracle_rmse_mean", "bio_top3_minus_latest_mean", "bio_corr_mean"],
            max_rows=40,
        )
    )
    lines.append("")
    lines.append("## bad_top10_vehicle_ambiguous EDA 可用子集")
    lines.append("")
    lines.append(
        table_to_md(
            amb_usable,
            ["feature_set", "split", "n", "latest_rmse_mean", "bio_top1_rmse_mean", "bio_top1_minus_latest_mean", "bio_top3_oracle_rmse_mean", "bio_top3_minus_latest_mean", "bio_corr_mean"],
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
    lines.append("## train-only EDA feature screen 摘要")
    lines.append("")
    lines.append(
        table_to_md(
            screen_summary,
            ["feature_category", "offset_group", "duration_group", "feature_n", "behavior_eta_max", "bad_eta_max", "identity_eta_median", "behavior_identity_score_max"],
            max_rows=50,
        )
    )
    lines.append("")
    lines.append("## EDA 质量摘要")
    lines.append("")
    lines.append(
        table_to_md(
            quality,
            ["subject", "recording", "split", "event_n", "ok_rate", "recording_usable", "event_usable_rate", "bio290_recording_raw_spread_median", "bio290_baseline_raw_spread_median"],
            max_rows=80,
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
    if bool(decision["route_viable_now"].iloc[0]):
        lines.append("- 标准 route gate 通过：EDA/SCR 可以进入下一步轨迹模型。")
    else:
        lines.append("- 标准 route gate 未通过：EDA/SCR 未形成全体样本上的可部署 top1 改善。")
    if bool(subset_decision["eda_subset_route_viable_now"].iloc[0]) if len(subset_decision) else False:
        lines.append("- EDA 可用子集 route gate 通过：这只支持质量受限策略，不等于全体样本改善。")
    else:
        lines.append("- EDA 可用子集 route gate 也未通过：近常数/缺失记录不是唯一瓶颈。")
    lines.append("- 若只有 top3 oracle 或 test-best 诊断变好，不能写成可部署模型改善。")
    lines.append("")
    lines.append("## guardrail")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(guardrail, ensure_ascii=False, indent=2))
    lines.append("```")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    print("[v290] 目的：重建 EDA/SCR 源信号特征，并分别验证全体与 EDA 可用子集 route gate。", flush=True)
    np.random.seed(SEED)
    clean_out_dir()

    loaded = V285.V252.load_fixed_inputs()
    manifest = loaded["data"].manifest.copy()
    cand = V284.load_candidate_table()
    context = V284.build_event_context(cand)

    features = build_eda_source_features(manifest)
    events, _targets = V285.add_targets_and_labels(features, loaded, context)
    cols = numeric_feature_columns(events)
    if len(cols) < 50:
        raise RuntimeError(f"v290 EDA 可用数值特征太少：{len(cols)}")

    screen = feature_screening(events, cols)
    screen_summary = summarize_feature_screen(screen)
    feature_sets, feature_audit = build_feature_sets(screen)
    if len(feature_sets) < 8:
        raise RuntimeError(f"v290 可用 EDA feature set 太少：{len(feature_sets)}")

    quality = summarize_eda_quality(features)

    eda_flags = features[["event_uid", "bio290_eda_recording_usable", "bio290_eda_event_usable"]].copy()
    per_event_parts = []
    scaler_parts = []
    eval_audit_parts = []
    for name, fs_cols in feature_sets.items():
        print(f"[v290] evaluate feature_set={name} feature_n={len(fs_cols)}", flush=True)
        per_event, scaler, audit = V284.evaluate_feature_set(name, fs_cols, events, cand, context)
        per_event = per_event.merge(eda_flags, on="event_uid", how="left")
        per_event["bio290_eda_recording_usable"] = per_event["bio290_eda_recording_usable"].fillna(False).astype(bool)
        per_event["bio290_eda_event_usable"] = per_event["bio290_eda_event_usable"].fillna(False).astype(bool)
        per_event_parts.append(per_event)
        scaler_parts.append(scaler)
        eval_audit_parts.append(audit)

    per_event_all = pd.concat(per_event_parts, ignore_index=True)
    scaler_all = pd.concat(scaler_parts, ignore_index=True)
    eval_audit = pd.concat(eval_audit_parts, ignore_index=True)
    feature_audit = feature_audit.merge(eval_audit, on="feature_set", how="left", suffixes=("", "_eval"))
    expanded = expand_groups_with_eda(per_event_all)
    summary = V284.summarize_groups(expanded)
    val_test = val_chosen_generalization(summary)
    decision = V284.route_gate_decision(summary, val_test)
    subset_dec = subset_route_decision(val_test)

    write_csv(features, TABLES / "v290_eda_source_features.csv")
    write_csv(events, TABLES / "v290_eda_source_features_with_targets.csv")
    write_csv(screen, TABLES / "v290_train_only_feature_screen.csv")
    write_csv(screen_summary, TABLES / "v290_feature_screen_summary.csv")
    write_csv(quality, TABLES / "v290_eda_quality_by_recording.csv")
    write_csv(feature_audit, TABLES / "v290_feature_set_audit.csv")
    write_csv(scaler_all, TABLES / "v290_train_scaler_audit.csv")
    write_csv(per_event_all, TABLES / "v290_route_gate_per_event.csv")
    write_csv(summary, TABLES / "v290_route_group_summary.csv")
    write_csv(val_test, TABLES / "v290_val_chosen_generalization.csv")
    write_csv(decision, TABLES / "v290_route_gate_decision.csv")
    write_csv(subset_dec, TABLES / "v290_eda_subset_route_decision.csv")
    write_input_hashes()

    figures = [plot_badtop10_delta(summary), plot_eda_subset(summary), plot_feature_screen(screen_summary)]

    v289_guard = json.loads(V289_GUARDRAIL.read_text(encoding="utf-8")) if V289_GUARDRAIL.exists() else {}
    fixed_latest = summary[
        summary["split"].eq("test")
        & summary["event_group"].eq("bad_top10")
        & summary["feature_set"].eq(feature_audit["feature_set"].iloc[0])
    ]["latest_rmse_mean"]
    test_bad = summary[summary["split"].eq("test") & summary["event_group"].eq("bad_top10")]
    test_bad_usable = summary[summary["split"].eq("test") & summary["event_group"].eq("bad_top10_eda_usable")]
    best_top1_delta = float(test_bad["bio_top1_minus_latest_mean"].min()) if not test_bad.empty else math.nan
    best_corr = float(test_bad["bio_corr_mean"].max()) if not test_bad.empty else math.nan
    best_usable_delta = float(test_bad_usable["bio_top1_minus_latest_mean"].min()) if not test_bad_usable.empty else math.nan

    guardrail: Dict[str, object] = {
        "pass": True,
        "zip_testzip": False,
        "event_n": int(events["event_uid"].nunique()),
        "candidate_rows": int(len(cand)),
        "eda_source_feature_n": int(len(cols)),
        "feature_set_n": int(len(feature_sets)),
        "uses_post_observation_any": bool(events["bio290_uses_post_observation"].astype(bool).any()),
        "ok_rate": float(events["bio290_status"].astype(str).eq("ok").mean()),
        "eda_recording_usable_event_rate": float(events["bio290_eda_recording_usable"].astype(bool).mean()),
        "eda_event_usable_rate": float(events["bio290_eda_event_usable"].astype(bool).mean()),
        "eda_event_usable_n": int(events["bio290_eda_event_usable"].astype(bool).sum()),
        "fixed_wait_latest_badtop10": float(fixed_latest.iloc[0]) if len(fixed_latest) else math.nan,
        "route_viable_now": bool(decision["route_viable_now"].iloc[0]),
        "eda_subset_route_viable_now": bool(subset_dec["eda_subset_route_viable_now"].iloc[0]) if len(subset_dec) else False,
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
        "best_test_badtop10_eda_usable_top1_delta": best_usable_delta,
        "reused_v260_feature_table": False,
        "test_used_for_current_feature_selection": False,
        "v289_source_guardrail_pass": bool(v289_guard.get("pass", False)),
        "v289_source_route_viable_now": bool(v289_guard.get("route_viable_now", False)),
    }
    guardrail["pass"] = bool(
        guardrail["event_n"] > 0
        and guardrail["candidate_rows"] > 0
        and guardrail["eda_source_feature_n"] >= 50
        and guardrail["feature_set_n"] >= 8
        and not guardrail["uses_post_observation_any"]
        and not guardrail["reused_v260_feature_table"]
        and not guardrail["test_used_for_current_feature_selection"]
        and guardrail["v289_source_guardrail_pass"]
    )
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    report = write_report(feature_audit, screen_summary, quality, summary, val_test, decision, subset_dec, guardrail, figures)
    write_file_inventory()

    first_zip_ok = make_zip()
    guardrail["zip_testzip"] = bool(first_zip_ok)
    guardrail["pass"] = bool(guardrail["pass"] and first_zip_ok)
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    report = write_report(feature_audit, screen_summary, quality, summary, val_test, decision, subset_dec, guardrail, figures)
    write_file_inventory()

    second_zip_ok = make_zip()
    guardrail["zip_testzip"] = bool(second_zip_ok)
    guardrail["pass"] = bool(guardrail["pass"] and second_zip_ok)
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    write_file_inventory()

    if not bool(guardrail["pass"]):
        raise AssertionError("v290 guardrail 未通过：\n" + json.dumps(guardrail, ensure_ascii=False, indent=2))

    print(f"[v290] report={report}", flush=True)
    print(f"[v290] zip={ZIP_PATH}", flush=True)
    print(json.dumps(guardrail, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
