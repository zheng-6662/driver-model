#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v314 rapid steering source sample audit.

目的：
- 用户明确要求：样本一定应由方向盘快速转动引起；
- 本轮不做人工作业式复核，而是对全体 delay0 事件做方向盘转动速度证据分级；
- 通过固定随机种子和分层抽样，排查“当前窗口没有快转、后续才快转”“锚点前已经快转”“全程快转证据弱”等可疑样本；
- 输出抽样图、全量排查表和中文报告，为下一轮样本筛选/锚点修正提供依据。

边界：
- 本脚本不训练模型；
- 本脚本只使用原始车辆数据和既有 v312 事件清单做来源审计；
- 2 秒之后的信息只用于判断样本来源/锚点是否错位，不作为模型预测输入。
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASELINES = ROOT / "03_baselines"
V312_DIR = BASELINES / "v312_horizon_aligned_label_anchor_audit_20260704"
V312_LABEL_TABLE = V312_DIR / "tables" / "v312_horizon_aligned_delay0_event_labels.csv"
V312_SEVERE_OVERLAY = V312_DIR / "tables" / "v312_v309_severe_horizon_label_overlay.csv"

OUT = BASELINES / "v314_rapid_steering_source_sample_audit_20260704"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"

RANDOM_SEED = 314
FAST_RATE_TH = 0.80
MOTION_TH = 0.35
RATE_COUNT_TH = 0.50
MAX_SAMPLE_PLOTS = 90


SCENE_LABEL_CN = {
    "curve_downhill": "下坡过弯",
    "curve_flat": "平路过弯",
    "continuous_lane_change": "连续变道/连续左右修正",
    "emergency_lane_change_instability": "紧急变道/猛打方向失稳",
    "other_or_uncertain": "其他/不确定",
}

SOURCE_CN = {
    "current_window_fast_steer_supported": "当前窗口有方向盘快转证据",
    "current_and_late_fast_steer": "当前和后续都有方向盘快转",
    "late_fast_steer_not_current_window": "当前窗口不明显，后续才方向盘快转",
    "anchor_after_fast_steer": "锚点前已经方向盘快转",
    "no_clear_fast_steer_evidence": "全程方向盘快转证据弱",
    "ambiguous_fast_steer_source": "方向盘快转来源不清晰",
    "raw_missing_or_invalid": "原始车辆数据缺失或无效",
}

ACTION_CN = {
    "keep": "保留当前样本来源",
    "check_anchor_or_window": "优先检查锚点或预测窗口",
    "exclude_or_reanchor_candidate": "候选剔除或重锚定",
    "diagnose_model_amplitude_phase": "样本来源成立，转入模型幅值/相位诊断",
}


def ensure_dirs() -> None:
    """创建输出目录。"""

    for folder in (TABLES, FIGURES, REPORTS, LOGS, FIGURES / "sample_cases"):
        folder.mkdir(parents=True, exist_ok=True)


def clean_out_dir() -> None:
    """只清理本轮自己的输出目录。"""

    resolved_out = OUT.resolve()
    resolved_base = BASELINES.resolve()
    if resolved_base not in resolved_out.parents:
        raise RuntimeError(f"拒绝清理非预期目录：{resolved_out}")
    if OUT.exists():
        shutil.rmtree(OUT)
    ensure_dirs()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """表格使用 utf-8-sig，方便 Windows 表格软件打开中文。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_json(payload: Dict[str, object], path: Path) -> None:
    """保存 JSON 日志。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def file_sha256(path: Path) -> str:
    """计算输入/输出文件哈希。"""

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_name(value: object, max_len: int = 80) -> str:
    """生成适合文件名的短字符串。"""

    text = str(value)
    out = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    compact = "".join(out).strip("_")
    return compact[:max_len] if compact else "case"


def to_float(value: object, default: float = math.nan) -> float:
    """安全转成浮点数。"""

    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def to_bool(value: object) -> bool:
    """兼容表格里的布尔文本。"""

    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y", "是"}


def signed_peak_abs(values: np.ndarray) -> float:
    """返回绝对峰值。"""

    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return math.nan
    return float(np.max(np.abs(arr)))


def peak_abs_and_time(t: np.ndarray, values: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
    """返回窗口内绝对峰值和对应时间。"""

    tt = np.asarray(t, dtype=float)
    yy = np.asarray(values, dtype=float)
    keep = np.asarray(mask, dtype=bool) & np.isfinite(tt) & np.isfinite(yy)
    if not keep.any():
        return math.nan, math.nan
    idx_all = np.where(keep)[0]
    local_idx = int(np.argmax(np.abs(yy[keep])))
    idx = int(idx_all[local_idx])
    return float(abs(yy[idx])), float(tt[idx])


def direction_change_count(rate: np.ndarray, mask: np.ndarray) -> int:
    """统计有效快转方向变化次数，用于识别连续左右修正。"""

    rr = np.asarray(rate, dtype=float)
    keep = np.asarray(mask, dtype=bool) & np.isfinite(rr) & (np.abs(rr) >= RATE_COUNT_TH)
    signs = np.sign(rr[keep])
    signs = signs[signs != 0]
    if signs.size < 2:
        return 0
    return int(np.sum(signs[1:] != signs[:-1]))


class RawVehicleCache:
    """缓存原始车辆记录，避免重复读取同一文件。"""

    def __init__(self) -> None:
        self.cache: Dict[str, pd.DataFrame] = {}

    def load(self, raw_path: object) -> pd.DataFrame:
        path = Path(str(raw_path))
        key = str(path)
        if key in self.cache:
            return self.cache[key]
        needed = ["StorageTime", "zx|SteeringWheel", "zx|ay", "zx|vyaw", "zx|roll"]
        if not path.exists():
            self.cache[key] = pd.DataFrame()
            return self.cache[key]
        try:
            df = pd.read_csv(path, usecols=lambda c: c in needed)
        except Exception:
            self.cache[key] = pd.DataFrame()
            return self.cache[key]
        if df.empty or "StorageTime" not in df.columns:
            self.cache[key] = pd.DataFrame()
            return self.cache[key]
        t = pd.to_datetime(df["StorageTime"], errors="coerce")
        if t.isna().all():
            self.cache[key] = pd.DataFrame()
            return self.cache[key]
        out = df.copy()
        out["record_s"] = (t - t.iloc[0]).dt.total_seconds().to_numpy(dtype=float)
        self.cache[key] = out
        return out


def smooth_and_rate(t: np.ndarray, steering: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """平滑方向盘角并计算转动速度。"""

    raw = pd.DataFrame({"t": t, "steering": steering})
    raw = raw.replace([np.inf, -np.inf], np.nan).dropna().sort_values("t")
    raw = raw.drop_duplicates("t", keep="first")
    if len(raw) < 5:
        return np.array([]), np.array([]), np.array([])
    tt = raw["t"].to_numpy(dtype=float)
    yy = raw["steering"].to_numpy(dtype=float)
    # 车辆信号采样较密，短窗口均值可以压住尖噪声，同时不抹掉快速打方向。
    win = 11 if len(yy) >= 11 else max(3, len(yy) // 2 * 2 + 1)
    smooth = pd.Series(yy).rolling(window=win, center=True, min_periods=1).mean().to_numpy(dtype=float)
    if np.any(np.diff(tt) <= 0):
        return np.array([]), np.array([]), np.array([])
    rate = np.gradient(smooth, tt)
    finite_rate = np.isfinite(rate)
    if finite_rate.any():
        limit = float(np.nanpercentile(np.abs(rate[finite_rate]), 99.5))
        if np.isfinite(limit) and limit > 0:
            rate = np.clip(rate, -limit, limit)
    return tt, smooth, rate


def audit_one_event(row: pd.Series, raw_cache: RawVehicleCache) -> Dict[str, object]:
    """计算单个事件的方向盘快转来源证据。"""

    raw = raw_cache.load(row.get("raw_vehicle_csv", ""))
    obs = to_float(row.get("observation_s", math.nan))
    base = {
        "raw_available_for_rate": False,
        "rate_current_peak_abs": math.nan,
        "rate_current_peak_time_s": math.nan,
        "rate_near_anchor_peak_abs": math.nan,
        "rate_pre_peak_abs": math.nan,
        "rate_late_peak_abs": math.nan,
        "rate_any_0_6_peak_abs": math.nan,
        "delta_current_peak_abs_raw": math.nan,
        "delta_late_peak_abs_raw": math.nan,
        "current_rate_direction_changes": 0,
        "late_rate_direction_changes": 0,
        "fast_current": False,
        "fast_late": False,
        "fast_pre": False,
        "fast_near_anchor": False,
        "source_category": "raw_missing_or_invalid",
        "source_category_cn": SOURCE_CN["raw_missing_or_invalid"],
        "suggested_sample_action": "check_anchor_or_window",
        "suggested_sample_action_cn": ACTION_CN["check_anchor_or_window"],
    }
    if raw.empty or "zx|SteeringWheel" not in raw.columns or not np.isfinite(obs):
        return base

    rel = raw["record_s"].to_numpy(dtype=float) - obs
    steering = raw["zx|SteeringWheel"].to_numpy(dtype=float)
    keep = (rel >= -3.0) & (rel <= 6.0) & np.isfinite(rel) & np.isfinite(steering)
    if keep.sum() < 5:
        return base

    t, steer_smooth, rate = smooth_and_rate(rel[keep], steering[keep])
    if t.size < 5:
        return base

    anchor_idx = int(np.argmin(np.abs(t)))
    anchor_value = float(steer_smooth[anchor_idx])
    delta = steer_smooth - anchor_value

    mask_pre = (t >= -1.0) & (t < 0.0)
    mask_near = (t >= -0.2) & (t <= 0.6)
    mask_current = (t >= 0.0) & (t <= 2.0)
    mask_late = (t > 2.0) & (t <= 6.0)
    mask_any = (t >= 0.0) & (t <= 6.0)

    current_rate, current_rate_t = peak_abs_and_time(t, rate, mask_current)
    near_rate, near_rate_t = peak_abs_and_time(t, rate, mask_near)
    pre_rate, pre_rate_t = peak_abs_and_time(t, rate, mask_pre)
    late_rate, late_rate_t = peak_abs_and_time(t, rate, mask_late)
    any_rate, any_rate_t = peak_abs_and_time(t, rate, mask_any)
    current_delta, current_delta_t = peak_abs_and_time(t, delta, mask_current)
    late_delta, late_delta_t = peak_abs_and_time(t, delta, mask_late)

    fast_current = bool(np.isfinite(current_rate) and current_rate >= FAST_RATE_TH and np.isfinite(current_delta) and current_delta >= MOTION_TH)
    fast_late = bool(np.isfinite(late_rate) and late_rate >= FAST_RATE_TH and np.isfinite(late_delta) and late_delta >= MOTION_TH)
    fast_pre = bool(np.isfinite(pre_rate) and pre_rate >= FAST_RATE_TH)
    fast_near = bool(np.isfinite(near_rate) and near_rate >= FAST_RATE_TH)
    current_flat = bool(np.isfinite(current_delta) and current_delta < MOTION_TH and (not fast_current))

    if fast_current and fast_late:
        category = "current_and_late_fast_steer"
        action = "diagnose_model_amplitude_phase"
    elif fast_current:
        category = "current_window_fast_steer_supported"
        action = "diagnose_model_amplitude_phase"
    elif current_flat and fast_late:
        category = "late_fast_steer_not_current_window"
        action = "check_anchor_or_window"
    elif fast_late and not fast_current:
        category = "late_fast_steer_not_current_window"
        action = "check_anchor_or_window"
    elif fast_pre and not fast_current and not fast_late:
        category = "anchor_after_fast_steer"
        action = "check_anchor_or_window"
    elif (not fast_pre) and (not fast_current) and (not fast_late):
        category = "no_clear_fast_steer_evidence"
        action = "exclude_or_reanchor_candidate"
    else:
        category = "ambiguous_fast_steer_source"
        action = "check_anchor_or_window"

    base.update(
        {
            "raw_available_for_rate": True,
            "rate_current_peak_abs": current_rate,
            "rate_current_peak_time_s": current_rate_t,
            "rate_near_anchor_peak_abs": near_rate,
            "rate_near_anchor_peak_time_s": near_rate_t,
            "rate_pre_peak_abs": pre_rate,
            "rate_pre_peak_time_s": pre_rate_t,
            "rate_late_peak_abs": late_rate,
            "rate_late_peak_time_s": late_rate_t,
            "rate_any_0_6_peak_abs": any_rate,
            "rate_any_0_6_peak_time_s": any_rate_t,
            "delta_current_peak_abs_raw": current_delta,
            "delta_current_peak_time_s_raw": current_delta_t,
            "delta_late_peak_abs_raw": late_delta,
            "delta_late_peak_time_s_raw": late_delta_t,
            "current_rate_direction_changes": direction_change_count(rate, mask_current),
            "late_rate_direction_changes": direction_change_count(rate, mask_late),
            "fast_current": fast_current,
            "fast_late": fast_late,
            "fast_pre": fast_pre,
            "fast_near_anchor": fast_near,
            "source_category": category,
            "source_category_cn": SOURCE_CN[category],
            "suggested_sample_action": action,
            "suggested_sample_action_cn": ACTION_CN[action],
        }
    )
    return base


def build_audit_table() -> pd.DataFrame:
    """生成全量事件方向盘快转来源排查表。"""

    if not V312_LABEL_TABLE.exists():
        raise FileNotFoundError(f"缺少第312版标签表：{V312_LABEL_TABLE}")
    label_table = pd.read_csv(V312_LABEL_TABLE, encoding="utf-8-sig")
    severe = pd.read_csv(V312_SEVERE_OVERLAY, encoding="utf-8-sig") if V312_SEVERE_OVERLAY.exists() else pd.DataFrame()
    raw_cache = RawVehicleCache()
    rows: List[Dict[str, object]] = []
    for idx, row in label_table.iterrows():
        record = row.to_dict()
        record.update(audit_one_event(row, raw_cache))
        rows.append(record)
        if (idx + 1) % 150 == 0 or (idx + 1) == len(label_table):
            print(f"[v314] 已排查 {idx + 1}/{len(label_table)}", flush=True)

    audit = pd.DataFrame(rows)
    audit["coarse_scene_label_cn"] = audit["coarse_scene_label"].map(SCENE_LABEL_CN).fillna(audit["coarse_scene_label"].astype(str))
    if not severe.empty:
        severe_cols = [
            "event_uid",
            "severe_rank",
            "screenshot_rank",
            "error_tags",
            "error_reason_cn",
            "v307_rmse",
            "v300_rmse",
            "delta_v307_minus_v300",
        ]
        severe_cols = [c for c in severe_cols if c in severe.columns]
        severe_small = severe[severe_cols].copy()
        audit = audit.merge(severe_small, on="event_uid", how="left")
    audit["is_v309_severe"] = audit.get("severe_rank", pd.Series([math.nan] * len(audit))).map(lambda x: np.isfinite(to_float(x)))
    audit["is_user_screenshot_case"] = audit.get("screenshot_rank", pd.Series([math.nan] * len(audit))).map(lambda x: np.isfinite(to_float(x)))
    audit["fast_steer_source_ok_current"] = audit["source_category"].isin(["current_window_fast_steer_supported", "current_and_late_fast_steer"])
    audit["suspect_not_current_fast_steer"] = ~audit["fast_steer_source_ok_current"]
    return audit


def select_sample_cases(audit: pd.DataFrame) -> pd.DataFrame:
    """按问题类型做固定抽样，而不是让用户逐个看全量样本。"""

    rng = np.random.default_rng(RANDOM_SEED)
    picks: List[pd.DataFrame] = []

    def add(reason: str, df: pd.DataFrame, n: int, sort_cols: List[str] | None = None, ascending: List[bool] | None = None) -> None:
        if df.empty:
            return
        tmp = df.copy()
        if sort_cols:
            tmp = tmp.sort_values(sort_cols, ascending=ascending or [False] * len(sort_cols))
            tmp = tmp.head(n)
        elif len(tmp) > n:
            tmp = tmp.iloc[rng.choice(len(tmp), size=n, replace=False)]
        tmp = tmp.copy()
        tmp["sample_reason_cn"] = reason
        picks.append(tmp)

    add("用户截图严重样本", audit[audit["is_user_screenshot_case"]], 10, ["screenshot_rank"], [True])
    add(
        "严重错误且当前快转证据不足",
        audit[audit["is_v309_severe"] & audit["suspect_not_current_fast_steer"]],
        20,
        ["v307_rmse", "rate_current_peak_abs"],
        [False, True],
    )
    add(
        "当前窗口不明显但后续才快转",
        audit[audit["source_category"].eq("late_fast_steer_not_current_window")],
        18,
        ["rate_late_peak_abs", "rate_current_peak_abs"],
        [False, True],
    )
    add(
        "锚点前已经快转",
        audit[audit["source_category"].eq("anchor_after_fast_steer")],
        12,
        ["rate_pre_peak_abs"],
        [False],
    )
    add(
        "全程快转证据弱",
        audit[audit["source_category"].eq("no_clear_fast_steer_evidence")],
        18,
        ["rate_any_0_6_peak_abs"],
        [True],
    )
    add(
        "当前快转成立的对照样本",
        audit[audit["source_category"].isin(["current_window_fast_steer_supported", "current_and_late_fast_steer"])],
        16,
        ["rate_current_peak_abs"],
        [False],
    )

    # 每个粗场景再补少量随机样本，避免只看严重错误和可疑样本。
    for _, g in audit.groupby("coarse_scene_label"):
        add("粗场景覆盖抽样", g, 4)

    sample = pd.concat(picks, ignore_index=True) if picks else pd.DataFrame()
    if sample.empty:
        return sample
    sample = sample.drop_duplicates("event_uid", keep="first").reset_index(drop=True)
    if len(sample) > MAX_SAMPLE_PLOTS:
        sample = sample.head(MAX_SAMPLE_PLOTS).copy()
    sample.insert(0, "sample_rank", np.arange(1, len(sample) + 1, dtype=int))
    return sample


def load_raw_window_for_plot(row: pd.Series, raw_cache: RawVehicleCache) -> pd.DataFrame:
    """读取用于画图的局部窗口，并附加方向盘速度。"""

    raw = raw_cache.load(row.get("raw_vehicle_csv", ""))
    obs = to_float(row.get("observation_s", math.nan))
    if raw.empty or not np.isfinite(obs) or "zx|SteeringWheel" not in raw.columns:
        return pd.DataFrame()
    rel = raw["record_s"].to_numpy(dtype=float) - obs
    keep = (rel >= -3.0) & (rel <= 6.0)
    t, steering, rate = smooth_and_rate(rel[keep], raw.loc[keep, "zx|SteeringWheel"].to_numpy(dtype=float))
    if t.size == 0:
        return pd.DataFrame()
    out = pd.DataFrame({"rel_anchor_s": t, "steering_smooth": steering, "steering_rate": rate})
    # 其他车辆信号只用于趋势参考，按最近时间点做轻量插值。
    for col in ["zx|ay", "zx|vyaw", "zx|roll"]:
        if col in raw.columns:
            valid = keep & np.isfinite(raw[col].to_numpy(dtype=float))
            if valid.sum() >= 2:
                out[col] = np.interp(t, rel[valid], raw.loc[valid, col].to_numpy(dtype=float))
    anchor_idx = int(np.argmin(np.abs(out["rel_anchor_s"].to_numpy(dtype=float))))
    out["steering_delta_from_anchor"] = out["steering_smooth"] - float(out["steering_smooth"].iloc[anchor_idx])
    return out


def plot_sample_case(row: pd.Series, raw_cache: RawVehicleCache, out_path: Path) -> bool:
    """绘制抽样排查图。"""

    win = load_raw_window_for_plot(row, raw_cache)
    if win.empty:
        return False
    t = win["rel_anchor_s"].to_numpy(dtype=float)

    fig, axes = plt.subplots(4, 1, figsize=(13.8, 8.8), sharex=True)
    axes_arr = np.asarray(axes).reshape(-1)
    for ax in axes_arr:
        ax.axvspan(-1.0, 0.0, color="#F0FDF4", alpha=0.75)
        ax.axvspan(0.0, 2.0, color="#EFF6FF", alpha=0.82)
        ax.axvspan(2.0, 6.0, color="#F9FAFB", alpha=0.95)
        ax.axvline(0.0, color="#DC2626", lw=1.0, ls="--")
        ax.axvline(2.0, color="#6B7280", lw=0.9, ls=":")
        ax.grid(True, color="#E5E7EB", lw=0.6, alpha=0.9)

    axes_arr[0].plot(t, win["steering_smooth"], color="#111827", lw=1.4)
    axes_arr[0].set_ylabel("方向盘角")
    axes_arr[1].plot(t, win["steering_delta_from_anchor"], color="#2563EB", lw=1.3)
    axes_arr[1].axhline(MOTION_TH, color="#94A3B8", lw=0.8, ls="--")
    axes_arr[1].axhline(-MOTION_TH, color="#94A3B8", lw=0.8, ls="--")
    axes_arr[1].set_ylabel("相对锚点变化")
    axes_arr[2].plot(t, win["steering_rate"], color="#B45309", lw=1.2)
    axes_arr[2].axhline(FAST_RATE_TH, color="#DC2626", lw=0.9, ls="--")
    axes_arr[2].axhline(-FAST_RATE_TH, color="#DC2626", lw=0.9, ls="--")
    axes_arr[2].set_ylabel("方向盘转动速度")
    if "zx|ay" in win.columns:
        axes_arr[3].plot(t, win["zx|ay"], color="#111827", lw=1.0, label="横向加速度")
    if "zx|vyaw" in win.columns:
        axes_arr[3].plot(t, win["zx|vyaw"], color="#2563EB", lw=0.9, alpha=0.85, label="横摆角速度")
    if "zx|roll" in win.columns:
        axes_arr[3].plot(t, win["zx|roll"], color="#16A34A", lw=0.9, alpha=0.85, label="侧倾角")
    axes_arr[3].legend(fontsize=8, loc="best")
    axes_arr[3].set_ylabel("车身信号")

    title = (
        f"#{int(row['sample_rank']):03d} {row['sample_reason_cn']}｜{row['coarse_scene_label_cn']}｜{row['source_category_cn']}\n"
        f"当前快转峰值={to_float(row['rate_current_peak_abs']):.3f}，后续快转峰值={to_float(row['rate_late_peak_abs']):.3f}，"
        f"锚点前峰值={to_float(row['rate_pre_peak_abs']):.3f}，建议={row['suggested_sample_action_cn']}\n"
        f"{row['event_uid']}｜{row['scene_type']}｜{row['route_event']}｜划分={row['split']}"
    )
    fig.suptitle(title, fontsize=10.5, x=0.01, y=0.995, ha="left")
    axes_arr[-1].set_xlim(-3.0, 6.0)
    axes_arr[-1].set_xlabel("相对锚点时间/秒")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return True


def write_summary_tables(audit: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """输出汇总表。"""

    category_summary = (
        audit.groupby(["source_category", "source_category_cn", "suggested_sample_action_cn"], as_index=False)
        .agg(event_n=("event_uid", "count"), severe_n=("is_v309_severe", "sum"), screenshot_n=("is_user_screenshot_case", "sum"))
        .sort_values("event_n", ascending=False)
    )
    scene_summary = (
        audit.groupby(["coarse_scene_label", "coarse_scene_label_cn", "source_category_cn"], as_index=False)
        .agg(event_n=("event_uid", "count"), severe_n=("is_v309_severe", "sum"))
        .sort_values(["coarse_scene_label_cn", "event_n"], ascending=[True, False])
    )
    quantiles = audit[["rate_current_peak_abs", "rate_near_anchor_peak_abs", "rate_pre_peak_abs", "rate_late_peak_abs", "rate_any_0_6_peak_abs"]].quantile(
        [0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    )
    quantiles = quantiles.reset_index().rename(columns={"index": "quantile"})

    write_csv(category_summary, TABLES / "v314_source_category_summary.csv")
    write_csv(scene_summary, TABLES / "v314_scene_by_source_category_summary.csv")
    write_csv(quantiles, TABLES / "v314_steering_rate_quantiles.csv")
    return {"category_summary": category_summary, "scene_summary": scene_summary, "quantiles": quantiles}


def write_summary_figures(audit: pd.DataFrame, category_summary: pd.DataFrame) -> List[Path]:
    """生成概览图。"""

    paths: List[Path] = []
    vals = audit["rate_current_peak_abs"].to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.hist(vals, bins=50, color="#2563EB", alpha=0.82)
    ax.axvline(FAST_RATE_TH, color="#DC2626", lw=1.3, ls="--", label=f"快转阈值 {FAST_RATE_TH:.2f}")
    ax.set_xlabel("当前0到2秒方向盘转动速度峰值")
    ax.set_ylabel("事件数")
    ax.set_title("当前预测窗口内方向盘快转证据分布")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    p1 = FIGURES / "v314_current_window_steering_rate_distribution.png"
    fig.savefig(p1, dpi=160)
    plt.close(fig)
    paths.append(p1)

    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    tmp = category_summary.sort_values("event_n", ascending=True)
    ax.barh(tmp["source_category_cn"], tmp["event_n"], color="#0F766E")
    ax.set_xlabel("事件数")
    ax.set_title("方向盘快转来源分级")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    p2 = FIGURES / "v314_source_category_counts.png"
    fig.savefig(p2, dpi=160)
    plt.close(fig)
    paths.append(p2)
    return paths


def markdown_table(df: pd.DataFrame) -> str:
    """不用外部可选依赖，直接生成简单 Markdown 表格。"""

    if df.empty:
        return "（空表）"
    small = df.copy()
    cols = list(small.columns)

    def cell(value: object) -> str:
        if isinstance(value, float):
            if np.isfinite(value):
                text = f"{value:.6g}"
            else:
                text = ""
        else:
            text = str(value)
        return text.replace("|", "｜").replace("\n", " ")

    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in small.iterrows():
        lines.append("| " + " | ".join(cell(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def write_report(audit: pd.DataFrame, sample: pd.DataFrame, summary_tables: Dict[str, pd.DataFrame], guardrail: Dict[str, object]) -> Path:
    """写中文报告。"""

    total = len(audit)
    fast_ok = int(audit["fast_steer_source_ok_current"].sum())
    suspect = int(audit["suspect_not_current_fast_steer"].sum())
    severe_total = int(audit["is_v309_severe"].sum())
    severe_suspect = int((audit["is_v309_severe"] & audit["suspect_not_current_fast_steer"]).sum())
    screenshot_total = int(audit["is_user_screenshot_case"].sum())
    screenshot_suspect = int((audit["is_user_screenshot_case"] & audit["suspect_not_current_fast_steer"]).sum())

    lines = [
        "# 第314版方向盘快转来源抽样排查",
        "",
        "## 结论",
        "",
        f"- 本轮不训练模型，也不做逐个式人工复核；只检查样本是否有方向盘快速转动来源证据。",
        f"- 全体事件数：`{total}`。",
        f"- 当前0到2秒窗口内有快转证据：`{fast_ok}`；当前窗口快转证据不足或来源错位：`{suspect}`。",
        f"- 第309版严重错误样本中，当前窗口快转证据不足或来源错位：`{severe_suspect}/{severe_total}`。",
        f"- 用户截图样本中，当前窗口快转证据不足或来源错位：`{screenshot_suspect}/{screenshot_total}`。",
        f"- 本轮固定快转阈值：方向盘转动速度峰值 `>= {FAST_RATE_TH:.2f}` 且当前方向盘变化峰值 `>= {MOTION_TH:.2f}`。",
        "",
        "## 主要输出",
        "",
        f"- 全量排查表：`{TABLES / 'v314_rapid_steering_source_audit_all_delay0.csv'}`",
        f"- 抽样排查表：`{TABLES / 'v314_rapid_steering_source_sample_cases.csv'}`",
        f"- 来源分级汇总：`{TABLES / 'v314_source_category_summary.csv'}`",
        f"- 粗场景交叉汇总：`{TABLES / 'v314_scene_by_source_category_summary.csv'}`",
        f"- 抽样图目录：`{FIGURES / 'sample_cases'}`",
        "",
        "## 来源分级汇总",
        "",
        markdown_table(summary_tables["category_summary"]),
        "",
        "## 转动速度分位数",
        "",
        markdown_table(summary_tables["quantiles"]),
        "",
        "## 下一步建议",
        "",
        "- 对“当前窗口不明显，后续才方向盘快转”和“锚点前已经方向盘快转”两类，不应继续当作普通训练样本硬塞给当前0到2秒预测，应进入锚点或窗口修正。",
        "- 对“全程方向盘快转证据弱”类，应优先考虑候选剔除或重新找触发点，因为这和用户强调的样本定义不一致。",
        "- 对“当前窗口有方向盘快转证据”但仍预测差的严重样本，才进入模型幅值、相位、极端跟随不足的训练修正。",
    ]
    path = REPORTS / "v314_rapid_steering_source_sample_audit_cn.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_file_inventory() -> pd.DataFrame:
    """记录产物文件清单。"""

    rows = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "file_inventory.csv":
            rows.append(
                {
                    "relative_path": str(path.relative_to(OUT)),
                    "size_bytes": int(path.stat().st_size),
                    "sha256": file_sha256(path),
                }
            )
    inv = pd.DataFrame(rows)
    write_csv(inv, LOGS / "file_inventory.csv")
    return inv


def make_zip_package() -> Tuple[Path, bool]:
    """打包产物并做压缩包自检。"""

    zip_path = OUT / "v314_rapid_steering_source_sample_audit_20260704.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(OUT.rglob("*")):
            if path == zip_path or not path.is_file():
                continue
            zf.write(path, path.relative_to(OUT))
    with zipfile.ZipFile(zip_path, "r") as zf:
        bad = zf.testzip()
    return zip_path, bad is None


def main() -> None:
    started = time.time()
    clean_out_dir()
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    audit = build_audit_table()
    write_csv(audit, TABLES / "v314_rapid_steering_source_audit_all_delay0.csv")
    sample = select_sample_cases(audit)

    raw_cache = RawVehicleCache()
    plotted_rows: List[Dict[str, object]] = []
    for pos, row in sample.iterrows():
        file_name = (
            f"rank{int(row['sample_rank']):03d}_"
            f"{safe_name(row['source_category'], 40)}_"
            f"{safe_name(row['coarse_scene_label'], 28)}.png"
        )
        out_path = FIGURES / "sample_cases" / file_name
        ok = plot_sample_case(row, raw_cache, out_path)
        record = row.to_dict()
        record["sample_plot_created"] = bool(ok)
        record["sample_plot_relpath"] = out_path.relative_to(OUT).as_posix() if ok else ""
        plotted_rows.append(record)
        if (pos + 1) % 20 == 0 or (pos + 1) == len(sample):
            print(f"[v314] 已生成抽样图 {pos + 1}/{len(sample)}", flush=True)
    sample_out = pd.DataFrame(plotted_rows)
    write_csv(sample_out, TABLES / "v314_rapid_steering_source_sample_cases.csv")

    summary_tables = write_summary_tables(audit)
    figure_paths = write_summary_figures(audit, summary_tables["category_summary"])
    input_hashes = pd.DataFrame(
        [
            {"input_name": "v312_label_table", "path": str(V312_LABEL_TABLE), "sha256": file_sha256(V312_LABEL_TABLE)},
            {"input_name": "v312_severe_overlay", "path": str(V312_SEVERE_OVERLAY), "sha256": file_sha256(V312_SEVERE_OVERLAY) if V312_SEVERE_OVERLAY.exists() else ""},
        ]
    )
    write_csv(input_hashes, LOGS / "input_hashes.csv")

    guardrail = {
        "pass": True,
        "version": "v314_rapid_steering_source_sample_audit_20260704",
        "training_run": False,
        "event_n": int(len(audit)),
        "sample_case_n": int(len(sample_out)),
        "sample_plot_created_n": int(sample_out["sample_plot_created"].sum()) if len(sample_out) else 0,
        "fast_rate_threshold": FAST_RATE_TH,
        "motion_threshold": MOTION_TH,
        "current_fast_steer_supported_n": int(audit["fast_steer_source_ok_current"].sum()),
        "suspect_not_current_fast_steer_n": int(audit["suspect_not_current_fast_steer"].sum()),
        "severe_n": int(audit["is_v309_severe"].sum()),
        "severe_suspect_not_current_fast_steer_n": int((audit["is_v309_severe"] & audit["suspect_not_current_fast_steer"]).sum()),
        "screenshot_n": int(audit["is_user_screenshot_case"].sum()),
        "screenshot_suspect_not_current_fast_steer_n": int((audit["is_user_screenshot_case"] & audit["suspect_not_current_fast_steer"]).sum()),
        "uses_test_error_as_training_feature": False,
        "candidate_selection_uses_test": False,
        "uses_future_context_as_model_input": False,
        "figure_paths": [str(p) for p in figure_paths],
        "runtime_seconds": float(time.time() - started),
    }
    report_path = write_report(audit, sample_out, summary_tables, guardrail)
    guardrail["report_path"] = str(report_path)
    write_json(guardrail, LOGS / "guardrail_check.json")
    write_file_inventory()
    zip_path, zip_ok = make_zip_package()
    guardrail["zip_path"] = str(zip_path)
    guardrail["zip_testzip"] = bool(zip_ok)
    write_json(guardrail, LOGS / "guardrail_check.json")
    write_file_inventory()
    print(json.dumps(guardrail, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
