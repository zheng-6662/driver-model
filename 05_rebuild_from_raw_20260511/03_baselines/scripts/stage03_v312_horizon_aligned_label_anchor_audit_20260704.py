#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v312 horizon-aligned label / anchor audit.

目的：
- 将“模型 0-2s 预测窗口内真正要预测的局部动作”和“2-6s 后续事件上下文”拆开；
- 解释 v309/v311 severe 错例中哪些是后续事件标签驱动了当前窗口预测；
- 产出可人工复核、可作为下一轮标签修正输入的表，而不是直接训练模型。

边界：
- 本脚本不训练模型；
- `local_0_2_motion_label` 由真实 0-2s 目标曲线计算，只能用于审计、人工标签定义和训练集监督实验，不能当作部署时可见输入；
- `late_2_6_context_label` 来自 raw 车辆后续窗口，只能作为后续上下文/锚点审计，不可直接作为原锚点预测输入。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import shutil
import sys
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
SCRIPTS = BASELINES / "scripts"
V307_SCRIPT = SCRIPTS / "stage03_v307_coarse_scene_label_conditioned_curve_model_20260704.py"
V309_SEVERE = (
    BASELINES
    / "v309_recent_best_prediction_effect_gallery_20260704"
    / "tables"
    / "v309_severe_direction_or_intent_errors.csv"
)
V311_AUDIT = (
    BASELINES
    / "v311_severe_anchor_horizon_misalignment_audit_20260704"
    / "tables"
    / "v311_severe_anchor_horizon_misalignment_audit.csv"
)

OUT = BASELINES / "v312_horizon_aligned_label_anchor_audit_20260704"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"


LOCAL_FLAT_TH = 0.35
LOCAL_STRONG_TH = 1.00
LOCAL_EXTREME_TH = 2.00
LATE_LARGE_TH = 1.00


def import_module_from_path(module_name: str, path: Path):
    """按路径导入前序脚本，复用已审计的数据构造。"""

    if not path.exists():
        raise FileNotFoundError(f"缺少依赖脚本：{path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法导入脚本：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V307 = import_module_from_path("stage03_v307_for_v312_horizon_labels", V307_SCRIPT)
V304 = V307.V304
FUTURE_GRID = V307.FUTURE_GRID.astype(np.float32)


def ensure_dirs() -> None:
    """创建输出目录。"""

    for folder in (TABLES, FIGURES, REPORTS, LOGS):
        folder.mkdir(parents=True, exist_ok=True)


def clean_out_dir() -> None:
    """只清理 v312 自己的输出。"""

    if OUT.exists():
        shutil.rmtree(OUT)
    ensure_dirs()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """CSV 使用 utf-8-sig，方便 Windows/Excel 查看中文。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_json(payload: Dict[str, object], path: Path) -> None:
    """保存 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def file_sha256(path: Path) -> str:
    """计算文件哈希。"""

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sign_label(value: float, eps: float = 0.05) -> str:
    """把连续峰值转为方向标签。"""

    if not np.isfinite(value):
        return "NA"
    if value > eps:
        return "positive"
    if value < -eps:
        return "negative"
    return "flat"


def signed_peak(values: np.ndarray) -> Tuple[float, float, int]:
    """返回绝对值最大峰值、绝对峰值和局部索引。"""

    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return math.nan, math.nan, -1
    idx_all = np.where(finite)[0]
    valid = arr[finite]
    idx_local = int(np.argmax(np.abs(valid)))
    idx = int(idx_all[idx_local])
    peak = float(arr[idx])
    return peak, float(abs(peak)), idx


def reversal_count(values: np.ndarray, min_step: float = 0.06) -> int:
    """粗略统计曲线一阶差分的有效反转次数。"""

    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if finite.sum() < 3:
        return 0
    diff = np.diff(arr[finite])
    diff[np.abs(diff) < min_step] = 0.0
    signs = np.sign(diff)
    signs = signs[signs != 0]
    if signs.size < 2:
        return 0
    return int(np.sum(signs[1:] != signs[:-1]))


def motion_label_from_peak(prefix: str, peak: float, peak_abs: float) -> Tuple[str, str, str]:
    """把峰值幅值和方向转成离散动作标签。"""

    direction = sign_label(peak)
    if not np.isfinite(peak_abs) or direction == "NA":
        return f"{prefix}_unknown", f"{prefix}_unknown", direction
    if peak_abs < LOCAL_FLAT_TH:
        return f"{prefix}_flat_hold", f"{prefix}_flat", "flat"
    if peak_abs < LOCAL_STRONG_TH:
        return f"{prefix}_mild_{direction}", f"{prefix}_{direction}", direction
    if peak_abs < LOCAL_EXTREME_TH:
        return f"{prefix}_strong_{direction}", f"{prefix}_{direction}", direction
    return f"{prefix}_extreme_{direction}", f"{prefix}_{direction}", direction


def local_features(curve: np.ndarray, valid: np.ndarray) -> Dict[str, object]:
    """从 0-2s 真实目标曲线构造 horizon-aligned 局部标签。"""

    values = np.asarray(curve, dtype=float)[valid]
    t = FUTURE_GRID[valid]
    peak, peak_abs, idx = signed_peak(values)
    peak_t = float(t[idx]) if idx >= 0 and idx < len(t) else math.nan
    end_delta = float(values[-1]) if values.size else math.nan
    min_delta = float(np.nanmin(values)) if values.size else math.nan
    max_delta = float(np.nanmax(values)) if values.size else math.nan
    crosses_zero = bool(np.isfinite(min_delta) and np.isfinite(max_delta) and min_delta < -0.20 and max_delta > 0.20)
    rev_count = reversal_count(values)
    label, family, direction = motion_label_from_peak("local_0_2", peak, peak_abs)
    if crosses_zero and peak_abs >= LOCAL_FLAT_TH:
        shape = "local_zero_cross_transition"
    elif rev_count >= 2 and peak_abs >= LOCAL_FLAT_TH:
        shape = "local_multi_correction"
    elif peak_abs < LOCAL_FLAT_TH:
        shape = "local_flat"
    else:
        shape = "local_single_direction"
    return {
        "local_0_2_peak": peak,
        "local_0_2_peak_abs": peak_abs,
        "local_0_2_peak_time_s": peak_t,
        "local_0_2_end_delta": end_delta,
        "local_0_2_min_delta": min_delta,
        "local_0_2_max_delta": max_delta,
        "local_0_2_direction": direction,
        "local_0_2_motion_label": label,
        "local_0_2_motion_family": family,
        "local_0_2_shape_label": shape,
        "local_0_2_zero_cross": crosses_zero,
        "local_0_2_reversal_count": rev_count,
    }


class RawVehicleCache:
    """按 raw CSV 路径缓存车辆数据，避免重复读取同一记录。"""

    def __init__(self) -> None:
        self.cache: Dict[str, pd.DataFrame] = {}

    def load(self, raw_path: str) -> pd.DataFrame:
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


def late_features(raw_cache: RawVehicleCache, raw_path: str, observation_s: float) -> Dict[str, object]:
    """从 raw 车辆数据计算 2-6s 后续上下文标签。"""

    raw = raw_cache.load(raw_path)
    if raw.empty or "zx|SteeringWheel" not in raw.columns or not np.isfinite(float(observation_s)):
        label, family, direction = "late_2_6_unknown", "late_2_6_unknown", "NA"
        return {
            "raw_available": False,
            "late_2_6_peak": math.nan,
            "late_2_6_peak_abs": math.nan,
            "late_2_6_peak_time_s": math.nan,
            "late_2_6_direction": direction,
            "late_2_6_context_label": label,
            "late_2_6_context_family": family,
            "late_2_6_ay_peak_abs": math.nan,
            "late_2_6_yaw_peak_abs": math.nan,
            "late_2_6_roll_peak_abs": math.nan,
        }
    rel = raw["record_s"].to_numpy(dtype=float) - float(observation_s)
    steer = raw["zx|SteeringWheel"].to_numpy(dtype=float)
    finite = np.isfinite(rel) & np.isfinite(steer)
    if not finite.any():
        label, family, direction = "late_2_6_unknown", "late_2_6_unknown", "NA"
        return {
            "raw_available": False,
            "late_2_6_peak": math.nan,
            "late_2_6_peak_abs": math.nan,
            "late_2_6_peak_time_s": math.nan,
            "late_2_6_direction": direction,
            "late_2_6_context_label": label,
            "late_2_6_context_family": family,
            "late_2_6_ay_peak_abs": math.nan,
            "late_2_6_yaw_peak_abs": math.nan,
            "late_2_6_roll_peak_abs": math.nan,
        }

    anchor_idx = int(np.argmin(np.abs(rel[finite])))
    finite_idx = np.where(finite)[0]
    anchor_value = float(steer[finite_idx[anchor_idx]])
    steer_delta = steer - anchor_value
    win = (rel > 2.0) & (rel <= 6.0)
    peak, peak_abs, idx = signed_peak(steer_delta[win])
    rel_win = rel[win]
    peak_t = float(rel_win[idx]) if idx >= 0 and idx < rel_win.size else math.nan
    label, family, direction = motion_label_from_peak("late_2_6", peak, peak_abs)

    def abs_peak_col(col: str) -> float:
        if col not in raw.columns or not win.any():
            return math.nan
        vals = raw.loc[win, col].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return math.nan
        return float(np.max(np.abs(vals)))

    return {
        "raw_available": True,
        "late_2_6_peak": peak,
        "late_2_6_peak_abs": peak_abs,
        "late_2_6_peak_time_s": peak_t,
        "late_2_6_direction": direction,
        "late_2_6_context_label": label,
        "late_2_6_context_family": family,
        "late_2_6_ay_peak_abs": abs_peak_col("zx|ay"),
        "late_2_6_yaw_peak_abs": abs_peak_col("zx|vyaw"),
        "late_2_6_roll_peak_abs": abs_peak_col("zx|roll"),
    }


def alignment_features(row: pd.Series) -> Dict[str, object]:
    """比较 local 0-2s 与 late 2-6s 标签是否一致。"""

    local_abs = float(row.get("local_0_2_peak_abs", math.nan))
    late_abs = float(row.get("late_2_6_peak_abs", math.nan))
    local_dir = str(row.get("local_0_2_direction", "NA"))
    late_dir = str(row.get("late_2_6_direction", "NA"))
    coarse = str(row.get("coarse_scene_label", ""))

    local_flat_late_large = bool(np.isfinite(local_abs) and np.isfinite(late_abs) and local_abs < LOCAL_FLAT_TH and late_abs >= LATE_LARGE_TH)
    late_dominant = bool(
        np.isfinite(local_abs)
        and np.isfinite(late_abs)
        and late_abs >= max(LATE_LARGE_TH, 1.50 * max(local_abs, 0.10))
    )
    local_dominant = bool(
        np.isfinite(local_abs)
        and np.isfinite(late_abs)
        and local_abs >= max(LATE_LARGE_TH, 1.50 * max(late_abs, 0.10))
    )
    direction_conflict = bool(
        np.isfinite(local_abs)
        and np.isfinite(late_abs)
        and local_abs >= LOCAL_FLAT_TH
        and late_abs >= 0.60
        and local_dir not in {"flat", "NA"}
        and late_dir not in {"flat", "NA"}
        and local_dir != late_dir
    )
    coarse_horizon_mismatch = bool(
        (coarse in {"continuous_lane_change", "emergency_lane_change_instability"} and local_flat_late_large)
        or (coarse in {"curve_flat", "curve_downhill"} and direction_conflict)
        or late_dominant
    )
    if local_flat_late_large:
        alignment_label = "current_flat_late_event"
    elif direction_conflict:
        alignment_label = "current_late_direction_conflict"
    elif late_dominant:
        alignment_label = "late_dominant_context"
    elif local_dominant:
        alignment_label = "local_dominant_current"
    elif local_abs < LOCAL_FLAT_TH and late_abs < LOCAL_FLAT_TH:
        alignment_label = "stable_flat"
    else:
        alignment_label = "roughly_aligned"

    if local_flat_late_large:
        recommended_action = "split_local_flat_and_late_context"
    elif direction_conflict:
        recommended_action = "split_current_and_late_direction"
    elif late_dominant:
        recommended_action = "keep_late_context_separate"
    elif local_dominant:
        recommended_action = "local_label_can_drive_current_window"
    else:
        recommended_action = "no_anchor_change_needed"

    return {
        "local_flat_late_large": local_flat_late_large,
        "late_dominant_context": late_dominant,
        "local_dominant_current": local_dominant,
        "local_late_direction_conflict": direction_conflict,
        "coarse_label_horizon_mismatch": coarse_horizon_mismatch,
        "horizon_alignment_label": alignment_label,
        "recommended_label_action": recommended_action,
        "late_over_local_abs_ratio": float(late_abs / max(local_abs, 0.10)) if np.isfinite(local_abs) and np.isfinite(late_abs) else math.nan,
    }


def build_delay0_label_table(prepared) -> pd.DataFrame:
    """生成 delay0 事件级 horizon-aligned 标签表。"""

    manifest = prepared.data.manifest.reset_index(drop=True)
    delay0_idx = manifest.index[manifest["delay_ms"].astype(int).eq(0)].to_numpy(dtype=int)
    y_curve = prepared.data.y_future[:, :, 0].astype(np.float32)
    valid_all = prepared.prepared.point_data.valid_original_remaining_all.reshape(len(manifest), len(FUTURE_GRID)).astype(bool)
    raw_cache = RawVehicleCache()

    rows: List[Dict[str, object]] = []
    for idx in delay0_idx:
        row = manifest.iloc[idx]
        valid = valid_all[idx]
        base = {
            "array_row": int(idx),
            "event_uid": str(row["event_uid"]),
            "subject": str(row.get("subject", "")),
            "recording": str(row.get("recording", "")),
            "split": str(row.get("split", "")),
            "delay_ms": int(row.get("delay_ms", 0)),
            "scene_type": str(row.get("scene_type", "")),
            "route_event": str(row.get("route_event", "")),
            "observation_s": float(row.get("observation_s", math.nan)),
            "raw_vehicle_csv": str(row.get("raw_vehicle_csv", "")),
            "coarse_scene_label": str(prepared.event_label_name[idx]),
            "strong_steer": bool(row.get("strong_steer", False)),
            "vehicle_strong": bool(row.get("vehicle_strong", False)),
            "zero_cross": bool(row.get("zero_cross", False)),
            "extreme_peak": bool(row.get("extreme_peak", False)),
            "within_bad_top10_by_v249": int(row.get("within_bad_top10_by_v249", 0)),
            "within_bad_top20_by_v249": int(row.get("within_bad_top20_by_v249", 0)),
        }
        local = local_features(y_curve[idx], valid)
        late = late_features(raw_cache, base["raw_vehicle_csv"], base["observation_s"])
        base.update(local)
        base.update(late)
        base.update(alignment_features(pd.Series(base)))
        rows.append(base)
    return pd.DataFrame(rows)


def build_summary_tables(label_table: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """构造 split、粗标签、对齐状态分布。"""

    split_summary = (
        label_table.groupby(["split", "local_0_2_motion_label"], as_index=False)
        .agg(event_n=("event_uid", "nunique"))
        .sort_values(["split", "event_n"], ascending=[True, False])
    )
    coarse_summary = (
        label_table.groupby(["coarse_scene_label", "local_0_2_motion_label", "late_2_6_context_label"], as_index=False)
        .agg(
            event_n=("event_uid", "nunique"),
            mismatch_n=("coarse_label_horizon_mismatch", "sum"),
            local_flat_late_large_n=("local_flat_late_large", "sum"),
            direction_conflict_n=("local_late_direction_conflict", "sum"),
        )
        .sort_values(["coarse_scene_label", "event_n"], ascending=[True, False])
    )
    alignment_summary = (
        label_table.groupby(["split", "horizon_alignment_label"], as_index=False)
        .agg(
            event_n=("event_uid", "nunique"),
            severe_bad10_n=("within_bad_top10_by_v249", "sum"),
            coarse_mismatch_n=("coarse_label_horizon_mismatch", "sum"),
        )
        .sort_values(["split", "event_n"], ascending=[True, False])
    )
    return split_summary, coarse_summary, alignment_summary


def build_severe_overlay(label_table: pd.DataFrame) -> pd.DataFrame:
    """把 horizon-aligned 标签覆盖到 v309 severe 集合。"""

    if not V309_SEVERE.exists():
        return pd.DataFrame()
    severe = pd.read_csv(V309_SEVERE, encoding="utf-8-sig")
    overlay = severe.merge(label_table, on="event_uid", how="left", suffixes=("_severe", ""))
    if V311_AUDIT.exists():
        v311 = pd.read_csv(V311_AUDIT, encoding="utf-8-sig")
        keep = [
            "event_uid",
            "misalignment_tags",
            "predicts_future_too_early",
            "label_horizon_mismatch_suspected",
        ]
        keep = [c for c in keep if c in v311.columns]
        overlay = overlay.merge(v311[keep], on="event_uid", how="left", suffixes=("", "_v311"))
    return overlay


def plot_label_distributions(label_table: pd.DataFrame, alignment_summary: pd.DataFrame) -> List[Path]:
    """生成分布图。"""

    paths: List[Path] = []
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    top_labels = label_table["local_0_2_motion_label"].value_counts().head(12)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(np.arange(len(top_labels)), top_labels.values, color="#2563EB")
    ax.set_xticks(np.arange(len(top_labels)))
    ax.set_xticklabels(top_labels.index.tolist(), rotation=25, ha="right")
    ax.set_ylabel("event count")
    ax.set_title("v312 local 0-2s motion label distribution")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path1 = FIGURES / "v312_local_0_2_label_distribution.png"
    fig.savefig(path1, dpi=170)
    plt.close(fig)
    paths.append(path1)

    pivot = alignment_summary.pivot(index="horizon_alignment_label", columns="split", values="event_n").fillna(0)
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    bottom = np.zeros(len(pivot))
    colors = {"train": "#22C55E", "val": "#F97316", "test": "#2563EB"}
    for split in [c for c in ["train", "val", "test"] if c in pivot.columns]:
        vals = pivot[split].to_numpy(dtype=float)
        ax.bar(np.arange(len(pivot)), vals, bottom=bottom, label=split, color=colors.get(split))
        bottom += vals
    ax.set_xticks(np.arange(len(pivot)))
    ax.set_xticklabels(pivot.index.tolist(), rotation=25, ha="right")
    ax.set_ylabel("event count")
    ax.set_title("v312 horizon alignment labels by split")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path2 = FIGURES / "v312_horizon_alignment_by_split.png"
    fig.savefig(path2, dpi=170)
    plt.close(fig)
    paths.append(path2)
    return paths


def write_report(
    label_table: pd.DataFrame,
    severe_overlay: pd.DataFrame,
    split_summary: pd.DataFrame,
    coarse_summary: pd.DataFrame,
    alignment_summary: pd.DataFrame,
    guardrail: Dict[str, object],
) -> Path:
    """写中文报告。"""

    path = REPORTS / "v312_horizon_aligned_label_anchor_audit_cn.md"
    severe_cols = [
        "severe_rank",
        "screenshot_rank",
        "event_uid",
        "coarse_scene_label_cn",
        "error_tags",
        "local_0_2_motion_label",
        "late_2_6_context_label",
        "horizon_alignment_label",
        "recommended_label_action",
        "late_over_local_abs_ratio",
    ]
    severe_cols = [c for c in severe_cols if c in severe_overlay.columns]
    severe_top = severe_overlay.sort_values(
        ["coarse_label_horizon_mismatch", "late_over_local_abs_ratio"],
        ascending=[False, False],
    ).head(20)
    top_coarse = coarse_summary.head(25)
    lines = [
        "# v312 horizon-aligned label / anchor audit",
        "",
        "## 这一步做了什么",
        "",
        "v312 把事件标签拆成两层：",
        "",
        "- `local_0_2_motion_label`：模型当前 0-2s 预测窗口内真实要预测的局部动作。",
        "- `late_2_6_context_label`：2-6s 后续真实车辆动作，只作为后续上下文和锚点审计。",
        "",
        "这一步不训练模型，目的是为下一轮 confirmed 标签和模型输入边界做准备。",
        "",
        "## 总体结果",
        "",
        f"- delay0 事件数：`{len(label_table)}`",
        f"- coarse label 与 horizon 局部窗口存在错位嫌疑：`{int(label_table['coarse_label_horizon_mismatch'].sum())}`",
        f"- local flat 但 late 2-6s 出现大动作：`{int(label_table['local_flat_late_large'].sum())}`",
        f"- local 与 late 方向冲突：`{int(label_table['local_late_direction_conflict'].sum())}`",
        f"- v309 severe overlay 事件数：`{len(severe_overlay)}`",
        f"- severe 中错位嫌疑：`{int(severe_overlay['coarse_label_horizon_mismatch'].sum()) if len(severe_overlay) else 0}`",
        "",
        "## 按 split 的 horizon alignment 分布",
        "",
        alignment_summary.to_markdown(index=False),
        "",
        "## severe 复核优先 Top 20",
        "",
        severe_top[severe_cols].to_markdown(index=False) if len(severe_top) else "NA",
        "",
        "## 粗标签与 local/late 标签组合 Top 25",
        "",
        top_coarse.to_markdown(index=False),
        "",
        "## 当前判断",
        "",
        "- 下一轮不应把 `late_2_6_context_label` 当作 0-2s 预测输入。",
        "- 对 `current_flat_late_event`，0-2s 训练应以 flat/hold 为局部标签，late event 只作为后续上下文。",
        "- 对 `current_late_direction_conflict`，必须把当前方向和后续方向分开，否则模型会学到反向。",
        "- 可以优先让用户人工复核 severe overlay 表中的 `split_local_flat_and_late_context` 和 `split_current_and_late_direction`。",
        "",
        "## guardrail",
        "",
        "```json",
        json.dumps(guardrail, ensure_ascii=False, indent=2),
        "```",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_file_inventory() -> pd.DataFrame:
    """记录输出文件清单。"""

    rows = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            rows.append({"relative_path": str(path.relative_to(OUT)), "size_bytes": path.stat().st_size, "sha256": file_sha256(path)})
    inv = pd.DataFrame(rows)
    write_csv(inv, LOGS / "file_inventory.csv")
    return inv


def make_zip_package() -> Tuple[Path, bool]:
    """打包并校验产物。"""

    zip_path = OUT / "v312_horizon_aligned_label_anchor_audit_20260704.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(OUT.rglob("*")):
            if path == zip_path or not path.is_file():
                continue
            zf.write(path, path.relative_to(OUT))
    with zipfile.ZipFile(zip_path, "r") as zf:
        ok = zf.testzip() is None
    return zip_path, ok


def main() -> None:
    started = time.time()
    clean_out_dir()
    prepared = V307.prepare_v307_data(hard_event_extra=0.0)
    label_table = build_delay0_label_table(prepared)
    write_csv(label_table, TABLES / "v312_horizon_aligned_delay0_event_labels.csv")

    split_summary, coarse_summary, alignment_summary = build_summary_tables(label_table)
    write_csv(split_summary, TABLES / "v312_local_label_distribution_by_split.csv")
    write_csv(coarse_summary, TABLES / "v312_coarse_local_late_crosstab.csv")
    write_csv(alignment_summary, TABLES / "v312_horizon_alignment_summary_by_split.csv")

    severe_overlay = build_severe_overlay(label_table)
    write_csv(severe_overlay, TABLES / "v312_v309_severe_horizon_label_overlay.csv")

    figure_paths = plot_label_distributions(label_table, alignment_summary)
    input_hashes = pd.DataFrame(
        [
            {"input_name": "v307_script_reused", "path": str(V307_SCRIPT), "sha256": file_sha256(V307_SCRIPT)},
            {"input_name": "v309_severe_table", "path": str(V309_SEVERE), "sha256": file_sha256(V309_SEVERE) if V309_SEVERE.exists() else ""},
            {"input_name": "v311_audit_table", "path": str(V311_AUDIT), "sha256": file_sha256(V311_AUDIT) if V311_AUDIT.exists() else ""},
        ]
    )
    write_csv(input_hashes, LOGS / "input_hashes.csv")

    guardrail = {
        "pass": True,
        "version": "v312_horizon_aligned_label_anchor_audit_20260704",
        "training_run": False,
        "event_n": int(len(label_table)),
        "local_0_2_label_source": "true_target_curve_0_2s_diagnostic_not_deployable_input",
        "late_2_6_context_source": "raw_vehicle_future_2_6s_diagnostic_not_original_anchor_input",
        "uses_test_error_as_training_feature": False,
        "candidate_selection_uses_test": False,
        "deployable_without_manual_or_preanchor_label": False,
        "coarse_label_horizon_mismatch_n": int(label_table["coarse_label_horizon_mismatch"].sum()),
        "local_flat_late_large_n": int(label_table["local_flat_late_large"].sum()),
        "local_late_direction_conflict_n": int(label_table["local_late_direction_conflict"].sum()),
        "severe_overlay_n": int(len(severe_overlay)),
        "severe_coarse_label_horizon_mismatch_n": int(severe_overlay["coarse_label_horizon_mismatch"].sum()) if len(severe_overlay) else 0,
        "figure_paths": [str(p) for p in figure_paths],
        "runtime_seconds": float(time.time() - started),
    }
    report_path = write_report(label_table, severe_overlay, split_summary, coarse_summary, alignment_summary, guardrail)
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
