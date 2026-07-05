"""
v232 过晚锚点重锚定候选包。

目标：
1. 读取 v230 casebook 和 v231 人工反馈，把疑似晚锚点样本系统化转成“候选新锚点”。
2. 只生成证据表、审核表和图，不直接修改训练标签，不训练模型。
3. 明确避开两条已经尝试且不足的路线：硬响应类型分类前置、简单多候选轨迹输出。

输出：
- tables/v232_reanchor_candidate_all_scored.csv
- tables/v232_reanchor_candidate_review_table.csv
- tables/v232_reanchor_grid_0p05s.csv
- figures/*.png
- reports/v232_late_anchor_reanchor_candidates_cn.md
"""

from __future__ import annotations

import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(r"F:\data_set_process\data_process")
REBUILD = ROOT / "05_rebuild_from_raw_20260511"
BASELINES = REBUILD / "03_baselines"
RAW_ROOT = ROOT / "01_datasets" / "数据预处理" / "原始车辆数据"

V225 = BASELINES / "v225_formal_route_reconstruction_evidence_pack_20260622"
V230 = BASELINES / "v230_failure_case_manual_review_casebook_20260623"
V231 = BASELINES / "v231_worst_case_anchor_context_20260624"
OUT = BASELINES / "v232_late_anchor_reanchor_candidates_20260624"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"

for folder in (TABLES, FIGURES, REPORTS, LOGS):
    folder.mkdir(parents=True, exist_ok=True)

EVAL_PATH = V225 / "tables" / "per_sample_formal_reconstruction_eval.csv"
CASEBOOK_PATH = V230 / "tables" / "v230_failure_casebook_table.csv"
V231_META_PATH = V231 / "tables" / "v231_anchor_metadata.csv"
V231_OVERRIDE_PATH = V231 / "tables" / "v231_user_feedback_overrides.csv"

# 原始车辆信号候选列。不同被试和不同导出版本字段不完全一致，所以后续按存在性读取。
RAW_CANDIDATE_COLS = [
    "ID",
    "StorageTime",
    "zx|SteeringWheel",
    "zx1|v_km/h",
    "zx|v_km/h",
    "zx|vx",
    "zx|vy",
    "zx|ax",
    "zx|ay",
    "zx|vyaw",
    "zx|ayaw",
    "zx|yaw",
    "zx|roll",
    "zx|pitch",
    "zx|aroll",
    "zx|apitch",
    "zx1|lanecurvatureXY",
    "zx|lanecurvatureXY",
    "zx1|lateraldistance",
    "zx|lateraldistance",
    "zx1|mu",
    "zx|mu",
    "zx|AcceleratorPedal",
    "zx|BrakePedal",
    "zx|x",
    "zx|y",
    "zx|z",
    "zx1|distance7",
    "zx1|distance8",
    "zx1|pointdistance",
    "zx1|pointdistance9",
]

GRID_STEP_S = 0.05
GRID_START_S = -10.0
GRID_END_S = 8.0
GRID = np.round(np.arange(GRID_START_S, GRID_END_S + 1e-9, GRID_STEP_S), 4)

KEY_OFFSETS_OLD = [-8, -5, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 5, 8]
KEY_OFFSETS_CANDIDATE = [-2, -1, -0.5, 0, 0.5, 1, 2, 3, 5]

MANUAL_CONFIRMED_LATE = {
    "rjy_Entity_Recording_2025_09_28_20_02_20_v108_010",
}

mpl.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
mpl.rcParams["axes.unicode_minus"] = False


@dataclass
class RawVehicle:
    df: pd.DataFrame
    start_time: pd.Timestamp
    usecols: list[str]
    encoding: str


def read_csv_header(path: Path) -> tuple[list[str], str]:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            header = pd.read_csv(path, nrows=0, encoding=encoding).columns.tolist()
            return header, encoding
        except Exception as exc:  # pragma: no cover - 用于兼容原始文件编码。
            last_error = exc
    raise RuntimeError(f"无法读取表头：{path} / {last_error}")


def read_raw_vehicle(path: Path) -> RawVehicle:
    header, encoding = read_csv_header(path)
    usecols = [col for col in RAW_CANDIDATE_COLS if col in header]
    if "StorageTime" not in usecols or "zx|SteeringWheel" not in usecols:
        raise RuntimeError(f"原始车辆文件缺少 StorageTime 或 zx|SteeringWheel：{path}")

    df = pd.read_csv(path, encoding=encoding, usecols=usecols, low_memory=False)
    df["StorageTime_dt"] = pd.to_datetime(df["StorageTime"], errors="coerce")
    df = df[df["StorageTime_dt"].notna()].copy()
    df = df.sort_values("StorageTime_dt").reset_index(drop=True)
    if df.empty:
        raise RuntimeError(f"StorageTime 解析后为空：{path}")

    start_time = df["StorageTime_dt"].iloc[0]
    df["t_rel_record_s"] = (df["StorageTime_dt"] - start_time).dt.total_seconds()
    for col in usecols:
        if col not in ("ID", "StorageTime"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return RawVehicle(df=df, start_time=start_time, usecols=usecols, encoding=encoding)


def nearest_values(times: np.ndarray, values: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """对一个信号取每个目标时刻最近的非空采样值，并返回时间误差(ms)。"""
    valid = np.isfinite(times) & np.isfinite(values)
    if not np.any(valid):
        return np.full(len(targets), np.nan), np.full(len(targets), np.nan)

    vt = times[valid]
    vv = values[valid]
    order = np.argsort(vt)
    vt = vt[order]
    vv = vv[order]

    pos = np.searchsorted(vt, targets)
    left = np.clip(pos - 1, 0, len(vt) - 1)
    right = np.clip(pos, 0, len(vt) - 1)
    choose_right = np.abs(vt[right] - targets) < np.abs(vt[left] - targets)
    idx = np.where(choose_right, right, left)
    return vv[idx], (vt[idx] - targets) * 1000.0


def signal_grid(raw: RawVehicle, anchor_s: float, cols: Iterable[str], grid: np.ndarray) -> pd.DataFrame:
    t_rel_anchor = raw.df["t_rel_record_s"].to_numpy(dtype=float) - anchor_s
    out = pd.DataFrame({"t_rel_old_anchor_s": grid})
    out["t_rel_record_s"] = anchor_s + grid
    out["StorageTime"] = [
        (raw.start_time + pd.to_timedelta(float(t), unit="s")).isoformat(sep=" ")
        for t in out["t_rel_record_s"]
    ]

    for col in cols:
        if col not in raw.df.columns:
            continue
        values = pd.to_numeric(raw.df[col], errors="coerce").to_numpy(dtype=float)
        sampled, err_ms = nearest_values(t_rel_anchor, values, grid)
        out[col] = sampled
        out[f"{col}__time_error_ms"] = err_ms
    return out


def rolling_smooth(values: pd.Series, window: int = 7) -> pd.Series:
    # 先 median 再 mean，减少单点尖峰对 onset 检测的影响。
    return (
        values.astype(float)
        .rolling(window=window, center=True, min_periods=max(2, window // 2))
        .median()
        .rolling(window=window, center=True, min_periods=max(2, window // 2))
        .mean()
    )


def nanmax_abs(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return math.nan
    return float(np.max(np.abs(arr)))


def peak_abs_in(grid_df: pd.DataFrame, lo: float, hi: float, col: str = "steering_delta_from_baseline") -> tuple[float, float, float]:
    seg = grid_df[(grid_df["t_rel_old_anchor_s"] >= lo) & (grid_df["t_rel_old_anchor_s"] <= hi)]
    if seg.empty or col not in seg.columns:
        return math.nan, math.nan, math.nan
    abs_values = seg[col].abs()
    if abs_values.notna().sum() == 0:
        return math.nan, math.nan, math.nan
    idx = abs_values.idxmax()
    return (
        float(abs_values.loc[idx]),
        float(seg.loc[idx, "t_rel_old_anchor_s"]),
        float(seg.loc[idx, col]),
    )


def first_sustained_onset(
    rel_t: np.ndarray,
    delta: np.ndarray,
    threshold: float,
    min_duration_s: float = 0.30,
    search_lo: float = -8.0,
    search_hi: float = 0.0,
) -> float:
    """寻找旧锚点前第一个持续越过阈值的方向盘起点。"""
    valid = np.isfinite(delta) & (rel_t >= search_lo) & (rel_t <= search_hi)
    if not np.any(valid):
        return math.nan

    mask = np.zeros(len(rel_t), dtype=bool)
    mask[valid] = np.abs(delta[valid]) >= threshold
    min_len = max(2, int(round(min_duration_s / GRID_STEP_S)))

    run_start: int | None = None
    for i, flag in enumerate(mask):
        if flag and run_start is None:
            run_start = i
        if (not flag or i == len(mask) - 1) and run_start is not None:
            run_end = i if not flag else i + 1
            if run_end - run_start >= min_len:
                return float(rel_t[run_start])
            run_start = None
    return math.nan


def finite_ratio(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or abs(b) < 1e-12:
        return math.nan
    return float(a / b)


def zscore_for_plot(values: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    std = np.nanstd(arr)
    if not np.isfinite(std) or std < 1e-12:
        return np.zeros_like(arr, dtype=float)
    return (arr - np.nanmean(arr)) / std


def build_targets(eval_df: pd.DataFrame, casebook_df: pd.DataFrame, v231_meta: pd.DataFrame) -> pd.DataFrame:
    """构建本轮重锚定候选的样本范围：v230 casebook 唯一样本 + v231 六样本。"""
    rows: list[dict] = []

    v231_rank = {}
    v231_display_pool = {}
    for _, row in v231_meta.iterrows():
        v231_rank[str(row["sample_id"])] = int(row["rank"])
        v231_display_pool[str(row["sample_id"])] = str(row["display_pool"])

    sample_ids = sorted(set(casebook_df["sample_id"].astype(str)) | set(v231_meta["sample_id"].astype(str)))
    for sample_id in sample_ids:
        case_rows = casebook_df[casebook_df["sample_id"].astype(str) == sample_id].copy()
        if not case_rows.empty:
            # 同一样本可能 loose/strict 都入选，显示时用 tail_rmSE 最大的那一行代表人工优先级。
            case_rows = case_rows.sort_values(["tail_rmse", "rmse"], ascending=False)
            display_case = case_rows.iloc[0]
            casebook_entries = " | ".join(
                f"{r.pool_key}#{int(r.casebook_rank)}:{r.selection_bucket}"
                for r in case_rows.itertuples(index=False)
            )
            casebook_tail_rmse_max = float(case_rows["tail_rmse"].max())
        else:
            display_case = None
            casebook_entries = ""
            casebook_tail_rmse_max = math.nan

        preferred_pool = v231_display_pool.get(sample_id)
        if preferred_pool is None and display_case is not None:
            preferred_pool = str(display_case["pool_key"])
        if preferred_pool is None:
            preferred_pool = "loose_main_pool"

        eval_rows = eval_df[(eval_df["sample_id"].astype(str) == sample_id) & (eval_df["pool_key"].astype(str) == preferred_pool)]
        if eval_rows.empty:
            eval_rows = eval_df[eval_df["sample_id"].astype(str) == sample_id]
        if eval_rows.empty:
            continue
        erow = eval_rows.iloc[0]

        rows.append(
            {
                "sample_id": sample_id,
                "source_scope": "v231_six" if sample_id in v231_rank else "v230_casebook",
                "v231_rank": v231_rank.get(sample_id, math.nan),
                "display_pool": preferred_pool,
                "display_model": erow.get("formal_model", ""),
                "subject": erow["subject"],
                "recording": erow["recording"],
                "old_anchor_s": float(erow["anchor_s"]),
                "scene_type": erow.get("scene_type", ""),
                "route_event": erow.get("route_event", ""),
                "eval_rmse": float(erow.get("rmse", math.nan)),
                "eval_tail_rmse": float(erow.get("tail_rmse", math.nan)),
                "eval_observed_peak_abs": float(erow.get("observed_peak_abs", math.nan)),
                "eval_pred_peak_abs": float(erow.get("pred_peak_abs", math.nan)),
                "eval_peak_ratio": float(erow.get("peak_ratio", math.nan)),
                "casebook_entries": casebook_entries,
                "casebook_tail_rmse_max": casebook_tail_rmse_max,
            }
        )
    return pd.DataFrame(rows).sort_values(["source_scope", "casebook_tail_rmse_max"], ascending=[False, False])


def score_one_sample(target: pd.Series, raw_cache: dict[str, RawVehicle]) -> tuple[dict, pd.DataFrame]:
    subject = str(target["subject"])
    recording = str(target["recording"])
    old_anchor_s = float(target["old_anchor_s"])
    sample_id = str(target["sample_id"])
    raw_path = RAW_ROOT / subject / f"{recording}_vehicle.csv"

    if not raw_path.exists():
        raise FileNotFoundError(str(raw_path))

    raw_key = str(raw_path)
    if raw_key not in raw_cache:
        raw_cache[raw_key] = read_raw_vehicle(raw_path)
    raw = raw_cache[raw_key]

    present_cols = [col for col in RAW_CANDIDATE_COLS if col in raw.usecols and col not in ("ID", "StorageTime")]
    grid_df = signal_grid(raw, old_anchor_s, present_cols, GRID)
    grid_df.insert(0, "sample_id", sample_id)
    grid_df.insert(1, "old_anchor_s", old_anchor_s)

    steering = pd.to_numeric(grid_df["zx|SteeringWheel"], errors="coerce")
    steering_smooth = rolling_smooth(steering)
    grid_df["steering_smooth"] = steering_smooth

    baseline_seg = grid_df[(grid_df["t_rel_old_anchor_s"] >= -8.0) & (grid_df["t_rel_old_anchor_s"] <= -6.0)]
    baseline = float(np.nanmedian(baseline_seg["steering_smooth"]))
    if not np.isfinite(baseline):
        baseline_seg = grid_df[(grid_df["t_rel_old_anchor_s"] >= -10.0) & (grid_df["t_rel_old_anchor_s"] <= -8.0)]
        baseline = float(np.nanmedian(baseline_seg["steering_smooth"]))
    if not np.isfinite(baseline):
        baseline = float(np.nanmedian(grid_df["steering_smooth"]))

    grid_df["steering_baseline"] = baseline
    grid_df["steering_delta_from_baseline"] = grid_df["steering_smooth"] - baseline
    rel_t = grid_df["t_rel_old_anchor_s"].to_numpy(dtype=float)
    delta = grid_df["steering_delta_from_baseline"].to_numpy(dtype=float)

    if np.isfinite(delta).sum() >= 3:
        rate = np.gradient(delta, rel_t)
    else:
        rate = np.full(len(delta), np.nan)
    grid_df["steering_delta_rate"] = rate

    peak_all = nanmax_abs(grid_df.loc[(grid_df["t_rel_old_anchor_s"] >= -8.0) & (grid_df["t_rel_old_anchor_s"] <= 8.0), "steering_delta_from_baseline"])
    threshold = max(0.35, 0.18 * peak_all) if np.isfinite(peak_all) else math.nan
    candidate_rel_old = first_sustained_onset(rel_t, delta, threshold) if np.isfinite(threshold) else math.nan
    candidate_anchor_s = old_anchor_s + candidate_rel_old if np.isfinite(candidate_rel_old) else math.nan
    anchor_shift_s = candidate_rel_old if np.isfinite(candidate_rel_old) else math.nan
    grid_df["candidate_anchor_s"] = candidate_anchor_s
    grid_df["t_rel_candidate_anchor_s"] = grid_df["t_rel_old_anchor_s"] - candidate_rel_old if np.isfinite(candidate_rel_old) else math.nan
    grid_df["reanchor_threshold_abs_delta"] = threshold

    pre3_peak, pre3_t, pre3_signed = peak_abs_in(grid_df, -3.0, 0.0)
    pre8_peak, pre8_t, pre8_signed = peak_abs_in(grid_df, -8.0, 0.0)
    post03_peak, post03_t, post03_signed = peak_abs_in(grid_df, 0.0, 3.0)
    post38_peak, post38_t, post38_signed = peak_abs_in(grid_df, 3.0, 8.0)
    old_anchor_idx = int(np.nanargmin(np.abs(rel_t - 0.0)))
    old_delta = float(delta[old_anchor_idx]) if np.isfinite(delta[old_anchor_idx]) else math.nan

    pre_to_post_ratio = finite_ratio(pre3_peak, post03_peak)
    old_phase_ratio = finite_ratio(abs(old_delta), peak_all)
    manual_confirmed = sample_id in MANUAL_CONFIRMED_LATE

    score = 0.0
    reasons: list[str] = []
    if manual_confirmed:
        score += 5.0
        reasons.append("用户人工确认锚点晚")
    if np.isfinite(pre3_peak) and np.isfinite(post03_peak) and pre3_peak >= max(0.6, 0.65 * post03_peak):
        score += 2.0
        reasons.append("旧锚点前3秒已有明显转向")
    elif np.isfinite(pre3_peak) and pre3_peak >= 0.6:
        score += 1.0
        reasons.append("旧锚点前3秒存在可见转向")
    if np.isfinite(old_phase_ratio) and old_phase_ratio >= 0.35:
        score += 1.0
        reasons.append("旧锚点处已处于响应进程中")
    if np.isfinite(candidate_rel_old) and candidate_rel_old <= -0.5:
        score += 1.0
        reasons.append("检测到旧锚点前的候选起点")
    if np.isfinite(post38_peak) and np.isfinite(post03_peak) and post38_peak >= max(0.8, 1.25 * post03_peak):
        reasons.append("旧锚点后3-8秒仍有更大峰值，需另查窗口/horizon")

    if manual_confirmed:
        priority = "P0_manual_confirmed_reanchor"
    elif score >= 4 and np.isfinite(anchor_shift_s) and anchor_shift_s <= -0.5:
        priority = "P1_high_reanchor_review"
    elif score >= 2.5 and np.isfinite(anchor_shift_s) and anchor_shift_s <= -0.5:
        priority = "P2_medium_reanchor_review"
    else:
        priority = "P3_not_primary_late_anchor"

    old_anchor_abs = raw.start_time + pd.to_timedelta(old_anchor_s, unit="s")
    candidate_anchor_abs = raw.start_time + pd.to_timedelta(candidate_anchor_s, unit="s") if np.isfinite(candidate_anchor_s) else pd.NaT

    row = {
        "sample_id": sample_id,
        "source_scope": target["source_scope"],
        "v231_rank": target.get("v231_rank", math.nan),
        "display_pool": target["display_pool"],
        "display_model": target["display_model"],
        "subject": subject,
        "recording": recording,
        "raw_vehicle_csv": str(raw_path.resolve()),
        "scene_type": target["scene_type"],
        "route_event": target["route_event"],
        "old_anchor_s": old_anchor_s,
        "old_anchor_abs_time": old_anchor_abs.isoformat(sep=" "),
        "candidate_anchor_s": candidate_anchor_s,
        "candidate_anchor_abs_time": candidate_anchor_abs.isoformat(sep=" ") if pd.notna(candidate_anchor_abs) else "",
        "anchor_shift_s": anchor_shift_s,
        "steering_baseline_old_minus8_to_minus6": baseline,
        "old_anchor_steering_delta_from_baseline": old_delta,
        "old_anchor_phase_ratio_abs_delta_over_peak": old_phase_ratio,
        "reanchor_threshold_abs_delta": threshold,
        "pre_8_0_peak_abs_delta": pre8_peak,
        "pre_8_0_peak_t_rel_old_s": pre8_t,
        "pre_8_0_peak_signed_delta": pre8_signed,
        "pre_3_0_peak_abs_delta": pre3_peak,
        "pre_3_0_peak_t_rel_old_s": pre3_t,
        "pre_3_0_peak_signed_delta": pre3_signed,
        "post_0_3_peak_abs_delta": post03_peak,
        "post_0_3_peak_t_rel_old_s": post03_t,
        "post_0_3_peak_signed_delta": post03_signed,
        "post_3_8_peak_abs_delta": post38_peak,
        "post_3_8_peak_t_rel_old_s": post38_t,
        "post_3_8_peak_signed_delta": post38_signed,
        "pre3_to_post03_peak_ratio": pre_to_post_ratio,
        "late_anchor_score": score,
        "review_priority": priority,
        "evidence_reason_cn": "；".join(reasons) if reasons else "未达到晚锚点主证据阈值",
        "manual_feedback_status": "manual_confirmed_anchor_late" if manual_confirmed else "",
        "eval_rmse": target["eval_rmse"],
        "eval_tail_rmse": target["eval_tail_rmse"],
        "eval_observed_peak_abs": target["eval_observed_peak_abs"],
        "eval_pred_peak_abs": target["eval_pred_peak_abs"],
        "eval_peak_ratio": target["eval_peak_ratio"],
        "casebook_entries": target["casebook_entries"],
    }
    return row, grid_df


def make_review_table(all_scored: pd.DataFrame) -> pd.DataFrame:
    review = all_scored[all_scored["review_priority"].isin(
        ["P0_manual_confirmed_reanchor", "P1_high_reanchor_review", "P2_medium_reanchor_review"]
    )].copy()
    review = review.sort_values(
        ["review_priority", "late_anchor_score", "eval_tail_rmse"],
        ascending=[True, False, False],
    )
    review["human_decision"] = ""
    review["human_corrected_anchor_s"] = ""
    review["human_corrected_anchor_abs_time"] = ""
    review["human_use_for_training"] = ""
    review["human_note_cn"] = ""
    return review


def make_key_table(review_df: pd.DataFrame, grid_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for r in review_df.itertuples(index=False):
        sample_grid = grid_df[grid_df["sample_id"] == r.sample_id].copy()
        for ref_name, offsets in [("old_anchor", KEY_OFFSETS_OLD), ("candidate_anchor", KEY_OFFSETS_CANDIDATE)]:
            if ref_name == "old_anchor":
                ref_col = "t_rel_old_anchor_s"
            else:
                ref_col = "t_rel_candidate_anchor_s"
                if sample_grid[ref_col].isna().all():
                    continue
            rel_values = sample_grid[ref_col].to_numpy(dtype=float)
            for off in offsets:
                idx = int(np.nanargmin(np.abs(rel_values - off)))
                row = sample_grid.iloc[idx]
                out = {
                    "sample_id": r.sample_id,
                    "reference": ref_name,
                    "target_offset_s": off,
                    "actual_offset_s": float(row[ref_col]),
                    "t_rel_old_anchor_s": float(row["t_rel_old_anchor_s"]),
                    "t_rel_candidate_anchor_s": float(row["t_rel_candidate_anchor_s"]) if np.isfinite(row["t_rel_candidate_anchor_s"]) else math.nan,
                    "StorageTime": row["StorageTime"],
                    "zx|SteeringWheel": row.get("zx|SteeringWheel", math.nan),
                    "steering_smooth": row.get("steering_smooth", math.nan),
                    "steering_delta_from_baseline": row.get("steering_delta_from_baseline", math.nan),
                    "steering_delta_rate": row.get("steering_delta_rate", math.nan),
                    "zx1|v_km/h": row.get("zx1|v_km/h", math.nan),
                    "zx|v_km/h": row.get("zx|v_km/h", math.nan),
                    "zx|ay": row.get("zx|ay", math.nan),
                    "zx|vyaw": row.get("zx|vyaw", math.nan),
                    "zx|yaw": row.get("zx|yaw", math.nan),
                    "zx|roll": row.get("zx|roll", math.nan),
                    "zx1|lanecurvatureXY": row.get("zx1|lanecurvatureXY", math.nan),
                    "zx|lanecurvatureXY": row.get("zx|lanecurvatureXY", math.nan),
                    "zx1|lateraldistance": row.get("zx1|lateraldistance", math.nan),
                    "zx|lateraldistance": row.get("zx|lateraldistance", math.nan),
                }
                rows.append(out)
    return pd.DataFrame(rows)


def plot_review_case(row: pd.Series, grid_df: pd.DataFrame) -> Path:
    sample_id = str(row["sample_id"])
    sample_grid = grid_df[grid_df["sample_id"] == sample_id].copy()
    x = sample_grid["t_rel_old_anchor_s"].to_numpy(dtype=float)
    candidate_rel = float(row["anchor_shift_s"]) if np.isfinite(row["anchor_shift_s"]) else math.nan

    fig, axes = plt.subplots(5, 1, figsize=(15, 13), sharex=True)
    title = (
        f"{sample_id}\n"
        f"old_anchor={float(row['old_anchor_s']):.3f}s, candidate={float(row['candidate_anchor_s']):.3f}s, "
        f"shift={candidate_rel:.2f}s, priority={row['review_priority']}"
    )
    fig.suptitle(title, fontsize=12)

    def mark(ax):
        ax.axvline(0, color="#dc2626", lw=1.7, label="old anchor")
        if np.isfinite(candidate_rel):
            ax.axvline(candidate_rel, color="#16a34a", lw=1.7, label="candidate anchor")
        ax.axvspan(-3, 0, color="#9ca3af", alpha=0.10, linewidth=0)
        ax.grid(True, alpha=0.25)

    ax = axes[0]
    mark(ax)
    ax.plot(x, sample_grid["zx|SteeringWheel"], color="#94a3b8", lw=0.9, label="Steering raw")
    ax.plot(x, sample_grid["steering_smooth"], color="#1d4ed8", lw=1.3, label="Steering smooth")
    ax.axhline(float(row["steering_baseline_old_minus8_to_minus6"]), color="#475569", lw=1.0, ls="--", label="baseline")
    threshold = float(row["reanchor_threshold_abs_delta"])
    baseline = float(row["steering_baseline_old_minus8_to_minus6"])
    ax.axhline(baseline + threshold, color="#16a34a", lw=0.9, ls=":", label="onset threshold")
    ax.axhline(baseline - threshold, color="#16a34a", lw=0.9, ls=":")
    ax.set_title("方向盘原始值 / 平滑值 / onset 阈值", loc="left", fontsize=10)
    ax.legend(loc="upper right", fontsize=8, ncol=3)

    ax = axes[1]
    mark(ax)
    ax.plot(x, sample_grid["steering_delta_from_baseline"], color="#7c3aed", lw=1.2, label="delta from baseline")
    ax.plot(x, sample_grid["steering_delta_rate"], color="#ea580c", lw=1.0, alpha=0.85, label="delta rate")
    ax.axhline(threshold, color="#16a34a", lw=0.9, ls=":")
    ax.axhline(-threshold, color="#16a34a", lw=0.9, ls=":")
    ax.set_title("方向盘相对基线变化与变化率", loc="left", fontsize=10)
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    ax = axes[2]
    mark(ax)
    for col in ["zx1|v_km/h", "zx|v_km/h", "zx|vx", "zx|vy"]:
        if col in sample_grid.columns and sample_grid[col].notna().sum() > 0:
            ax.plot(x, sample_grid[col], lw=1.0, label=col)
    ax.set_title("速度 / 速度分量", loc="left", fontsize=10)
    ax.legend(loc="upper right", fontsize=8, ncol=3)

    ax = axes[3]
    mark(ax)
    for col in ["zx|ay", "zx|vyaw", "zx|ayaw", "zx|yaw", "zx|roll", "zx|pitch"]:
        if col in sample_grid.columns and sample_grid[col].notna().sum() > 0:
            ax.plot(x, zscore_for_plot(sample_grid[col]), lw=1.0, label=f"{col} (z)")
    ax.set_title("车辆动力学信号（z-score，仅看相对变化）", loc="left", fontsize=10)
    ax.legend(loc="upper right", fontsize=8, ncol=3)

    ax = axes[4]
    mark(ax)
    for col in ["zx1|lanecurvatureXY", "zx|lanecurvatureXY", "zx1|lateraldistance", "zx|lateraldistance", "zx|AcceleratorPedal", "zx|BrakePedal"]:
        if col in sample_grid.columns and sample_grid[col].notna().sum() > 0:
            ax.plot(x, zscore_for_plot(sample_grid[col]), lw=1.0, label=f"{col} (z)")
    ax.set_title("道路/车道/踏板信号（z-score，仅看相对变化）", loc="left", fontsize=10)
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    ax.set_xlabel("seconds relative to old anchor")
    ax.set_xlim(-10, 8)

    fig.text(0.01, 0.01, str(row["evidence_reason_cn"]), fontsize=9)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    safe_id = sample_id.replace(":", "_").replace("/", "_")
    out_path = FIGURES / f"{int(row['review_rank']):02d}_{safe_id}_reanchor_candidate.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def make_contact_sheet(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    thumb_w, thumb_h = 820, 710
    pad = 24
    label_h = 38
    cols = 2
    rows = math.ceil(len(paths) / cols)
    canvas = Image.new(
        "RGB",
        (cols * thumb_w + (cols + 1) * pad, rows * (thumb_h + label_h) + (rows + 1) * pad),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:  # pragma: no cover
        font = ImageFont.load_default()
    for i, path in enumerate(paths):
        image = Image.open(path).convert("RGB")
        image.thumbnail((thumb_w, thumb_h), Image.LANCZOS)
        x = pad + (i % cols) * (thumb_w + pad)
        y = pad + (i // cols) * (thumb_h + label_h + pad)
        draw.text((x, y), path.name[:88], fill=(20, 20, 20), font=font)
        canvas.paste(image, (x, y + label_h))
    out_path = FIGURES / "v232_reanchor_candidates_contact_sheet.png"
    canvas.save(out_path, quality=95)
    return out_path


def write_report(
    all_scored: pd.DataFrame,
    review_df: pd.DataFrame,
    figure_paths: list[Path],
    contact_sheet: Path | None,
) -> Path:
    report_path = REPORTS / "v232_late_anchor_reanchor_candidates_cn.md"
    lines: list[str] = []
    lines.append("# v232 过晚锚点重锚定候选审核包")
    lines.append("")
    lines.append("## 目的")
    lines.append("")
    lines.append("本包继续 v231 人工审核结论：先处理过晚锚点和目标窗口错位，再讨论模型结构。")
    lines.append("本轮只生成重锚定候选和证据，不直接修改训练标签，不训练模型，不改 formal headline。")
    lines.append("")
    lines.append("## 已纳入的人工边界")
    lines.append("")
    lines.append("- `rjy_Entity_Recording_2025_09_28_20_02_20_v108_010` 已由用户人工确认锚点晚了。")
    lines.append("- 不重启“先硬判断响应类型，再预测轨迹”的路线；该路线此前已尝试过，且存在分类错误传播。")
    lines.append("- 不把“一次性输出多个候选轨迹”作为下一步主线；该路线此前也已尝试过，即使 best candidate 仍有偏差。")
    lines.append("")
    lines.append("## 输出文件")
    lines.append("")
    lines.append(f"- 全量打分表：`{TABLES / 'v232_reanchor_candidate_all_scored.csv'}`")
    lines.append(f"- 人工审核表：`{TABLES / 'v232_reanchor_candidate_review_table.csv'}`")
    lines.append(f"- 0.05 秒信号网格：`{TABLES / 'v232_reanchor_grid_0p05s.csv'}`")
    lines.append(f"- 关键时刻表：`{TABLES / 'v232_reanchor_key_points.csv'}`")
    lines.append(f"- 图目录：`{FIGURES}`")
    if contact_sheet is not None:
        lines.append(f"- 候选图拼接总览：`{contact_sheet}`")
    lines.append(f"- ZIP 包：`{OUT / 'v232_late_anchor_reanchor_candidates_pack.zip'}`")
    lines.append("")
    lines.append("## 检测方法")
    lines.append("")
    lines.append("每个样本从原始车辆 CSV 读取旧锚点前后 `-10s~+8s` 的信号，并做信号级最近非空采样。")
    lines.append("方向盘以旧锚点前 `-8s~-6s` 的平滑中位数作为基线，计算 `steering_delta_from_baseline`。")
    lines.append("候选新锚点定义为旧锚点前第一次持续超过阈值的方向盘变化起点，阈值为 `max(0.35, 0.18 * window_peak_abs_delta)`。")
    lines.append("候选只作为人工审核入口，不自动生效。")
    lines.append("")
    lines.append("## 审核优先级汇总")
    lines.append("")
    if review_df.empty:
        lines.append("本轮没有产生 P0/P1/P2 重锚定候选。")
    else:
        lines.append("|rank|priority|sample_id|old_anchor_s|candidate_anchor_s|shift_s|score|pre3_peak|post03_peak|reason|")
        lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|---|")
        for row in review_df.itertuples(index=False):
            lines.append(
                f"|{int(row.review_rank)}|{row.review_priority}|`{row.sample_id}`|"
                f"{float(row.old_anchor_s):.3f}|{float(row.candidate_anchor_s):.3f}|{float(row.anchor_shift_s):.2f}|"
                f"{float(row.late_anchor_score):.1f}|{float(row.pre_3_0_peak_abs_delta):.3f}|{float(row.post_0_3_peak_abs_delta):.3f}|"
                f"{row.evidence_reason_cn}|"
            )
    lines.append("")
    lines.append("## 人工审核建议")
    lines.append("")
    lines.append("1. 先看 P0/P1：确认候选新锚点是否确实比旧锚点更接近事件起点。")
    lines.append("2. 如果确认，填写 `human_decision=accept_reanchor`、`human_corrected_anchor_s` 和 `human_use_for_training`。")
    lines.append("3. 如果候选过早或过晚，人工改写 `human_corrected_anchor_s`，不要直接采用算法候选。")
    lines.append("4. 只有人工确认后的样本才允许进入下一轮 label window 重建。")
    lines.append("5. 锚点确认无误但仍预测差的样本，才进入模型方法提升；不要把锚点晚样本混入模型失败结论。")
    lines.append("")
    lines.append("## 后续方法边界")
    lines.append("")
    lines.append("下一步不是继续加候选轨迹数，也不是硬响应类型分类，而是先完成重锚定候选的人工确认。")
    lines.append("重锚定后如果仍有系统偏差，再考虑目标窗口重建、偏差校正、连续相位/延迟参数或对齐鲁棒损失。")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def write_zip() -> Path:
    zip_path = OUT / "v232_late_anchor_reanchor_candidates_pack.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for folder in (TABLES, REPORTS, LOGS):
            for path in folder.rglob("*"):
                if path.is_file():
                    zf.write(path, path.relative_to(OUT))
        # 图文件较大，但审核时很重要，保留进包内。
        for path in FIGURES.rglob("*.png"):
            zf.write(path, path.relative_to(OUT))
    return zip_path


def main() -> None:
    eval_df = pd.read_csv(EVAL_PATH, encoding="utf-8-sig")
    casebook_df = pd.read_csv(CASEBOOK_PATH, encoding="utf-8-sig")
    v231_meta = pd.read_csv(V231_META_PATH, encoding="utf-8-sig")

    targets = build_targets(eval_df, casebook_df, v231_meta)
    targets_path = TABLES / "v232_target_samples.csv"
    targets.to_csv(targets_path, index=False, encoding="utf-8-sig")

    raw_cache: dict[str, RawVehicle] = {}
    scored_rows: list[dict] = []
    grid_frames: list[pd.DataFrame] = []
    errors: list[dict] = []

    for row in targets.itertuples(index=False):
        target = pd.Series(row._asdict())
        try:
            scored, grid = score_one_sample(target, raw_cache)
            scored_rows.append(scored)
            grid_frames.append(grid)
        except Exception as exc:
            errors.append({"sample_id": target.get("sample_id", ""), "error": repr(exc)})

    all_scored = pd.DataFrame(scored_rows)
    if not all_scored.empty:
        all_scored = all_scored.sort_values(
            ["review_priority", "late_anchor_score", "eval_tail_rmse"],
            ascending=[True, False, False],
        )
    grid_all = pd.concat(grid_frames, ignore_index=True) if grid_frames else pd.DataFrame()
    review_df = make_review_table(all_scored)
    if not review_df.empty:
        review_df.insert(0, "review_rank", range(1, len(review_df) + 1))
        # 同步 rank 回 all_scored 方便画图文件命名。
        rank_map = dict(zip(review_df["sample_id"], review_df["review_rank"]))
        all_scored["review_rank"] = all_scored["sample_id"].map(rank_map)
    else:
        all_scored["review_rank"] = math.nan

    review_grid = grid_all[grid_all["sample_id"].isin(set(review_df["sample_id"]))].copy() if not review_df.empty else pd.DataFrame()
    key_df = make_key_table(review_df, grid_all) if not review_df.empty else pd.DataFrame()

    all_scored_path = TABLES / "v232_reanchor_candidate_all_scored.csv"
    review_path = TABLES / "v232_reanchor_candidate_review_table.csv"
    grid_path = TABLES / "v232_reanchor_grid_0p05s.csv"
    key_path = TABLES / "v232_reanchor_key_points.csv"
    errors_path = LOGS / "v232_reanchor_errors.json"
    manifest_path = LOGS / "run_manifest.json"

    all_scored.to_csv(all_scored_path, index=False, encoding="utf-8-sig")
    review_df.to_csv(review_path, index=False, encoding="utf-8-sig")
    review_grid.to_csv(grid_path, index=False, encoding="utf-8-sig")
    key_df.to_csv(key_path, index=False, encoding="utf-8-sig")
    errors_path.write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")

    figure_paths: list[Path] = []
    for row in review_df.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        path = plot_review_case(row_series, grid_all)
        figure_paths.append(path)
    contact_sheet = make_contact_sheet(figure_paths)

    report_path = write_report(all_scored, review_df, figure_paths, contact_sheet)
    zip_path = write_zip()

    manifest = {
        "version": "v232_late_anchor_reanchor_candidates_20260624",
        "target_sample_count": int(len(targets)),
        "scored_sample_count": int(len(all_scored)),
        "review_candidate_count": int(len(review_df)),
        "figure_count": int(len(figure_paths)),
        "errors": errors,
        "outputs": {
            "targets": str(targets_path),
            "all_scored": str(all_scored_path),
            "review": str(review_path),
            "grid": str(grid_path),
            "key_points": str(key_path),
            "report": str(report_path),
            "contact_sheet": str(contact_sheet) if contact_sheet else "",
            "zip": str(zip_path),
        },
        "method_boundaries": [
            "no_training",
            "no_formal_headline_change",
            "no_hard_response_type_classifier_mainline",
            "no_simple_multi_candidate_trajectory_mainline",
            "manual_review_required_before_label_rebuild",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("DONE v232")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if not review_df.empty:
        print(review_df[[
            "review_rank",
            "review_priority",
            "sample_id",
            "old_anchor_s",
            "candidate_anchor_s",
            "anchor_shift_s",
            "late_anchor_score",
            "evidence_reason_cn",
        ]].to_string(index=False))


if __name__ == "__main__":
    main()
