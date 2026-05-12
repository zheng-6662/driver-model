from __future__ import annotations

import json
import math
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


PROJECT_ROOT = Path("F:/data_set_process/data_process")
REBUILD_ROOT = PROJECT_ROOT / "05_rebuild_from_raw_20260511"
MANIFEST_PATH = (
    REBUILD_ROOT
    / "02_samples"
    / "vehicle_instability_response_task_decision_v0_1"
    / "tables"
    / "sample_response_task_manifest.csv"
)
OUT_ROOT = REBUILD_ROOT / "04_style" / "stage04_continuous_style_protocol_v0_1"
TABLE_DIR = OUT_ROOT / "tables"
FIGURE_DIR = OUT_ROOT / "figures"
LOG_DIR = OUT_ROOT / "logs"
REPORT_DIR = REBUILD_ROOT / "09_reports"

WINDOW_CONFIG_ID = "pre3_label3_response_coverage"
TASK_ROLE = "response3s_strict_core_candidate"
DIRECT_INPUT_START_REL_S = -3.0
DIRECT_INPUT_END_REL_S = 0.0
LABEL_START_REL_S = 0.0
LABEL_END_REL_S = 3.0

STYLE_WINDOWS = [
    {
        "window_id": "prefix_until_guard3",
        "description_cn": "从本条记录开始到事件前 3 秒，排除直接输入窗口。",
        "lookback_s": None,
        "guard_s": 3.0,
        "min_duration_s": 20.0,
        "min_rows": 200,
    },
    {
        "window_id": "last120_guard3",
        "description_cn": "事件前 3 秒之前的最近 120 秒历史。",
        "lookback_s": 120.0,
        "guard_s": 3.0,
        "min_duration_s": 40.0,
        "min_rows": 400,
    },
    {
        "window_id": "last60_guard3",
        "description_cn": "事件前 3 秒之前的最近 60 秒历史。",
        "lookback_s": 60.0,
        "guard_s": 3.0,
        "min_duration_s": 20.0,
        "min_rows": 200,
    },
    {
        "window_id": "last30_guard3",
        "description_cn": "事件前 3 秒之前的最近 30 秒历史，作为较短连续风格候选。",
        "lookback_s": 30.0,
        "guard_s": 3.0,
        "min_duration_s": 12.0,
        "min_rows": 120,
    },
]

META_COLS = [
    "sample_id",
    "event_uid",
    "subject",
    "session_stamp",
    "session_level_split",
    "subject_level_split",
    "road_design_module_name",
    "road_design_instance_name",
    "anchor_time_rel_s",
    "input_start_time_rel_s",
    "input_end_time_rel_s",
    "label_start_time_rel_s",
    "label_end_time_rel_s",
    "eval_label_peak_abs",
    "eval_label_peak_signed",
    "eval_label_peak_direction",
    "eval_label_reversal_count",
    "eval_label_morphology",
    "eval_is_large_response_train_session_p75",
    "eval_is_difficult_train_session_p80",
    "vehicle_raw_absolute_path",
]

SOURCE_FEATURES = {
    "speed_kmh": "zx1|v_km/h",
    "steering": "zx|SteeringWheel",
    "longitudinal_accel": "zx|ax",
    "lateral_accel": "zx|ay",
    "yaw_rate": "zx|vyaw",
    "roll_rate": "zx|vroll",
    "lane_offset": "zx1|lateraldistance",
    "lane_curvature": "zx1|lanecurvatureXY",
    "brake_pedal": "zx|BrakePedal",
    "accelerator_pedal": "zx|AcceleratorPedal",
    "friction_mu": "zx1|mu",
}


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIGURE_DIR, LOG_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False, **kwargs)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8-sig")


def numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def storage_time_to_rel_seconds(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() >= max(10, int(len(series) * 0.5)):
        first = numeric[numeric.notna()].iloc[0]
        rel = numeric - first
        diffs = np.diff(rel[numeric.notna()].to_numpy(dtype=float))
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if len(diffs):
            median_dt = float(np.median(diffs))
            if median_dt > 0.1:
                rel = rel / 1000.0
        return rel

    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.notna().sum() == 0:
        return pd.Series(np.nan, index=series.index)
    first_ts = parsed[parsed.notna()].iloc[0]
    return (parsed - first_ts).dt.total_seconds()


def load_vehicle(vehicle_path: Path, cache: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    key = str(vehicle_path)
    if key in cache:
        return cache[key]
    df = read_csv(vehicle_path)
    if "StorageTime" not in df.columns:
        raise ValueError(f"StorageTime column missing: {vehicle_path}")
    df = df.copy()
    df["time_rel_s"] = storage_time_to_rel_seconds(df["StorageTime"])
    df = df[df["time_rel_s"].notna()].sort_values("time_rel_s")
    df = df.drop_duplicates(subset=["time_rel_s"], keep="first").reset_index(drop=True)
    cache[key] = df
    return df


def finite_values(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.array([], dtype=float)
    values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def add_basic_stats(out: Dict[str, float], prefix: str, values: np.ndarray, denom: int) -> None:
    valid = values[np.isfinite(values)]
    out[f"{prefix}_valid_ratio"] = float(len(valid) / denom) if denom else np.nan
    if len(valid) == 0:
        for suffix in ["mean", "std", "p10", "p50", "p90", "abs_mean", "abs_p95", "rms"]:
            out[f"{prefix}_{suffix}"] = np.nan
        return
    abs_valid = np.abs(valid)
    out[f"{prefix}_mean"] = float(np.mean(valid))
    out[f"{prefix}_std"] = float(np.std(valid))
    out[f"{prefix}_p10"] = float(np.percentile(valid, 10))
    out[f"{prefix}_p50"] = float(np.percentile(valid, 50))
    out[f"{prefix}_p90"] = float(np.percentile(valid, 90))
    out[f"{prefix}_abs_mean"] = float(np.mean(abs_valid))
    out[f"{prefix}_abs_p95"] = float(np.percentile(abs_valid, 95))
    out[f"{prefix}_rms"] = float(math.sqrt(np.mean(valid * valid)))


def add_rate_stats(out: Dict[str, float], df: pd.DataFrame, source_col: str, prefix: str) -> None:
    if source_col not in df.columns or len(df) < 3:
        for suffix in ["abs_mean", "abs_p95", "rms", "valid_ratio"]:
            out[f"{prefix}_{suffix}"] = np.nan
        return
    time = pd.to_numeric(df["time_rel_s"], errors="coerce").to_numpy(dtype=float)
    values = pd.to_numeric(df[source_col], errors="coerce").to_numpy(dtype=float)
    dt = np.diff(time)
    dv = np.diff(values)
    mask = np.isfinite(dt) & np.isfinite(dv) & (dt > 1e-6)
    rates = dv[mask] / dt[mask]
    denom = max(1, len(df) - 1)
    out[f"{prefix}_valid_ratio"] = float(len(rates) / denom)
    if len(rates) == 0:
        for suffix in ["abs_mean", "abs_p95", "rms"]:
            out[f"{prefix}_{suffix}"] = np.nan
        return
    abs_rates = np.abs(rates)
    out[f"{prefix}_abs_mean"] = float(np.mean(abs_rates))
    out[f"{prefix}_abs_p95"] = float(np.percentile(abs_rates, 95))
    out[f"{prefix}_rms"] = float(math.sqrt(np.mean(rates * rates)))


def compute_window_features(window_df: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    denom = len(window_df)
    for feature_name, col in SOURCE_FEATURES.items():
        add_basic_stats(out, feature_name, finite_values(window_df, col), denom)
    add_rate_stats(out, window_df, "zx|SteeringWheel", "steering_rate")
    add_rate_stats(out, window_df, "zx1|v_km/h", "speed_rate")
    return out


def interval_overlaps(
    a_start: float, a_end: float, b_start: float, b_end: float, eps: float = 1e-6
) -> bool:
    if not all(np.isfinite([a_start, a_end, b_start, b_end])):
        return False
    return (a_start < b_end - eps) and (a_end > b_start + eps)


def make_style_feature_rows(manifest: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cache: Dict[str, pd.DataFrame] = {}
    rows: List[Dict[str, object]] = []

    for _, sample in manifest.iterrows():
        vehicle_path_value = sample.get("vehicle_raw_absolute_path")
        if not isinstance(vehicle_path_value, str) or not vehicle_path_value.strip():
            vehicle_path_value = sample.get("vehicle_absolute_path")
        vehicle_path = Path(str(vehicle_path_value))
        anchor = float(sample["anchor_time_rel_s"])
        input_start = float(sample["input_start_time_rel_s"])
        input_end = float(sample["input_end_time_rel_s"])
        label_start = float(sample["label_start_time_rel_s"])
        label_end = float(sample["label_end_time_rel_s"])

        try:
            vehicle = load_vehicle(vehicle_path, cache)
            file_min = float(vehicle["time_rel_s"].min())
            file_max = float(vehicle["time_rel_s"].max())
            load_status = "loaded"
            load_error = ""
        except Exception as exc:
            vehicle = pd.DataFrame()
            file_min = np.nan
            file_max = np.nan
            load_status = "read_failed"
            load_error = str(exc)

        for spec in STYLE_WINDOWS:
            guard_s = float(spec["guard_s"])
            guard_end_s = anchor - guard_s
            if np.isfinite(input_start):
                end_s = min(guard_end_s, input_start - 1e-6)
            else:
                end_s = guard_end_s
            if spec["lookback_s"] is None:
                start_s = file_min
            else:
                start_s = max(file_min, end_s - float(spec["lookback_s"]))

            row: Dict[str, object] = {col: sample.get(col, "") for col in META_COLS if col in sample.index}
            row.update(
                {
                    "window_id": spec["window_id"],
                    "window_description_cn": spec["description_cn"],
                    "style_window_start_rel_s": start_s,
                    "style_window_end_rel_s": end_s,
                    "style_nominal_guard_end_rel_s": guard_end_s,
                    "style_guard_s": guard_s,
                    "style_requested_lookback_s": spec["lookback_s"]
                    if spec["lookback_s"] is not None
                    else np.nan,
                    "vehicle_time_min_s": file_min,
                    "vehicle_time_max_s": file_max,
                    "vehicle_load_status": load_status,
                    "vehicle_load_error": load_error,
                    "overlaps_direct_input_window": interval_overlaps(
                        start_s, end_s, input_start, input_end
                    ),
                    "overlaps_label_window": interval_overlaps(start_s, end_s, label_start, label_end),
                    "uses_post_anchor_future": bool(np.isfinite(end_s) and end_s > anchor),
                }
            )

            if load_status != "loaded":
                row.update(
                    {
                        "style_window_status": "read_failed",
                        "style_duration_s": np.nan,
                        "style_row_count": 0,
                        "style_sampling_rate_est_hz": np.nan,
                    }
                )
                rows.append(row)
                continue

            window = vehicle[(vehicle["time_rel_s"] >= start_s) & (vehicle["time_rel_s"] <= end_s)]
            duration_s = float(max(0.0, end_s - start_s)) if np.isfinite([start_s, end_s]).all() else np.nan
            row_count = int(len(window))
            row["style_duration_s"] = duration_s
            row["style_row_count"] = row_count
            row["style_sampling_rate_est_hz"] = float(row_count / duration_s) if duration_s > 0 else np.nan

            if row["overlaps_label_window"] or row["uses_post_anchor_future"]:
                row["style_window_status"] = "blocked_future_overlap"
            elif row["overlaps_direct_input_window"]:
                row["style_window_status"] = "blocked_direct_input_overlap"
            elif not np.isfinite(duration_s) or duration_s < float(spec["min_duration_s"]):
                row["style_window_status"] = "insufficient_history_duration"
            elif row_count < int(spec["min_rows"]):
                row["style_window_status"] = "insufficient_history_rows"
            else:
                row["style_window_status"] = "usable_no_future_leakage"

            row.update(compute_window_features(window))
            rows.append(row)

    long_df = pd.DataFrame(rows)
    wide = manifest[[col for col in META_COLS if col in manifest.columns]].copy()
    status_cols = [
        "style_window_status",
        "style_duration_s",
        "style_row_count",
        "style_sampling_rate_est_hz",
        "overlaps_direct_input_window",
        "overlaps_label_window",
        "uses_post_anchor_future",
    ]
    feature_cols = [
        c
        for c in long_df.columns
        if c not in set(META_COLS)
        | {
            "window_id",
            "window_description_cn",
            "style_window_start_rel_s",
            "style_window_end_rel_s",
            "style_nominal_guard_end_rel_s",
            "style_guard_s",
            "style_requested_lookback_s",
            "vehicle_time_min_s",
            "vehicle_time_max_s",
            "vehicle_load_status",
            "vehicle_load_error",
        }
    ]
    for window_id, sub in long_df.groupby("window_id", sort=False):
        by_sample = sub.set_index("sample_id")
        for col in status_cols:
            if col in by_sample.columns:
                wide[f"{window_id}__{col}"] = wide["sample_id"].map(by_sample[col])
        for col in feature_cols:
            if col in status_cols or col not in by_sample.columns:
                continue
            if pd.api.types.is_numeric_dtype(by_sample[col]):
                wide[f"{window_id}__{col}"] = wide["sample_id"].map(by_sample[col])

    usable_cols = [f"{spec['window_id']}__style_window_status" for spec in STYLE_WINDOWS]
    available = pd.DataFrame(
        {
            col: wide[col].eq("usable_no_future_leakage") if col in wide.columns else False
            for col in usable_cols
        }
    )
    wide["style_any_window_usable"] = available.any(axis=1)
    wide["style_all_windows_usable"] = available.all(axis=1)
    wide["style_usable_window_count"] = available.sum(axis=1)
    return long_df, wide


def make_train_z_table(wide: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    z_wide = wide.copy()
    split_col = "session_level_split"
    train_mask = z_wide[split_col].astype(str).str.lower().eq("train")
    numeric_cols = [
        c
        for c in z_wide.columns
        if "__" in c
        and pd.api.types.is_numeric_dtype(z_wide[c])
        and not c.endswith("__overlaps_direct_input_window")
        and not c.endswith("__overlaps_label_window")
        and not c.endswith("__uses_post_anchor_future")
    ]
    scaler_rows = []
    for col in numeric_cols:
        values = pd.to_numeric(z_wide.loc[train_mask, col], errors="coerce")
        finite = values[np.isfinite(values)]
        if len(finite) < 2:
            mean = np.nan
            std = np.nan
            status = "insufficient_train_values"
        else:
            mean = float(finite.mean())
            std = float(finite.std(ddof=0))
            status = "usable" if std > 1e-12 else "zero_variance_train"
        scaler_rows.append(
            {
                "feature": col,
                "split_strategy": "session_level_split",
                "fit_scope": "train_only",
                "train_value_count": int(len(finite)),
                "train_mean": mean,
                "train_std": std,
                "scaler_status": status,
            }
        )
        if status == "usable":
            z_wide[col] = (pd.to_numeric(z_wide[col], errors="coerce") - mean) / std
        else:
            z_wide[col] = np.nan
    return z_wide, pd.DataFrame(scaler_rows)


def make_protocol_tables(
    manifest: pd.DataFrame, long_df: pd.DataFrame, wide: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    source_protocol = pd.DataFrame(
        [
            {
                "source": "raw_vehicle_prefix_until_anchor_minus_3s",
                "allowed_role": "continuous_style_candidate",
                "status": "allowed_with_guard",
                "reason_cn": "只使用事件前 3 秒以前的车辆历史，不接触直接输入窗口和标签未来。",
            },
            {
                "source": "direct_vehicle_input_minus3_to_anchor",
                "allowed_role": "vehicle_baseline_input",
                "status": "not_style_feature",
                "reason_cn": "这是强车辆基线输入，阶段 4 风格特征刻意排除，避免把即时状态当风格。",
            },
            {
                "source": "post_anchor_label_0_to_3s",
                "allowed_role": "label_only",
                "status": "blocked",
                "reason_cn": "属于方向盘响应标签窗口，不能进入任何输入、风格或标准化拟合。",
            },
            {
                "source": "subject_id",
                "allowed_role": "control_only",
                "status": "control_required",
                "reason_cn": "只能作为驾驶员 ID 对照，不能把 ID 收益直接解释为连续风格收益。",
            },
            {
                "source": "road_design_module_or_event_context",
                "allowed_role": "context_or_coupling_audit",
                "status": "audit_required",
                "reason_cn": "道路/事件分布可能与被试和风格耦合，必须做耦合审计和置乱对照。",
            },
            {
                "source": "physio_eeg_emg_resp",
                "allowed_role": "none_in_stage4",
                "status": "blocked_until_stage5",
                "reason_cn": "阶段 4 只验证连续车辆风格；生理/脑电仍等待车辆+风格参照形成后再进入。",
            },
        ]
    )

    leakage_guard = pd.DataFrame(
        [
            {
                "guard_item": "style_window_end",
                "rule": "end <= anchor_time_rel_s - 3.0",
                "observed_bad_count": int(
                    long_df["overlaps_direct_input_window"].fillna(False).astype(bool).sum()
                ),
                "status": "pass" if not long_df["overlaps_direct_input_window"].any() else "fail",
                "note_cn": "风格窗口不得接触 [-3, 0] 直接车辆输入窗口。",
            },
            {
                "guard_item": "label_future",
                "rule": "style window must not overlap label [0, 3] s",
                "observed_bad_count": int(long_df["overlaps_label_window"].fillna(False).astype(bool).sum()),
                "status": "pass" if not long_df["overlaps_label_window"].any() else "fail",
                "note_cn": "风格特征不得使用事件后的方向盘响应结果。",
            },
            {
                "guard_item": "standardization",
                "rule": "fit scaler on session_level_split=train only",
                "observed_bad_count": 0,
                "status": "pass_protocol",
                "note_cn": "本脚本输出 train-only scaler 参数，val/test 只应用训练集统计。",
            },
            {
                "guard_item": "style_claim",
                "rule": "no effectiveness claim before RBF comparison + permutations + subject/session checks",
                "observed_bad_count": 0,
                "status": "blocked_claim",
                "note_cn": "本阶段只完成候选特征处理和协议，不证明连续风格有效。",
            },
        ]
    )

    permutation_plan = pd.DataFrame(
        [
            {
                "control_name": "within_subject_shuffle",
                "shuffle_scope": "same subject, same split",
                "purpose_cn": "检验风格向量是否只是驾驶员 ID 代理。",
                "implementation_rule_cn": "在每个 split 内按被试打乱样本的风格向量，车辆输入和标签不动。",
                "expected_if_style_real_cn": "收益应明显下降，但不应完全等同于跨被试打乱。",
            },
            {
                "control_name": "cross_subject_shuffle",
                "shuffle_scope": "same split across subjects",
                "purpose_cn": "检验跨驾驶员连续风格是否提供可定位的个体差异信息。",
                "implementation_rule_cn": "在同一 split 内跨被试随机替换风格向量，保持车辆输入、道路和标签不动。",
                "expected_if_style_real_cn": "如果风格有效，性能和物理指标收益应下降。",
            },
            {
                "control_name": "cross_session_shuffle",
                "shuffle_scope": "same subject when multiple sessions exist, otherwise mark unavailable",
                "purpose_cn": "检验风格是否依赖具体 session/道路段，而不是稳定驾驶习惯。",
                "implementation_rule_cn": "优先同被试跨 session 交换；无多 session 被试单独标记不可用。",
                "expected_if_style_real_cn": "若风格主要是 session/道路代理，跨 session 打乱会造成强下降或不稳定。",
            },
            {
                "control_name": "road_balanced_shuffle",
                "shuffle_scope": "same split and same road_design_module_name",
                "purpose_cn": "控制道路/事件分布耦合，避免把道路难度误当风格。",
                "implementation_rule_cn": "只在相同道路模块内打乱风格向量，观察收益是否仍存在。",
                "expected_if_style_real_cn": "真实风格收益应在道路平衡后仍保留一部分。",
            },
            {
                "control_name": "subject_id_baseline",
                "shuffle_scope": "not a shuffle",
                "purpose_cn": "判断连续风格是否只是驾驶员 ID 的替代品。",
                "implementation_rule_cn": "用驾驶员 ID 编码替代连续风格，与连续风格模型同评价协议比较。",
                "expected_if_style_real_cn": "连续风格应至少在部分物理指标/困难样本上超过或补充 ID。",
            },
        ]
    )

    availability = (
        long_df.groupby(["window_id", "style_window_status"], dropna=False)
        .size()
        .reset_index(name="sample_count")
    )
    window_total = long_df.groupby("window_id").size().rename("window_total")
    availability = availability.merge(window_total, on="window_id", how="left")
    availability["sample_rate"] = availability["sample_count"] / availability["window_total"]

    split_feasibility = (
        wide.groupby(["session_level_split"], dropna=False)
        .agg(
            sample_count=("sample_id", "size"),
            subject_count=("subject", "nunique"),
            session_count=("session_stamp", "nunique"),
            any_style_usable=("style_any_window_usable", "sum"),
            all_style_usable=("style_all_windows_usable", "sum"),
        )
        .reset_index()
    )
    split_feasibility["any_style_usable_rate"] = (
        split_feasibility["any_style_usable"] / split_feasibility["sample_count"]
    )
    split_feasibility["all_style_usable_rate"] = (
        split_feasibility["all_style_usable"] / split_feasibility["sample_count"]
    )

    subject_feasibility = (
        wide.groupby(["subject", "session_level_split"], dropna=False)
        .agg(
            sample_count=("sample_id", "size"),
            session_count=("session_stamp", "nunique"),
            road_module_count=("road_design_module_name", "nunique"),
            any_style_usable=("style_any_window_usable", "sum"),
        )
        .reset_index()
    )
    subject_feasibility["any_style_usable_rate"] = (
        subject_feasibility["any_style_usable"] / subject_feasibility["sample_count"]
    )

    road_counts = (
        wide.groupby(["subject", "road_design_module_name"], dropna=False)
        .size()
        .reset_index(name="sample_count")
    )
    road_coupling_rows = []
    for subject, sub in road_counts.groupby("subject"):
        counts = sub["sample_count"].to_numpy(dtype=float)
        total = float(counts.sum())
        probs = counts / total if total else counts
        entropy = float(-(probs * np.log2(np.where(probs > 0, probs, 1.0))).sum()) if total else np.nan
        max_entropy = math.log2(len(probs)) if len(probs) > 1 else 0.0
        road_coupling_rows.append(
            {
                "subject": subject,
                "sample_count": int(total),
                "road_module_count": int(len(probs)),
                "top_road_share": float(probs.max()) if len(probs) else np.nan,
                "road_entropy": entropy,
                "road_entropy_norm": float(entropy / max_entropy) if max_entropy > 0 else np.nan,
                "top_road_module": sub.sort_values("sample_count", ascending=False)[
                    "road_design_module_name"
                ].iloc[0],
            }
        )
    road_coupling = pd.DataFrame(road_coupling_rows).sort_values(
        ["top_road_share", "sample_count"], ascending=[False, False]
    )

    split_road_distribution = (
        wide.groupby(["session_level_split", "road_design_module_name"], dropna=False)
        .size()
        .reset_index(name="sample_count")
    )
    split_totals = split_road_distribution.groupby("session_level_split")["sample_count"].transform("sum")
    split_road_distribution["split_share"] = split_road_distribution["sample_count"] / split_totals

    gate_rows = [
        {
            "gate": "style_feature_source_defined",
            "status": "pass_protocol",
            "evidence": "source_protocol_table + leakage_guard_table",
            "decision_cn": "已定义事件前车辆历史风格来源，且排除直接输入和标签未来。",
        },
        {
            "gate": "style_candidate_features_extracted",
            "status": "pass" if wide["style_any_window_usable"].any() else "fail",
            "evidence": "style_feature_candidate_long/wide",
            "decision_cn": "已生成候选风格特征表；样本可用性见 split/subject feasibility。",
        },
        {
            "gate": "train_only_standardization_ready",
            "status": "pass_protocol",
            "evidence": "style_train_only_scaler_session_split.csv",
            "decision_cn": "标准化只允许用 session split 的训练集拟合。",
        },
        {
            "gate": "permutation_controls_defined",
            "status": "pass_protocol",
            "evidence": "style_permutation_plan.csv",
            "decision_cn": "已定义被试内、跨被试、跨 session、道路平衡和 ID 对照。",
        },
        {
            "gate": "style_effectiveness_claim_allowed",
            "status": "blocked",
            "evidence": "no model/permutation result yet",
            "decision_cn": "还没有与 RBF 固定参照、置乱和分被试验证比较，不能说风格有效。",
        },
        {
            "gate": "stage05_physio_eeg_allowed",
            "status": "blocked",
            "evidence": "style baseline not validated yet",
            "decision_cn": "生理/脑电继续阻塞，直到车辆+风格参照完成公平验证。",
        },
    ]
    gate_table = pd.DataFrame(gate_rows)

    return {
        "style_source_protocol_table": source_protocol,
        "style_leakage_guard_table": leakage_guard,
        "style_permutation_plan": permutation_plan,
        "style_feature_availability_by_window": availability,
        "style_split_feasibility": split_feasibility,
        "style_subject_split_feasibility": subject_feasibility,
        "style_subject_road_coupling_audit": road_coupling,
        "style_split_road_distribution": split_road_distribution,
        "style_protocol_gate_table": gate_table,
    }


def make_figures(tables: Dict[str, pd.DataFrame], wide: pd.DataFrame) -> None:
    availability = tables["style_feature_availability_by_window"]
    usable = availability[availability["style_window_status"].eq("usable_no_future_leakage")].copy()
    all_windows = [spec["window_id"] for spec in STYLE_WINDOWS]
    usable = usable.set_index("window_id").reindex(all_windows).reset_index()
    usable["sample_rate"] = usable["sample_rate"].fillna(0.0)

    plt.figure(figsize=(9, 4.8))
    plt.bar(usable["window_id"], usable["sample_rate"], color="#3b82f6")
    plt.ylim(0, 1.05)
    plt.ylabel("Usable sample rate")
    plt.xlabel("Style window")
    plt.title("Stage 4 style feature availability")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "style_feature_availability_by_window.png", dpi=180)
    plt.close()

    split_dist = tables["style_split_road_distribution"]
    pivot = split_dist.pivot_table(
        index="session_level_split",
        columns="road_design_module_name",
        values="sample_count",
        aggfunc="sum",
        fill_value=0,
    )
    if not pivot.empty:
        plt.figure(figsize=(max(8, 0.7 * len(pivot.columns)), 4.5))
        plt.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="Blues")
        plt.colorbar(label="Sample count")
        plt.yticks(range(len(pivot.index)), pivot.index)
        plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=35, ha="right")
        plt.title("Road module distribution by split")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "style_split_road_distribution_heatmap.png", dpi=180)
        plt.close()

    subject_road = (
        wide.groupby(["subject", "road_design_module_name"], dropna=False)
        .size()
        .reset_index(name="sample_count")
    )
    pivot_sr = subject_road.pivot_table(
        index="subject",
        columns="road_design_module_name",
        values="sample_count",
        aggfunc="sum",
        fill_value=0,
    )
    if not pivot_sr.empty:
        plt.figure(figsize=(max(8, 0.7 * len(pivot_sr.columns)), max(5, 0.28 * len(pivot_sr.index))))
        plt.imshow(pivot_sr.to_numpy(dtype=float), aspect="auto", cmap="YlOrRd")
        plt.colorbar(label="Sample count")
        plt.yticks(range(len(pivot_sr.index)), pivot_sr.index)
        plt.xticks(range(len(pivot_sr.columns)), pivot_sr.columns, rotation=35, ha="right")
        plt.title("Subject-road coupling audit")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "style_subject_road_coupling_heatmap.png", dpi=180)
        plt.close()


def format_rate(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    return f"{value:.1%}"


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]

    def clean_cell(value: object) -> str:
        if pd.isna(value):
            return ""
        text = str(value)
        return text.replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(clean_cell(row[c]) for c in df.columns) + " |")
    return "\n".join(lines)


def make_reports(
    manifest: pd.DataFrame,
    long_df: pd.DataFrame,
    wide: pd.DataFrame,
    scaler: pd.DataFrame,
    tables: Dict[str, pd.DataFrame],
    run_summary: Dict[str, object],
) -> None:
    split = tables["style_split_feasibility"]
    availability = tables["style_feature_availability_by_window"]
    usable = availability[availability["style_window_status"].eq("usable_no_future_leakage")][
        ["window_id", "sample_count", "sample_rate"]
    ]
    usable_lines = []
    for _, row in usable.iterrows():
        usable_lines.append(
            f"- `{row['window_id']}`：{int(row['sample_count'])}/{manifest.shape[0]} 可用，比例 {format_rate(float(row['sample_rate']))}"
        )
    if not usable_lines:
        usable_lines.append("- 没有窗口达到可用标准，需要回到样本或时间窗规则检查。")

    split_lines = []
    for _, row in split.iterrows():
        split_lines.append(
            f"- `{row['session_level_split']}`：样本 {int(row['sample_count'])}，被试 {int(row['subject_count'])}，session {int(row['session_count'])}，至少一个风格窗口可用 {int(row['any_style_usable'])} ({format_rate(float(row['any_style_usable_rate']))})"
        )

    scaler_ok = int(scaler["scaler_status"].eq("usable").sum()) if not scaler.empty else 0
    scaler_total = int(len(scaler))

    user_report = f"""# 阶段 4 用户查看版总结：连续驾驶风格协议与候选特征 v0.1

生成时间：{run_summary['run_time']}

## 这个阶段为什么做

阶段 3 已经把 RBF/KRR 车辆-only 模型固定成“有限主参照”。它可以作为后续比较底线，但错侧、反向修正和困难样本还没有解决。所以阶段 4 不能直接说连续风格有效，必须先把风格特征的来源、泄漏边界、标准化方式和置乱对照讲清楚。

## 这个阶段检查了什么

本轮只处理车辆原始数据中的事件前连续历史。主规则是：风格窗口最晚只能到事件锚点前 3 秒，也就是排除 `[-3, 0]` 的直接车辆输入窗口，并完全不接触 `[0, 3]` 的方向盘响应标签窗口。

同时生成了：

- 候选风格特征表；
- train-only 标准化参数；
- 道路/被试耦合审计；
- 被试内、跨被试、跨 session、道路平衡置乱和驾驶员 ID 对照协议。

## 目前发现了什么

本轮纳入 B 轨道严格核心样本 {manifest.shape[0]} 个。候选窗口可用性如下：

{chr(10).join(usable_lines)}

按 session-level split 看：

{chr(10).join(split_lines)}

train-only 标准化已准备：{scaler_ok}/{scaler_total} 个数值特征有可用训练集均值和标准差。

## 哪些结果可信

可信的是“处理规则”和“候选特征表”：这些特征只来自事件前 3 秒以前的原始车辆历史；标准化参数只从训练集拟合；脚本没有读取服务器密码，也没有使用生理或脑电数据。

## 哪些结果还不能下结论

现在还不能说连续驾驶风格有效。原因是还没有把这些风格特征接入固定 RBF 参照后的模型，也没有完成置乱对照、分被试验证、分道路验证和物理错误指标比较。

## 下一阶段是否可以继续

可以继续做阶段 4 的探索性验证，但只能按固定 RBF 主参照比较，并必须报告置乱对照和物理指标。生理/脑电仍然不能进入有效性结论。

## 推荐优先查看

- `04_style/stage04_continuous_style_protocol_v0_1/tables/style_protocol_gate_table.csv`
- `04_style/stage04_continuous_style_protocol_v0_1/tables/style_feature_candidate_wide.csv`
- `04_style/stage04_continuous_style_protocol_v0_1/tables/style_train_only_scaler_session_split.csv`
- `04_style/stage04_continuous_style_protocol_v0_1/figures/style_feature_availability_by_window.png`
- `04_style/stage04_continuous_style_protocol_v0_1/figures/style_subject_road_coupling_heatmap.png`
"""

    technical_report = f"""# 阶段 4 连续驾驶风格协议与候选特征处理 v0.1

生成时间：{run_summary['run_time']}

## 输入

- 样本清单：`{MANIFEST_PATH}`
- 样本筛选：`window_config_id == {WINDOW_CONFIG_ID}` 且 `task_sample_role == {TASK_ROLE}`
- 原始车辆文件：来自 manifest 的 `vehicle_raw_absolute_path`

## 无泄漏定义

- 车辆直接输入窗口：`[-3, 0]` 秒，属于车辆-only 基线输入，不作为连续风格；
- 标签窗口：`[0, 3]` 秒，完全禁止进入输入、风格、标准化和任何特征拟合；
- 连续风格候选窗口：`prefix_until_guard3`、`last120_guard3`、`last60_guard3`、`last30_guard3`，全部要求 `end <= anchor - 3s`；
- 标准化：只用 `session_level_split=train` 的候选特征拟合均值和标准差。

## 产物

- `tables/style_feature_candidate_long.csv`：一行一个样本-风格窗口；
- `tables/style_feature_candidate_wide.csv`：一行一个样本，适合后续建模；
- `tables/style_feature_candidate_wide_trainz_session_split.csv`：按 session train-only 统计标准化后的候选特征；
- `tables/style_train_only_scaler_session_split.csv`：训练集均值/标准差；
- `tables/style_source_protocol_table.csv`：风格来源允许/阻塞规则；
- `tables/style_leakage_guard_table.csv`：泄漏边界检查；
- `tables/style_permutation_plan.csv`：置乱和 ID 对照协议；
- `tables/style_subject_road_coupling_audit.csv`：被试-道路耦合审计；
- `tables/style_protocol_gate_table.csv`：是否允许进入下一步的 gate。

## 关键数量

- B 轨道严格核心样本数：{manifest.shape[0]}
- long 表行数：{long_df.shape[0]}
- wide 表列数：{wide.shape[1]}
- train-only 可标准化数值特征：{scaler_ok}/{scaler_total}

## Gate 结论

{dataframe_to_markdown(tables['style_protocol_gate_table'])}

## 当前限制

本轮没有训练模型，也没有评估风格增量。连续风格是否有效，必须在下一步接入固定 RBF 参照后，通过原始风格、置乱风格、驾驶员 ID 对照、分被试/分 session 和物理指标共同判断。
"""

    write_text(REPORT_DIR / "stage04_continuous_style_protocol_user_summary_cn.md", user_report)
    write_text(REPORT_DIR / "stage04_user_summary_cn.md", user_report)
    write_text(REPORT_DIR / "stage04_continuous_style_protocol_v0_1_cn.md", technical_report)


def main() -> None:
    ensure_dirs()
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    manifest_all = read_csv(MANIFEST_PATH)
    manifest = manifest_all[
        manifest_all["window_config_id"].eq(WINDOW_CONFIG_ID)
        & manifest_all["task_sample_role"].eq(TASK_ROLE)
    ].copy()
    manifest = manifest.reset_index(drop=True)
    if manifest.empty:
        raise RuntimeError("No B-track strict core samples found for stage 4 style protocol.")

    long_df, wide = make_style_feature_rows(manifest)
    z_wide, scaler = make_train_z_table(wide)
    tables = make_protocol_tables(manifest, long_df, wide)

    write_csv(long_df, TABLE_DIR / "style_feature_candidate_long.csv")
    write_csv(wide, TABLE_DIR / "style_feature_candidate_wide.csv")
    write_csv(z_wide, TABLE_DIR / "style_feature_candidate_wide_trainz_session_split.csv")
    write_csv(scaler, TABLE_DIR / "style_train_only_scaler_session_split.csv")
    for name, table in tables.items():
        write_csv(table, TABLE_DIR / f"{name}.csv")

    make_figures(tables, wide)

    run_summary = {
        "run_time": run_time,
        "script": str(Path(__file__).resolve()),
        "manifest_path": str(MANIFEST_PATH),
        "output_root": str(OUT_ROOT),
        "window_config_id": WINDOW_CONFIG_ID,
        "task_role": TASK_ROLE,
        "sample_count": int(manifest.shape[0]),
        "style_window_count": len(STYLE_WINDOWS),
        "long_rows": int(long_df.shape[0]),
        "wide_shape": list(wide.shape),
        "train_only_scaler_feature_count": int(len(scaler)),
        "train_only_scaler_usable_count": int(scaler["scaler_status"].eq("usable").sum()),
        "server_used": False,
        "server_credential_file_read": False,
        "effectiveness_claim": "blocked_no_model_or_permutation_result",
    }
    (LOG_DIR / "run_summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8-sig"
    )
    make_reports(manifest, long_df, wide, scaler, tables, run_summary)

    print(json.dumps(run_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
