# -*- coding: utf-8 -*-
"""
连续车辆源数据审计：只读扫描车辆 CSV，不生成样本、不训练模型、不修改标签。

审计思路按常见时序数据质量框架组织：
1. 资产盘点：哪些文件是真正车辆信号，哪些只是文件名/目录里带 vehicle。
2. 完整性：关键车辆字段、道路字段、时间字段是否缺失。
3. 一致性：同一记录在不同目录/变体之间行数、时长、采样间隔是否一致。
4. 时序质量：采样频率、重复时间戳、非单调时间、异常大间隔。
5. 分布质量：速度、方向盘、横向加速度、横摆/侧倾等变量的范围和尾部。
6. 对下游建模的含义：哪些问题会影响锚点、窗口和车辆-only 预测解释。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


ENCODINGS = ("utf-8-sig", "utf-8", "gbk", "latin1")

VEHICLE_HINT_COLS = [
    "zx|SteeringWheel",
    "zx|AcceleratorPedal",
    "zx|BrakePedal",
    "zx|ax",
    "zx|ay",
    "zx|vx",
    "zx|vy",
    "zx|vyaw",
    "zx|roll",
    "zx|pitch",
    "zx|yaw",
    "zx1|v_km/h",
    "zx1|lanecurvatureXY",
    "zx1|lateraldistance",
]

CANONICAL_HASH_COLS = [
    "zx|x",
    "zx|y",
    "zx|z",
    "zx|SteeringWheel",
    "zx|AcceleratorPedal",
    "zx|BrakePedal",
    "zx|ax",
    "zx|ay",
    "zx|vx",
    "zx|vy",
    "zx|vyaw",
    "zx|roll",
    "zx|pitch",
    "zx|yaw",
]

NUMERIC_AUDIT_COLS = [
    "zx1|v_km/h",
    "zx1|lanecurvatureXY",
    "zx1|mu",
    "zx1|lateraldistance",
    "zx1|distance7",
    "zx1|distance8",
    "zx1|pointdistance",
    "zx1|pointdistance9",
    "zx|aroll",
    "zx|apitch",
    "zx|ayaw",
    "zx|x",
    "zx|y",
    "zx|z",
    "zx|AcceleratorPedal",
    "zx|BrakePedal",
    "zx|SteeringWheel",
    "zx|ax",
    "zx|ay",
    "zx|vx",
    "zx|vy",
    "zx|vyaw",
    "zx|vroll",
    "zx|vpitch",
    "zx|roll",
    "zx|pitch",
    "zx|yaw",
    "road_s_ref_m",
    "ref_nn_dist_m",
    "ref_nn_ok",
]

ROAD_TYPE_COLS = ("road_type_fixed_str", "road_type_fixed")


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def find_project_roots(script_path: Path) -> Dict[str, Path]:
    rebuild_root = script_path.resolve().parents[2]
    project_root = rebuild_root.parent
    audit_root = rebuild_root / "01_audit" / "vehicle_source_audit_20260624"
    return {
        "project_root": project_root,
        "rebuild_root": rebuild_root,
        "main_vehicle_root": project_root / "01_datasets" / "多模态数据" / "被试数据集合",
        "supp_vehicle_root": project_root / "01_datasets" / "补充采集数据" / "车辆清理后",
        "audit_root": audit_root,
        "tables": audit_root / "tables",
        "logs": audit_root / "logs",
        "reports": audit_root / "reports",
        "figures": audit_root / "figures",
        "report_entry": rebuild_root / "09_reports" / "vehicle_source_audit_20260624_cn.md",
    }


def ensure_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def rel_to(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def read_header(path: Path) -> Tuple[List[str], str, Optional[str]]:
    for enc in ENCODINGS:
        try:
            df = pd.read_csv(path, nrows=0, encoding=enc)
            return list(df.columns), enc, None
        except Exception as exc:  # pragma: no cover - 运行期盘点保留错误文本
            last_error = f"{type(exc).__name__}: {exc}"
    return [], "", last_error


def read_csv(path: Path, encoding: str) -> pd.DataFrame:
    # low_memory=False 可以避免同一列前后类型被分段推断成不同 dtype。
    return pd.read_csv(path, encoding=encoding or None, low_memory=False)


def recording_key(path: Path) -> str:
    match = re.search(r"Entity_Recording_(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})", path.name)
    return match.group(1) if match else path.stem


def subject_from_path(path: Path, roots: Dict[str, Path]) -> str:
    try:
        rel = path.relative_to(roots["main_vehicle_root"])
        if len(rel.parts) >= 1:
            return rel.parts[0]
    except ValueError:
        pass
    try:
        rel = path.relative_to(roots["supp_vehicle_root"])
        if len(rel.parts) >= 1:
            return rel.parts[0]
    except ValueError:
        pass
    return "unknown"


def classify_variant(path: Path) -> str:
    name = path.name
    if "(PhysioLAB" in name:
        return "physio_named_variant"
    if " (2)_vehicle_fixed_200Hz" in name:
        return "vehicle_fixed_200hz_variant_2"
    if name.endswith("_vehicle_fixed_200Hz.csv"):
        return "plain_fixed_200hz_variant"
    if name.endswith("_vehicle_aligned_cleaned.csv"):
        return "main_aligned_cleaned"
    if "roadtype_labeled" in name:
        return "roadtype_labeled_aux"
    if "roadtype_segments" in name:
        return "roadtype_segments_aux"
    return "other"


def classify_file(path: Path, columns: Sequence[str], roots: Dict[str, Path]) -> Dict[str, object]:
    colset = set(columns)
    vehicle_hits = [c for c in VEHICLE_HINT_COLS if c in colset]
    physio_hits = [c for c in columns if "PhysioLAB" in c or "|CH1-ECG" in c or "|CH2-EMG" in c]
    eeg_hits = [c for c in columns if c.startswith("LSLOutletStreamName-EEG|")]
    accel_eeg_hits = [c for c in columns if c.startswith("LSLOutletStreamName-Accelerometer|")]

    source_group = "unknown"
    try:
        rel = path.relative_to(roots["main_vehicle_root"])
        if len(rel.parts) >= 2 and rel.parts[1] == "vehicle":
            source_group = "main_subject_vehicle_dir"
    except ValueError:
        pass
    try:
        path.relative_to(roots["supp_vehicle_root"])
        source_group = "supplement_vehicle_cleaned_dir"
    except ValueError:
        pass

    variant = classify_variant(path)
    is_vehicle_like = len(vehicle_hits) >= 4
    include = False
    file_class = "excluded"
    reason = ""

    if source_group == "main_subject_vehicle_dir":
        if path.name.endswith("_vehicle_aligned_cleaned.csv") and is_vehicle_like:
            include = True
            file_class = "main_vehicle_aligned_cleaned"
            reason = "主数据集中连续车辆清洗文件"
        elif is_vehicle_like:
            file_class = "main_vehicle_aux_or_derived"
            reason = "主车辆目录中的辅助/派生车辆文件，本轮不纳入连续源审计"
        else:
            file_class = "main_vehicle_dir_non_vehicle"
            reason = "主车辆目录中未检出足够车辆字段"
    elif source_group == "supplement_vehicle_cleaned_dir":
        if is_vehicle_like:
            include = True
            file_class = "supplement_vehicle_fixed_200hz"
            reason = "补充采集车辆清理后目录中含车辆字段的 200Hz 文件"
        elif physio_hits:
            file_class = "supplement_physio_in_vehicle_dir"
            reason = "文件名/目录含 vehicle，但字段是 PhysioLAB 生理信号"
        elif eeg_hits or accel_eeg_hits:
            file_class = "supplement_eeg_in_vehicle_dir"
            reason = "文件名/目录含 vehicle，但字段是 EEG/加速度信号"
        else:
            file_class = "supplement_unknown_non_vehicle"
            reason = "补充采集车辆目录中未检出车辆/生理/EEG 典型字段"
    else:
        if is_vehicle_like:
            file_class = "other_vehicle_like"
            reason = "其他目录中的车辆字段文件，本轮只做目录盘点"
        else:
            reason = "非本轮目标目录"

    return {
        "file_class": file_class,
        "include_in_continuous_vehicle_audit": include,
        "exclude_reason": reason,
        "source_group": source_group,
        "variant_kind": variant,
        "vehicle_hint_count": len(vehicle_hits),
        "physio_hint_count": len(physio_hits),
        "eeg_hint_count": len(eeg_hits),
        "accelerometer_hint_count": len(accel_eeg_hits),
        "column_count": len(columns),
    }


def discover_files(roots: Dict[str, Path]) -> List[Dict[str, object]]:
    candidates: List[Path] = []
    if roots["main_vehicle_root"].exists():
        candidates.extend(sorted(roots["main_vehicle_root"].glob("*/vehicle/*.csv")))
    if roots["supp_vehicle_root"].exists():
        candidates.extend(sorted(roots["supp_vehicle_root"].glob("*/*.csv")))

    rows: List[Dict[str, object]] = []
    for idx, path in enumerate(candidates, start=1):
        columns, enc, error = read_header(path)
        cls = classify_file(path, columns, roots)
        rows.append(
            {
                "candidate_file_id": f"F{idx:04d}",
                "rel_path": rel_to(path, roots["project_root"]),
                "abs_path": str(path.resolve()),
                "subject": subject_from_path(path, roots),
                "recording_key": recording_key(path),
                "file_name": path.name,
                "file_size_bytes": path.stat().st_size if path.exists() else "",
                "header_encoding": enc,
                "header_error": error or "",
                **cls,
            }
        )
    return rows


def finite_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def safe_quantile(values: pd.Series | np.ndarray, q: float) -> float:
    arr = np.asarray(values, dtype="float64")
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return math.nan
    return float(np.quantile(arr, q))


def safe_max_abs(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values, dtype="float64")
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return math.nan
    return float(np.max(np.abs(arr)))


def compute_elapsed_and_dt(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, str]:
    if "t_s" in df.columns:
        t = pd.to_numeric(df["t_s"], errors="coerce")
        return t, t.diff() * 1000.0, "t_s"

    if "StorageTime" in df.columns:
        parsed = pd.to_datetime(df["StorageTime"], errors="coerce", format="mixed")
        valid = parsed.notna()
        if valid.sum() >= 2:
            first = parsed[valid].iloc[0]
            elapsed = (parsed - first).dt.total_seconds()
            dt_ms = parsed.diff().dt.total_seconds() * 1000.0
            return elapsed, dt_ms, "StorageTime"

    if "dt_ms" in df.columns:
        dt_ms = pd.to_numeric(df["dt_ms"], errors="coerce")
        elapsed = dt_ms.fillna(0).cumsum() / 1000.0
        return elapsed, dt_ms, "dt_ms"

    return pd.Series(np.arange(len(df)), dtype="float64"), pd.Series(dtype="float64"), "row_index"


def sample_hash(df: pd.DataFrame) -> str:
    cols = [c for c in CANONICAL_HASH_COLS if c in df.columns]
    if not cols:
        return ""
    sample = pd.concat([df[cols].head(200), df[cols].tail(200)], axis=0)
    norm = pd.DataFrame(index=sample.index)
    for col in cols:
        s = pd.to_numeric(sample[col], errors="coerce").round(6)
        norm[col] = s.map(lambda x: "" if pd.isna(x) else f"{x:.6f}")
    text = norm.to_csv(index=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def numeric_stats_for_file(file_id: str, df: pd.DataFrame) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    total = max(1, len(df))
    for col in NUMERIC_AUDIT_COLS:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        arr = s.to_numpy(dtype="float64", copy=False)
        finite = arr[np.isfinite(arr)]
        rows.append(
            {
                "file_id": file_id,
                "column": col,
                "total_rows": len(df),
                "valid_count": int(finite.size),
                "missing_count": int(total - finite.size),
                "missing_rate": float(1.0 - finite.size / total),
                "min": float(np.min(finite)) if finite.size else "",
                "p01": safe_quantile(finite, 0.01) if finite.size else "",
                "p50": safe_quantile(finite, 0.50) if finite.size else "",
                "p99": safe_quantile(finite, 0.99) if finite.size else "",
                "max": float(np.max(finite)) if finite.size else "",
                "mean": float(np.mean(finite)) if finite.size else "",
                "std": float(np.std(finite)) if finite.size else "",
                "abs_p99": safe_quantile(np.abs(finite), 0.99) if finite.size else "",
                "abs_max": float(np.max(np.abs(finite))) if finite.size else "",
                "near_constant": bool(np.nanstd(finite) < 1.0e-9) if finite.size else True,
            }
        )
    return rows


def count_active_seconds(elapsed_s: pd.Series, df: pd.DataFrame) -> Dict[str, object]:
    if elapsed_s.empty or elapsed_s.notna().sum() < 2:
        return {
            "drive_seconds_speed_gt_5": "",
            "lateral_action_seconds": "",
            "steer_action_seconds": "",
            "ay_action_seconds": "",
        }

    tmp = pd.DataFrame({"sec": np.floor(pd.to_numeric(elapsed_s, errors="coerce"))})
    tmp = tmp[np.isfinite(tmp["sec"])]
    if tmp.empty:
        return {
            "drive_seconds_speed_gt_5": "",
            "lateral_action_seconds": "",
            "steer_action_seconds": "",
            "ay_action_seconds": "",
        }

    speed = finite_series(df, "zx1|v_km/h")
    steer = finite_series(df, "zx|SteeringWheel").abs()
    ay = finite_series(df, "zx|ay").abs()

    tmp["speed_gt_5"] = (speed.reindex(tmp.index).fillna(0) > 5.0).astype(int)
    tmp["steer_action"] = (steer.reindex(tmp.index).fillna(0) > 0.20).astype(int)
    tmp["ay_action"] = (ay.reindex(tmp.index).fillna(0) > 0.50).astype(int)
    sec = tmp.groupby("sec", sort=False).max(numeric_only=True)
    lateral = ((sec["steer_action"] > 0) | (sec["ay_action"] > 0)).sum()
    return {
        "drive_seconds_speed_gt_5": int((sec["speed_gt_5"] > 0).sum()),
        "lateral_action_seconds": int(lateral),
        "steer_action_seconds": int((sec["steer_action"] > 0).sum()),
        "ay_action_seconds": int((sec["ay_action"] > 0).sum()),
    }


def road_type_rows(file_id: str, df: pd.DataFrame) -> List[Dict[str, object]]:
    road_col = ""
    for col in ROAD_TYPE_COLS:
        if col in df.columns:
            road_col = col
            break
    if not road_col:
        return []
    counts = df[road_col].astype("string").fillna("__MISSING__").value_counts(dropna=False)
    total = max(1, len(df))
    return [
        {
            "file_id": file_id,
            "road_type_column": road_col,
            "road_type": str(k),
            "rows": int(v),
            "row_rate": float(v / total),
        }
        for k, v in counts.items()
    ]


def audit_one_file(row: Dict[str, object], roots: Dict[str, Path]) -> Tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]]]:
    path = Path(str(row["abs_path"]))
    df = read_csv(path, str(row["header_encoding"]))
    file_id = str(row["candidate_file_id"])

    elapsed_s, dt_ms, time_source = compute_elapsed_and_dt(df)
    dt_valid = pd.to_numeric(dt_ms, errors="coerce")
    dt_valid = dt_valid[np.isfinite(dt_valid)]
    dt_valid_no_first = dt_valid.iloc[1:] if len(dt_valid) > 1 else dt_valid

    duration_s = float(elapsed_s.max() - elapsed_s.min()) if elapsed_s.notna().sum() >= 2 else math.nan
    nominal_hz = float((len(df) - 1) / duration_s) if duration_s and np.isfinite(duration_s) and duration_s > 0 else math.nan

    duplicate_ts = 0
    if "StorageTime" in df.columns:
        duplicate_ts = int(df["StorageTime"].duplicated().sum())
    nonmono = int((dt_valid_no_first <= 0).sum()) if len(dt_valid_no_first) else 0
    gap20 = int((dt_valid_no_first > 20.0).sum()) if len(dt_valid_no_first) else 0
    gap50 = int((dt_valid_no_first > 50.0).sum()) if len(dt_valid_no_first) else 0

    key_present = [c for c in VEHICLE_HINT_COLS if c in df.columns]
    key_missing_rates = []
    all_null_cols = []
    for col in key_present:
        miss = float(pd.to_numeric(df[col], errors="coerce").isna().mean())
        key_missing_rates.append(miss)
        if miss >= 0.999:
            all_null_cols.append(col)

    ref_ok_rate = ""
    ref_dist_p95 = ""
    if "ref_nn_ok" in df.columns:
        ref_ok = pd.to_numeric(df["ref_nn_ok"], errors="coerce")
        ref_ok_rate = float((ref_ok == 1).mean())
    if "ref_nn_dist_m" in df.columns:
        ref_dist = pd.to_numeric(df["ref_nn_dist_m"], errors="coerce")
        ref_dist_p95 = safe_quantile(ref_dist, 0.95)

    first_storage = str(df["StorageTime"].iloc[0]) if "StorageTime" in df.columns and len(df) else ""
    last_storage = str(df["StorageTime"].iloc[-1]) if "StorageTime" in df.columns and len(df) else ""

    speed = finite_series(df, "zx1|v_km/h")
    steer = finite_series(df, "zx|SteeringWheel")
    ay = finite_series(df, "zx|ay")
    brake = finite_series(df, "zx|BrakePedal")
    accel = finite_series(df, "zx|AcceleratorPedal")
    yaw_rate = finite_series(df, "zx|vyaw")
    roll = finite_series(df, "zx|roll")
    lateral_dist = finite_series(df, "zx1|lateraldistance")

    flags: List[str] = []
    if len(df) == 0:
        flags.append("empty_file")
    if time_source not in {"t_s", "StorageTime", "dt_ms"}:
        flags.append("weak_time_axis")
    if np.isfinite(nominal_hz) and (nominal_hz < 150.0 or nominal_hz > 250.0):
        flags.append("nominal_hz_outside_150_250")
    med_dt = safe_quantile(dt_valid_no_first, 0.50) if len(dt_valid_no_first) else math.nan
    if np.isfinite(med_dt) and abs(med_dt - 5.0) > 1.0:
        flags.append("median_dt_not_near_5ms")
    if gap50 > 0:
        flags.append("gap_gt_50ms")
    elif gap20 > 0:
        flags.append("gap_gt_20ms")
    if duplicate_ts > 0:
        flags.append("duplicate_storage_time")
    if nonmono > 0:
        flags.append("nonmonotonic_time")
    if key_missing_rates and float(np.mean(key_missing_rates)) > 0.20:
        flags.append("high_key_vehicle_missing_rate")
    if all_null_cols:
        flags.append("all_null_key_vehicle_columns")
    if ref_ok_rate != "" and float(ref_ok_rate) < 0.95:
        flags.append("low_road_reference_ok_rate")
    if speed.notna().sum() and (safe_quantile(speed, 0.99) > 180.0 or safe_quantile(speed, 0.01) < -1.0):
        flags.append("speed_tail_needs_review")
    if ay.notna().sum() and safe_quantile(ay.abs(), 0.999) > 15.0:
        flags.append("lateral_acc_tail_needs_review")
    if lateral_dist.notna().sum() and safe_quantile(lateral_dist.abs(), 0.999) > 5.0:
        flags.append("lateral_distance_tail_needs_reference_review")

    active = count_active_seconds(elapsed_s, df)
    summary = {
        "file_id": file_id,
        "source_layer": row["file_class"],
        "source_group": row["source_group"],
        "variant_kind": row["variant_kind"],
        "subject": row["subject"],
        "recording_key": row["recording_key"],
        "rel_path": row["rel_path"],
        "file_size_bytes": row["file_size_bytes"],
        "encoding": row["header_encoding"],
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "first_storage_time": first_storage,
        "last_storage_time": last_storage,
        "time_axis_source": time_source,
        "duration_s": duration_s,
        "nominal_hz": nominal_hz,
        "median_dt_ms": med_dt,
        "p95_dt_ms": safe_quantile(dt_valid_no_first, 0.95) if len(dt_valid_no_first) else math.nan,
        "p99_dt_ms": safe_quantile(dt_valid_no_first, 0.99) if len(dt_valid_no_first) else math.nan,
        "max_dt_ms": float(np.max(dt_valid_no_first)) if len(dt_valid_no_first) else math.nan,
        "duplicate_storage_time_count": duplicate_ts,
        "nonmonotonic_dt_count": nonmono,
        "gap_gt_20ms_count": gap20,
        "gap_gt_50ms_count": gap50,
        "vehicle_hint_columns_present": len(key_present),
        "vehicle_key_missing_rate_mean": float(np.mean(key_missing_rates)) if key_missing_rates else "",
        "all_null_key_columns": "|".join(all_null_cols),
        "ref_nn_ok_rate": ref_ok_rate,
        "ref_nn_dist_m_p95": ref_dist_p95,
        "speed_kmh_p99": safe_quantile(speed, 0.99) if speed.notna().sum() else "",
        "speed_kmh_max": float(np.nanmax(speed)) if speed.notna().sum() else "",
        "abs_steer_p99": safe_quantile(steer.abs(), 0.99) if steer.notna().sum() else "",
        "abs_steer_max": safe_max_abs(steer) if steer.notna().sum() else "",
        "abs_ay_p99": safe_quantile(ay.abs(), 0.99) if ay.notna().sum() else "",
        "abs_ay_max": safe_max_abs(ay) if ay.notna().sum() else "",
        "abs_yaw_rate_p99": safe_quantile(yaw_rate.abs(), 0.99) if yaw_rate.notna().sum() else "",
        "abs_roll_p99": safe_quantile(roll.abs(), 0.99) if roll.notna().sum() else "",
        "brake_min": float(np.nanmin(brake)) if brake.notna().sum() else "",
        "brake_max": float(np.nanmax(brake)) if brake.notna().sum() else "",
        "accelerator_p99": safe_quantile(accel, 0.99) if accel.notna().sum() else "",
        "lateral_distance_abs_p99": safe_quantile(lateral_dist.abs(), 0.99) if lateral_dist.notna().sum() else "",
        "canonical_vehicle_sample_hash": sample_hash(df),
        "suspect_flag_count": len(flags),
        "suspect_flags": ";".join(flags),
        **active,
    }
    return summary, numeric_stats_for_file(file_id, df), road_type_rows(file_id, df)


def build_cluster_summary(file_summary: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (subject, rec), g in file_summary.groupby(["subject", "recording_key"], dropna=False):
        layers = sorted(set(g["source_layer"].astype(str)))
        variants = sorted(set(g["variant_kind"].astype(str)))
        rows_count = pd.to_numeric(g["rows"], errors="coerce")
        durations = pd.to_numeric(g["duration_s"], errors="coerce")
        hashes = set(x for x in g["canonical_vehicle_sample_hash"].astype(str) if x)
        flags: List[str] = []
        if len(layers) == 1:
            flags.append("single_source_layer")
        if rows_count.max() - rows_count.min() > 2:
            flags.append("row_count_mismatch")
        if durations.notna().sum() >= 2 and durations.max() - durations.min() > 0.10:
            flags.append("duration_mismatch_gt_0p1s")
        if len(hashes) > 1:
            flags.append("canonical_signal_hash_mismatch")
        rows.append(
            {
                "subject": subject,
                "recording_key": rec,
                "included_file_count": int(len(g)),
                "source_layers": "|".join(layers),
                "variant_kinds": "|".join(variants),
                "rows_min": int(rows_count.min()) if rows_count.notna().any() else "",
                "rows_max": int(rows_count.max()) if rows_count.notna().any() else "",
                "duration_s_min": float(durations.min()) if durations.notna().any() else "",
                "duration_s_max": float(durations.max()) if durations.notna().any() else "",
                "hash_unique_count": int(len(hashes)),
                "cluster_flag_count": int(len(flags)),
                "cluster_flags": ";".join(flags),
                "file_ids": "|".join(g["file_id"].astype(str)),
            }
        )
    return pd.DataFrame(rows).sort_values(["cluster_flag_count", "subject", "recording_key"], ascending=[False, True, True])


def build_subject_summary(file_summary: pd.DataFrame) -> pd.DataFrame:
    agg_rows: List[Dict[str, object]] = []
    for subject, g in file_summary.groupby("subject", dropna=False):
        duration = pd.to_numeric(g["duration_s"], errors="coerce")
        rows_count = pd.to_numeric(g["rows"], errors="coerce")
        flags = [f for text in g["suspect_flags"].astype(str) for f in text.split(";") if f]
        agg_rows.append(
            {
                "subject": subject,
                "included_file_count": int(len(g)),
                "recording_count": int(g["recording_key"].nunique()),
                "total_rows": int(rows_count.sum()),
                "total_duration_min": float(duration.sum() / 60.0),
                "median_duration_s": float(duration.median()) if duration.notna().any() else "",
                "median_dt_ms_median": float(pd.to_numeric(g["median_dt_ms"], errors="coerce").median()),
                "gap_gt_50ms_files": int((pd.to_numeric(g["gap_gt_50ms_count"], errors="coerce") > 0).sum()),
                "suspect_file_count": int((pd.to_numeric(g["suspect_flag_count"], errors="coerce") > 0).sum()),
                "drive_seconds_speed_gt_5": int(pd.to_numeric(g["drive_seconds_speed_gt_5"], errors="coerce").fillna(0).sum()),
                "lateral_action_seconds": int(pd.to_numeric(g["lateral_action_seconds"], errors="coerce").fillna(0).sum()),
                "top_suspect_flags": "|".join([k for k, _ in Counter(flags).most_common(5)]),
            }
        )
    return pd.DataFrame(agg_rows).sort_values(["suspect_file_count", "included_file_count"], ascending=[False, False])


def build_source_layer_summary(file_summary: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if file_summary.empty:
        return pd.DataFrame()
    for source_layer, g in file_summary.groupby("source_layer", dropna=False):
        flags = [f for text in g["suspect_flags"].astype(str) for f in text.split(";") if f]
        rows.append(
            {
                "source_layer": source_layer,
                "file_count": int(len(g)),
                "subject_count": int(g["subject"].nunique()),
                "recording_count": int(g["recording_key"].nunique()),
                "total_duration_h": float(pd.to_numeric(g["duration_s"], errors="coerce").sum() / 3600.0),
                "median_rows": float(pd.to_numeric(g["rows"], errors="coerce").median()),
                "median_dt_ms": float(pd.to_numeric(g["median_dt_ms"], errors="coerce").median()),
                "median_nominal_hz": float(pd.to_numeric(g["nominal_hz"], errors="coerce").median()),
                "files_nominal_hz_outside_150_250": int(
                    g["suspect_flags"].astype(str).str.contains("nominal_hz_outside_150_250").sum()
                ),
                "files_median_dt_not_near_5ms": int(
                    g["suspect_flags"].astype(str).str.contains("median_dt_not_near_5ms").sum()
                ),
                "files_gap_gt_50ms": int((pd.to_numeric(g["gap_gt_50ms_count"], errors="coerce") > 0).sum()),
                "files_high_key_missing_rate": int(
                    (pd.to_numeric(g["vehicle_key_missing_rate_mean"], errors="coerce") > 0.20).sum()
                ),
                "files_low_road_ref_ok_rate": int((pd.to_numeric(g["ref_nn_ok_rate"], errors="coerce") < 0.95).sum()),
                "top_suspect_flags": "|".join([k for k, _ in Counter(flags).most_common(8)]),
            }
        )
    return pd.DataFrame(rows).sort_values("source_layer")


def write_figures(file_summary: pd.DataFrame, subject_summary: pd.DataFrame, roots: Dict[str, Path]) -> List[str]:
    created: List[str] = []
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return created

    fig_dir = roots["figures"]
    duration = pd.to_numeric(file_summary["duration_s"], errors="coerce").dropna()
    if not duration.empty:
        plt.figure(figsize=(8, 4.5))
        plt.hist(duration / 60.0, bins=30, color="#4575b4", edgecolor="white")
        plt.xlabel("duration_min")
        plt.ylabel("file_count")
        plt.title("Vehicle recording duration distribution")
        out = fig_dir / "vehicle_recording_duration_hist.png"
        plt.tight_layout()
        plt.savefig(out, dpi=160)
        plt.close()
        created.append(rel_to(out, roots["project_root"]))

    if not subject_summary.empty:
        top = subject_summary.sort_values("total_duration_min", ascending=False)
        plt.figure(figsize=(10, 5))
        plt.bar(top["subject"], top["total_duration_min"], color="#4d9221")
        plt.xticks(rotation=60, ha="right")
        plt.ylabel("total_duration_min")
        plt.title("Total vehicle duration by subject")
        out = fig_dir / "vehicle_duration_by_subject.png"
        plt.tight_layout()
        plt.savefig(out, dpi=160)
        plt.close()
        created.append(rel_to(out, roots["project_root"]))

    flags = [f for text in file_summary["suspect_flags"].astype(str) for f in text.split(";") if f]
    if flags:
        cnt = Counter(flags).most_common(12)
        labels, values = zip(*cnt)
        plt.figure(figsize=(10, 5))
        plt.bar(labels, values, color="#b2182b")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("file_count")
        plt.title("Top source vehicle audit flags")
        out = fig_dir / "vehicle_audit_flag_counts.png"
        plt.tight_layout()
        plt.savefig(out, dpi=160)
        plt.close()
        created.append(rel_to(out, roots["project_root"]))

    return created


def pct(x: float | int | str, digits: int = 1) -> str:
    try:
        if x == "":
            return ""
        return f"{float(x) * 100:.{digits}f}%"
    except Exception:
        return ""


def num(x: float | int | str, digits: int = 3) -> str:
    try:
        if x == "":
            return ""
        if not np.isfinite(float(x)):
            return "NA"
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def build_findings(
    inventory: pd.DataFrame,
    file_summary: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    subject_summary: pd.DataFrame,
    source_layer_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    inv_class = inventory["file_class"].value_counts().to_dict()
    included = inventory[inventory["include_in_continuous_vehicle_audit"] == True]  # noqa: E712
    non_vehicle_in_supp = inventory[
        inventory["file_class"].isin(["supplement_physio_in_vehicle_dir", "supplement_eeg_in_vehicle_dir"])
    ]

    rows.append(
        {
            "severity": "P1",
            "finding": "补充采集的“车辆清理后”目录存在明显命名混杂",
            "evidence": (
                f"该目录/文件名里带 vehicle，但检出非车辆字段文件 {len(non_vehicle_in_supp)} 个；"
                f"其中 PhysioLAB={inv_class.get('supplement_physio_in_vehicle_dir', 0)}，"
                f"EEG/加速度={inv_class.get('supplement_eeg_in_vehicle_dir', 0)}。"
            ),
            "impact": "如果脚本只按文件名 glob *_vehicle_fixed_200Hz.csv，会把生理/EEG 文件误当车辆源，导致字段缺失或错配。",
            "next_check": "后续读取补充采集数据时必须按字段白名单判定车辆源，而不是按目录/文件名判定。",
        }
    )

    if not file_summary.empty:
        total_duration_h = pd.to_numeric(file_summary["duration_s"], errors="coerce").sum() / 3600.0
        rows.append(
            {
                "severity": "P2",
                "finding": "连续车辆源文件规模足够做源级统计，而不是只看样本集",
                "evidence": (
                    f"纳入连续车辆审计 {len(file_summary)} 个文件，"
                    f"{file_summary['subject'].nunique()} 名被试，"
                    f"{file_summary['recording_key'].nunique()} 个记录键，"
                    f"总时长约 {total_duration_h:.2f} 小时。"
                ),
                "impact": "可以直接在源记录层面检查采样、缺失、道路标定和动作分布，再回头解释样本/窗口问题。",
                "next_check": "把源级异常记录映射回后续样本锚点，检查坏样本是否集中来自少数源记录。",
            }
        )

        gap50_files = int((pd.to_numeric(file_summary["gap_gt_50ms_count"], errors="coerce") > 0).sum())
        nonmono_files = int((pd.to_numeric(file_summary["nonmonotonic_dt_count"], errors="coerce") > 0).sum())
        median_dt = pd.to_numeric(file_summary["median_dt_ms"], errors="coerce").median()
        hz_outside = int(file_summary["suspect_flags"].astype(str).str.contains("nominal_hz_outside_150_250").sum())
        med_dt_bad = int(file_summary["suspect_flags"].astype(str).str.contains("median_dt_not_near_5ms").sum())
        rows.append(
            {
                "severity": "P2" if hz_outside or med_dt_bad or gap50_files or nonmono_files else "P3",
                "finding": "时间戳没有大断裂，但补充层存在采样率/行数层级不一致",
                "evidence": (
                    f"文件级 median_dt_ms 的中位数为 {num(median_dt, 3)}；"
                    f"存在 gap>50ms 的文件 {gap50_files} 个，非单调时间文件 {nonmono_files} 个；"
                    f"nominal_hz 超出 150-250 的文件 {hz_outside} 个，median_dt 不接近 5ms 的文件 {med_dt_bad} 个。"
                ),
                "impact": "主车辆层可按 200Hz 使用；补充层有些记录更像重采样/插值后的不同时间轴，不能和主层直接混用。",
                "next_check": "优先看 source_layer_summary.csv 与 recording_cluster_summary.csv，确认后续唯一源层。",
            }
        )

        road_cols = file_summary[file_summary["ref_nn_ok_rate"] != ""].copy()
        if not road_cols.empty:
            low_road = road_cols[pd.to_numeric(road_cols["ref_nn_ok_rate"], errors="coerce") < 0.95]
            rows.append(
                {
                    "severity": "P2" if len(low_road) else "P3",
                    "finding": "主车辆层已有道路参考字段，但低覆盖记录会影响 road/curve 分层判断",
                    "evidence": f"有道路参考字段的文件 {len(road_cols)} 个，其中 ref_nn_ok_rate<95% 的文件 {len(low_road)} 个。",
                    "impact": "如果低覆盖记录进入 curve/road 分层或横向偏移筛选，会把道路参考误差混入行为判断。",
                    "next_check": "对低 ref_nn_ok_rate 或 ref_nn_dist_m_p95 偏高记录做地图/道路参考复核。",
                }
            )

        missing_mean = pd.to_numeric(file_summary["vehicle_key_missing_rate_mean"], errors="coerce")
        high_missing = int((missing_mean > 0.20).sum())
        main_high_missing = 0
        supp_high_missing = 0
        if not source_layer_summary.empty:
            row_map = source_layer_summary.set_index("source_layer").to_dict("index")
            main_high_missing = int(row_map.get("main_vehicle_aligned_cleaned", {}).get("files_high_key_missing_rate", 0))
            supp_high_missing = int(row_map.get("supplement_vehicle_fixed_200hz", {}).get("files_high_key_missing_rate", 0))
        rows.append(
            {
                "severity": "P1" if high_missing else "P3",
                "finding": "主车辆层字段完整，补充车辆层字段缺失较多，必须分层使用",
                "evidence": (
                    f"关键车辆字段平均缺失率>20% 的纳入文件 {high_missing} 个；"
                    f"其中主 vehicle_aligned_cleaned 层 {main_high_missing} 个，"
                    f"补充 vehicle_fixed_200Hz 层 {supp_high_missing} 个。"
                ),
                "impact": "主层更适合做当前车辆建模和道路分层；补充层若直接混入，会让 speed/road/steer/ay 联合判断静默退化。",
                "next_check": "训练/样本构建前固定唯一车辆源层；若必须用补充层，先重建字段完整性和采样率规则。",
            }
        )

    if not cluster_summary.empty:
        row_mismatch = int(cluster_summary["cluster_flags"].astype(str).str.contains("row_count_mismatch").sum())
        hash_mismatch = int(cluster_summary["cluster_flags"].astype(str).str.contains("canonical_signal_hash_mismatch").sum())
        both_layers = int(cluster_summary["source_layers"].astype(str).str.contains("main_vehicle_aligned_cleaned").sum())
        rows.append(
            {
                "severity": "P2" if row_mismatch or hash_mismatch else "P3",
                "finding": "同一记录在主目录和补充目录之间需要保留 lineage 对照",
                "evidence": (
                    f"记录簇 {len(cluster_summary)} 个；包含主车辆层的簇 {both_layers} 个；"
                    f"行数不一致簇 {row_mismatch} 个；规范车辆信号抽样哈希不一致簇 {hash_mismatch} 个。"
                ),
                "impact": "如果不同阶段混用 main aligned 和 supplement fixed 版本，可能出现同名记录但信号/行数不完全一致。",
                "next_check": "用 recording_cluster_summary.csv 决定后续唯一源层，并记录每个样本来自哪个源层。",
            }
        )

    if not subject_summary.empty:
        top = subject_summary.sort_values("total_duration_min", ascending=False).head(3)
        bottom = subject_summary.sort_values("total_duration_min", ascending=True).head(3)
        rows.append(
            {
                "severity": "P2",
                "finding": "被试层面的总时长和横向动作秒数不均衡",
                "evidence": (
                    "总时长最高："
                    + ", ".join(f"{r.subject}={float(r.total_duration_min):.1f}min" for r in top.itertuples())
                    + "；最低："
                    + ", ".join(f"{r.subject}={float(r.total_duration_min):.1f}min" for r in bottom.itertuples())
                    + "。"
                ),
                "impact": "车辆-only 模型若按随机样本切分，容易把被试/记录分布差异误当可泛化信号。",
                "next_check": "继续坚持 subject/session-level split，并检查难样本是否集中在动作秒数少或道路覆盖异常的被试。",
            }
        )

    return pd.DataFrame(rows)


def write_report(
    roots: Dict[str, Path],
    inventory: pd.DataFrame,
    file_summary: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    subject_summary: pd.DataFrame,
    source_layer_summary: pd.DataFrame,
    findings: pd.DataFrame,
    created_figures: Sequence[str],
) -> None:
    tables_dir = roots["tables"]
    report_path = roots["reports"] / "vehicle_source_audit_20260624_cn.md"
    entry_path = roots["report_entry"]

    class_counts = inventory["file_class"].value_counts().reset_index()
    class_counts.columns = ["file_class", "count"]
    included = inventory[inventory["include_in_continuous_vehicle_audit"] == True]  # noqa: E712

    source_counts = file_summary["source_layer"].value_counts().to_dict() if not file_summary.empty else {}
    total_duration_h = pd.to_numeric(file_summary.get("duration_s", pd.Series(dtype=float)), errors="coerce").sum() / 3600.0

    lines: List[str] = []
    lines.append("# 连续车辆源数据审计报告（2026-06-24）")
    lines.append("")
    lines.append(f"- 生成时间：{now_text()}")
    lines.append("- 审计边界：只读扫描连续车辆 CSV；不使用样本集、训练标签、模型预测；不修改原始数据。")
    lines.append("- 审计方法：按时序数据质量常见框架检查资产盘点、完整性、一致性、唯一性、时间轴、分布尾部和下游风险。")
    lines.append(f"- 输出目录：`{roots['audit_root']}`")
    lines.append("")
    lines.append("## 1. 资产盘点")
    lines.append("")
    lines.append(f"- 候选文件总数：{len(inventory)}")
    lines.append(f"- 纳入连续车辆审计文件数：{len(included)}")
    lines.append(f"- 纳入文件覆盖被试：{file_summary['subject'].nunique() if not file_summary.empty else 0}")
    lines.append(f"- 纳入文件覆盖记录键：{file_summary['recording_key'].nunique() if not file_summary.empty else 0}")
    lines.append(f"- 纳入文件总时长：约 {total_duration_h:.2f} 小时")
    lines.append(f"- 主车辆层文件数：{source_counts.get('main_vehicle_aligned_cleaned', 0)}")
    lines.append(f"- 补充车辆 200Hz 层文件数：{source_counts.get('supplement_vehicle_fixed_200hz', 0)}")
    lines.append("")
    lines.append("候选文件类别计数见 `tables/vehicle_file_inventory.csv`；核心计数如下：")
    lines.append("")
    lines.append(class_counts.to_markdown(index=False))
    lines.append("")
    lines.append("纳入审计文件按源层分开看如下：")
    lines.append("")
    if source_layer_summary.empty:
        lines.append("无源层汇总。")
    else:
        keep_cols = [
            "source_layer",
            "file_count",
            "recording_count",
            "total_duration_h",
            "median_dt_ms",
            "median_nominal_hz",
            "files_nominal_hz_outside_150_250",
            "files_high_key_missing_rate",
            "files_low_road_ref_ok_rate",
        ]
        lines.append(source_layer_summary[keep_cols].to_markdown(index=False))
    lines.append("")
    lines.append("## 2. 主要发现")
    lines.append("")
    if findings.empty:
        lines.append("本轮未生成自动发现项；请直接查看明细表。")
    else:
        for i, row in enumerate(findings.itertuples(index=False), start=1):
            lines.append(f"### {i}. [{row.severity}] {row.finding}")
            lines.append("")
            lines.append(f"- 证据：{row.evidence}")
            lines.append(f"- 影响：{row.impact}")
            lines.append(f"- 下一步检查：{row.next_check}")
            lines.append("")

    lines.append("## 3. 时序质量摘要")
    lines.append("")
    if file_summary.empty:
        lines.append("没有纳入文件。")
    else:
        metrics = {
            "median_dt_ms_median": pd.to_numeric(file_summary["median_dt_ms"], errors="coerce").median(),
            "nominal_hz_median": pd.to_numeric(file_summary["nominal_hz"], errors="coerce").median(),
            "gap_gt_20ms_files": int((pd.to_numeric(file_summary["gap_gt_20ms_count"], errors="coerce") > 0).sum()),
            "gap_gt_50ms_files": int((pd.to_numeric(file_summary["gap_gt_50ms_count"], errors="coerce") > 0).sum()),
            "nonmonotonic_files": int((pd.to_numeric(file_summary["nonmonotonic_dt_count"], errors="coerce") > 0).sum()),
            "duplicate_storage_files": int((pd.to_numeric(file_summary["duplicate_storage_time_count"], errors="coerce") > 0).sum()),
        }
        lines.append(pd.DataFrame([metrics]).to_markdown(index=False))
        lines.append("")
        worst_time = file_summary.sort_values(
            ["gap_gt_50ms_count", "gap_gt_20ms_count", "max_dt_ms"], ascending=False
        ).head(12)
        cols = [
            "file_id",
            "subject",
            "recording_key",
            "source_layer",
            "rows",
            "duration_s",
            "median_dt_ms",
            "max_dt_ms",
            "gap_gt_50ms_count",
            "suspect_flags",
        ]
        lines.append("时间轴风险最高的记录：")
        lines.append("")
        lines.append(worst_time[cols].to_markdown(index=False))
        lines.append("")

    lines.append("## 4. 被试分布摘要")
    lines.append("")
    if subject_summary.empty:
        lines.append("没有被试汇总。")
    else:
        cols = [
            "subject",
            "included_file_count",
            "recording_count",
            "total_duration_min",
            "gap_gt_50ms_files",
            "suspect_file_count",
            "drive_seconds_speed_gt_5",
            "lateral_action_seconds",
        ]
        lines.append(subject_summary[cols].to_markdown(index=False))
        lines.append("")

    lines.append("## 5. 明细文件")
    lines.append("")
    table_files = [
        "vehicle_file_inventory.csv",
        "file_vehicle_quality_summary.csv",
        "vehicle_numeric_column_summary.csv",
        "recording_cluster_summary.csv",
        "subject_vehicle_summary.csv",
        "source_layer_summary.csv",
        "road_type_summary.csv",
        "vehicle_source_audit_findings.csv",
    ]
    for name in table_files:
        lines.append(f"- `{tables_dir / name}`")
    if created_figures:
        lines.append("")
        lines.append("## 6. 图")
        lines.append("")
        for fig in created_figures:
            lines.append(f"- `{fig}`")
    lines.append("")
    lines.append("## 7. 结论边界")
    lines.append("")
    lines.append("- 这轮审计说明哪些连续车辆记录更可信、哪些目录/文件容易误读，但它不自动判定任何样本标签正确。")
    lines.append("- 如果要解释现有模型失败样本，下一步应把 `file_id/recording_key` 映射回锚点窗口，检查失败是否来自源记录质量、道路参考、时间轴间隔、还是任务可观测性。")
    lines.append("- 补充采集目录必须先字段分类再读取，不能只凭 `vehicle_fixed_200Hz` 文件名。")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    entry_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit continuous vehicle source CSV files.")
    parser.add_argument("--limit", type=int, default=0, help="Debug only: process at most N included vehicle files.")
    args = parser.parse_args()

    roots = find_project_roots(Path(__file__))
    ensure_dirs([roots["audit_root"], roots["tables"], roots["logs"], roots["reports"], roots["figures"]])

    inventory_rows = discover_files(roots)
    inventory = pd.DataFrame(inventory_rows)
    inventory_path = roots["tables"] / "vehicle_file_inventory.csv"
    inventory.to_csv(inventory_path, index=False, encoding="utf-8-sig")

    included = inventory[inventory["include_in_continuous_vehicle_audit"] == True].copy()  # noqa: E712
    if args.limit and args.limit > 0:
        included = included.head(args.limit)

    file_rows: List[Dict[str, object]] = []
    numeric_rows: List[Dict[str, object]] = []
    road_rows: List[Dict[str, object]] = []
    errors: List[Dict[str, object]] = []

    for i, row in enumerate(included.to_dict("records"), start=1):
        print(f"[{i}/{len(included)}] {row['file_class']} {row['subject']} {row['file_name']}")
        try:
            summary, nums, roads = audit_one_file(row, roots)
            file_rows.append(summary)
            numeric_rows.extend(nums)
            road_rows.extend(roads)
        except Exception as exc:  # pragma: no cover - 审计继续执行，错误写日志
            errors.append(
                {
                    "file_id": row["candidate_file_id"],
                    "rel_path": row["rel_path"],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    file_summary = pd.DataFrame(file_rows)
    numeric_summary = pd.DataFrame(numeric_rows)
    road_summary = pd.DataFrame(road_rows)
    cluster_summary = build_cluster_summary(file_summary) if not file_summary.empty else pd.DataFrame()
    subject_summary = build_subject_summary(file_summary) if not file_summary.empty else pd.DataFrame()
    source_layer_summary = build_source_layer_summary(file_summary) if not file_summary.empty else pd.DataFrame()
    findings = build_findings(inventory, file_summary, cluster_summary, subject_summary, source_layer_summary)

    file_summary.to_csv(roots["tables"] / "file_vehicle_quality_summary.csv", index=False, encoding="utf-8-sig")
    numeric_summary.to_csv(roots["tables"] / "vehicle_numeric_column_summary.csv", index=False, encoding="utf-8-sig")
    road_summary.to_csv(roots["tables"] / "road_type_summary.csv", index=False, encoding="utf-8-sig")
    cluster_summary.to_csv(roots["tables"] / "recording_cluster_summary.csv", index=False, encoding="utf-8-sig")
    subject_summary.to_csv(roots["tables"] / "subject_vehicle_summary.csv", index=False, encoding="utf-8-sig")
    source_layer_summary.to_csv(roots["tables"] / "source_layer_summary.csv", index=False, encoding="utf-8-sig")
    findings.to_csv(roots["tables"] / "vehicle_source_audit_findings.csv", index=False, encoding="utf-8-sig")

    if errors:
        pd.DataFrame(errors).to_csv(roots["logs"] / "vehicle_source_audit_errors.csv", index=False, encoding="utf-8-sig")

    created_figures = write_figures(file_summary, subject_summary, roots)
    manifest = {
        "created_at": now_text(),
        "script": str(Path(__file__).resolve()),
        "audit_root": str(roots["audit_root"]),
        "candidate_files": int(len(inventory)),
        "included_files": int(len(included)),
        "processed_files": int(len(file_summary)),
        "errors": errors,
        "tables": sorted(p.name for p in roots["tables"].glob("*.csv")),
        "figures": created_figures,
    }
    (roots["logs"] / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    write_report(roots, inventory, file_summary, cluster_summary, subject_summary, source_layer_summary, findings, created_figures)
    print(f"Done. Report: {roots['report_entry']}")


if __name__ == "__main__":
    main()
