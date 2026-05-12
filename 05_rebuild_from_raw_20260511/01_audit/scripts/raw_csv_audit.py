# -*- coding: utf-8 -*-
"""
Stage 0/1 audit for the R2E-Steering rebuild.

The script is intentionally read-only with respect to source data. It creates
inventory tables, raw CSV schema/time/signal quality reports, lightweight PNG
figures, and the project transparency documents required by the rebuild plan.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


RAW_SCOPE_TOP_DIRS = {
    "原始车辆数据",
    "原始生理数据",
    "原始脑电数据",
}

ENCODINGS = ("utf-8-sig", "utf-8", "gbk", "latin1")

VEHICLE_SIGNAL_HINTS = [
    "zx|SteeringWheel",
    "zx|vx",
    "zx|vy",
    "zx|yaw",
    "zx|roll",
    "zx1|v_km/h",
    "zx1|lanecurvatureXY",
    "zx1|lateraldistance",
]
PHYSIO_SIGNAL_HINTS = ["CH1-ECG", "CH2-EMG", "CH3-EDA", "CH4-RESP"]
EEG_PREFIX_HINTS = ["LSLOutletStreamName-EEG|channel", "LSLOutletStreamName-Accelerometer|channel"]
DERIVED_KEYWORDS = (
    "feature",
    "features",
    "10Hz",
    "200Hz",
    "aligned",
    "cleaned",
    "processed",
    "处理",
    "对齐",
)


@dataclass
class NumericAccumulator:
    name: str
    n_valid: int = 0
    n_total: int = 0
    n_zero: int = 0
    v_min: float = math.inf
    v_max: float = -math.inf
    sum_v: float = 0.0
    sum_sq: float = 0.0
    last_valid: Optional[float] = None
    abs_diffs: List[np.ndarray] = field(default_factory=list)

    def update(self, values: pd.Series) -> None:
        arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64", copy=False)
        self.n_total += int(arr.size)
        valid = arr[np.isfinite(arr)]
        if valid.size == 0:
            return
        self.n_valid += int(valid.size)
        self.n_zero += int(np.isclose(valid, 0.0, atol=1.0e-12).sum())
        self.v_min = min(self.v_min, float(np.min(valid)))
        self.v_max = max(self.v_max, float(np.max(valid)))
        self.sum_v += float(np.sum(valid))
        self.sum_sq += float(np.sum(valid * valid))
        if self.last_valid is not None:
            valid_for_diff = np.concatenate(([self.last_valid], valid))
        else:
            valid_for_diff = valid
        if valid_for_diff.size > 1:
            self.abs_diffs.append(np.abs(np.diff(valid_for_diff)))
        self.last_valid = float(valid[-1])

    def finish(self) -> Dict[str, object]:
        if self.n_valid == 0:
            return {
                "signal": self.name,
                "valid_count": 0,
                "total_count": self.n_total,
                "valid_rate": 0.0,
                "min": "",
                "max": "",
                "mean": "",
                "std": "",
                "zero_rate": "",
                "near_constant": True,
                "median_abs_diff": "",
                "spike_rate_proxy": "",
            }
        mean = self.sum_v / self.n_valid
        var = max(0.0, self.sum_sq / self.n_valid - mean * mean)
        std = math.sqrt(var)
        if self.abs_diffs:
            diffs = np.concatenate(self.abs_diffs)
            finite = diffs[np.isfinite(diffs)]
        else:
            finite = np.array([], dtype="float64")
        if finite.size:
            median_abs_diff = float(np.median(finite))
            threshold = max(median_abs_diff * 20.0, std * 8.0, 1.0e-12)
            spike_rate = float((finite > threshold).sum() / finite.size)
        else:
            median_abs_diff = ""
            spike_rate = ""
        return {
            "signal": self.name,
            "valid_count": self.n_valid,
            "total_count": self.n_total,
            "valid_rate": self.n_valid / max(1, self.n_total),
            "min": self.v_min,
            "max": self.v_max,
            "mean": mean,
            "std": std,
            "zero_rate": self.n_zero / max(1, self.n_valid),
            "near_constant": bool(std < 1.0e-9),
            "median_abs_diff": median_abs_diff,
            "spike_rate_proxy": spike_rate,
        }


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_dirs(paths: Sequence[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def repo_paths(script_path: Path) -> Dict[str, Path]:
    rebuild_root = script_path.resolve().parents[2]
    project_root = rebuild_root.parent
    return {
        "project_root": project_root,
        "rebuild_root": rebuild_root,
        "raw_root": project_root / "01_datasets" / "数据预处理",
        "notes": rebuild_root / "00_project_notes",
        "daily_logs": rebuild_root / "00_project_notes" / "daily_logs",
        "audit": rebuild_root / "01_audit",
        "tables": rebuild_root / "01_audit" / "tables",
        "figures": rebuild_root / "01_audit" / "figures" / "audit",
        "logs": rebuild_root / "01_audit" / "logs",
        "reports": rebuild_root / "09_reports",
    }


def rel_to(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def top_dir_for(path: Path, raw_root: Path) -> str:
    rel = path.relative_to(raw_root)
    return rel.parts[0] if rel.parts else ""


def subject_for(path: Path, raw_root: Path) -> str:
    rel = path.relative_to(raw_root)
    if len(rel.parts) >= 3 and rel.parts[0] in RAW_SCOPE_TOP_DIRS:
        return rel.parts[1]
    return "unknown"


def session_stamp_for(path: Path) -> str:
    match = re.search(r"Entity_Recording_(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})", path.name)
    return match.group(1) if match else ""


def infer_modality(path: Path) -> str:
    text = path.as_posix().lower()
    if "vehicle" in text or "车辆" in text:
        return "vehicle"
    if "physio" in text or "生理" in text:
        return "physio"
    if "eeg" in text or "脑电" in text:
        return "eeg"
    return "unknown"


def is_subject_direct_csv(path: Path, raw_root: Path) -> bool:
    try:
        rel = path.relative_to(raw_root)
    except ValueError:
        return False
    if len(rel.parts) != 3:
        return False
    if rel.parts[0] not in RAW_SCOPE_TOP_DIRS:
        return False
    if not path.name.lower().endswith(".csv"):
        return False
    subject_dir = rel.parts[1]
    if any(keyword.lower() in subject_dir.lower() for keyword in DERIVED_KEYWORDS):
        return False
    return True


def is_raw_scope(path: Path, raw_root: Path) -> bool:
    return is_subject_direct_csv(path, raw_root)


def is_derived_file(path: Path, raw_root: Path) -> bool:
    rel = rel_to(path, raw_root)
    return any(keyword.lower() in rel.lower() for keyword in DERIVED_KEYWORDS)


def is_raw_sensor_scope(path: Path, raw_root: Path) -> bool:
    return is_raw_scope(path, raw_root) and not is_derived_file(path, raw_root)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def read_header(path: Path) -> Tuple[List[str], str]:
    last_error: Optional[Exception] = None
    for encoding in ENCODINGS:
        try:
            with path.open("r", encoding=encoding, newline="") as f:
                reader = csv.reader(f)
                header = next(reader)
            return [c.strip().replace("\ufeff", "") for c in header], encoding
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise RuntimeError(f"cannot read header: {last_error}")


def choose_time_col(columns: Sequence[str]) -> str:
    preferred = ["StorageTime", "t_s", "time", "Time", "timestamp", "Timestamp"]
    for col in preferred:
        if col in columns:
            return col
    for col in columns:
        if "time" in col.lower() or "时间" in col:
            return col
    return ""


def choose_signal_columns(columns: Sequence[str], modality: str) -> List[str]:
    if modality == "vehicle":
        return [c for c in VEHICLE_SIGNAL_HINTS if c in columns]
    if modality == "physio":
        picked = []
        for col in columns:
            if any(hint in col for hint in PHYSIO_SIGNAL_HINTS):
                picked.append(col)
        for col in ["HR_bpm", "HRV_RMSSD_30s", "RESP_BPM", "RESP_Amplitude", "EMG_RMS", "EDA_Tonic", "EDA_Phasic"]:
            if col in columns and col not in picked:
                picked.append(col)
        return picked
    if modality == "eeg":
        return [
            c
            for c in columns
            if any(c.startswith(prefix) for prefix in EEG_PREFIX_HINTS)
        ]
    return []


def parse_time_series(values: pd.Series) -> Tuple[np.ndarray, str]:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64", copy=False)
    numeric_valid = np.isfinite(numeric).sum()
    if numeric_valid >= max(2, int(0.5 * len(values))):
        return numeric, "numeric"

    dt = pd.to_datetime(values, errors="coerce", format="mixed")
    valid = dt.notna().to_numpy()
    out = np.full(len(values), np.nan, dtype="float64")
    if valid.any():
        # Force nanosecond epoch units. Some pandas builds preserve microsecond
        # resolution for parsed datetimes, and using the raw integer value would
        # inflate inferred sampling rates by 1000x.
        dt_ns = dt.to_numpy(dtype="datetime64[ns]")
        out[valid] = dt_ns[valid].astype("int64").astype("float64") / 1.0e9
    return out, "datetime"


def iter_chunks(path: Path, encoding: str, chunksize: int) -> Iterable[pd.DataFrame]:
    try:
        yield from pd.read_csv(
            path,
            chunksize=chunksize,
            encoding=encoding,
            low_memory=False,
            on_bad_lines="skip",
        )
    except UnicodeDecodeError:
        for fallback in ENCODINGS:
            if fallback == encoding:
                continue
            yield from pd.read_csv(
                path,
                chunksize=chunksize,
                encoding=fallback,
                low_memory=False,
                on_bad_lines="skip",
            )
            return


def infer_hz_from_dt(median_dt: float) -> Tuple[object, str]:
    if not np.isfinite(median_dt) or median_dt <= 0:
        return "", "unknown"
    if median_dt > 1.0:
        return 1000.0 / median_dt, "milliseconds_like"
    return 1.0 / median_dt, "seconds_like"


def audit_one_csv(
    path: Path,
    raw_root: Path,
    chunksize: int,
) -> Tuple[Dict[str, object], Dict[str, object], List[Dict[str, object]], List[Dict[str, object]]]:
    rel = rel_to(path, raw_root)
    top = top_dir_for(path, raw_root)
    modality = infer_modality(path)
    subject = subject_for(path, raw_root)
    session = session_stamp_for(path)
    header, encoding = read_header(path)
    time_col = choose_time_col(header)
    signal_cols = choose_signal_columns(header, modality)
    accumulators = {name: NumericAccumulator(name) for name in signal_cols}

    row_count = 0
    missing_cells = 0
    total_cells = 0
    time_valid = 0
    time_min = math.inf
    time_max = -math.inf
    last_time: Optional[float] = None
    diff_parts: List[np.ndarray] = []
    time_parse_mode = ""

    for chunk in iter_chunks(path, encoding, chunksize):
        chunk.columns = [str(c).strip().replace("\ufeff", "") for c in chunk.columns]
        row_count += int(len(chunk))
        total_cells += int(len(chunk) * len(chunk.columns))
        missing_cells += int(chunk.isna().sum().sum())

        if time_col and time_col in chunk.columns:
            t, mode = parse_time_series(chunk[time_col])
            if not time_parse_mode:
                time_parse_mode = mode
            valid = t[np.isfinite(t)]
            if valid.size:
                time_valid += int(valid.size)
                time_min = min(time_min, float(np.min(valid)))
                time_max = max(time_max, float(np.max(valid)))
                if last_time is not None:
                    valid_for_diff = np.concatenate(([last_time], valid))
                else:
                    valid_for_diff = valid
                if valid_for_diff.size > 1:
                    diff_parts.append(np.diff(valid_for_diff))
                last_time = float(valid[-1])

        for col, acc in accumulators.items():
            if col in chunk.columns:
                acc.update(chunk[col])

    if diff_parts:
        diffs = np.concatenate(diff_parts)
        finite_diffs = diffs[np.isfinite(diffs)]
        positive = finite_diffs[finite_diffs > 0]
    else:
        finite_diffs = np.array([], dtype="float64")
        positive = np.array([], dtype="float64")
    median_dt = float(np.median(positive)) if positive.size else math.nan
    inferred_hz, time_unit = infer_hz_from_dt(median_dt)
    large_gap_threshold = median_dt * 5.0 if np.isfinite(median_dt) and median_dt > 0 else math.nan
    large_gap_count = int((positive > large_gap_threshold).sum()) if np.isfinite(large_gap_threshold) else 0

    schema_row = {
        "relative_path": rel,
        "top_dir": top,
        "subject": subject,
        "session_stamp": session,
        "modality": modality,
        "parse_status": "ok",
        "encoding": encoding,
        "row_count": row_count,
        "column_count": len(header),
        "time_col": time_col,
        "id_col_present": "ID" in header,
        "raw_sensor_scope": is_raw_sensor_scope(path, raw_root),
        "derived_file": is_derived_file(path, raw_root),
        "missing_rate_all_cells": missing_cells / max(1, total_cells),
        "columns_json": json.dumps(header, ensure_ascii=False),
        "columns_preview": " | ".join(header[:30]),
    }
    timestamp_row = {
        "relative_path": rel,
        "top_dir": top,
        "subject": subject,
        "session_stamp": session,
        "modality": modality,
        "raw_sensor_scope": is_raw_sensor_scope(path, raw_root),
        "derived_file": is_derived_file(path, raw_root),
        "time_col": time_col,
        "time_parse_mode": time_parse_mode,
        "row_count": row_count,
        "time_valid_count": time_valid,
        "time_valid_rate": time_valid / max(1, row_count),
        "time_min": "" if time_min == math.inf else time_min,
        "time_max": "" if time_max == -math.inf else time_max,
        "duration_raw_units": "" if time_min == math.inf or time_max == -math.inf else time_max - time_min,
        "median_positive_dt_raw": "" if not np.isfinite(median_dt) else median_dt,
        "time_unit_inference": time_unit,
        "inferred_sampling_hz": inferred_hz,
        "zero_dt_count": int((finite_diffs == 0).sum()) if finite_diffs.size else 0,
        "negative_dt_count": int((finite_diffs < 0).sum()) if finite_diffs.size else 0,
        "large_gap_threshold_raw": "" if not np.isfinite(large_gap_threshold) else large_gap_threshold,
        "large_gap_count": large_gap_count,
        "max_positive_gap_raw": float(np.max(positive)) if positive.size else "",
    }

    signal_rows = []
    eeg_rows = []
    for acc in accumulators.values():
        row = acc.finish()
        row.update(
            {
                "relative_path": rel,
                "top_dir": top,
                "subject": subject,
                "session_stamp": session,
                "modality": modality,
                "raw_sensor_scope": is_raw_sensor_scope(path, raw_root),
                "derived_file": is_derived_file(path, raw_root),
            }
        )
        signal_rows.append(row)
        if modality == "eeg":
            eeg_rows.append(row.copy())

    return schema_row, timestamp_row, signal_rows, eeg_rows


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: List[str] = []
        for row in rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def build_inventory(raw_root: Path, hash_mode: str, log) -> List[Dict[str, object]]:
    rows = []
    csv_files: List[Path] = []
    for top in sorted(RAW_SCOPE_TOP_DIRS):
        top_path = raw_root / top
        if not top_path.exists():
            log(f"[inventory-warning] missing raw top dir: {top_path}")
            continue
        for subject_dir in sorted([p for p in top_path.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
            if any(keyword.lower() in subject_dir.name.lower() for keyword in DERIVED_KEYWORDS):
                log(f"[inventory-skip] derived/non-subject dir skipped: {rel_to(subject_dir, raw_root)}")
                continue
            csv_files.extend(sorted(subject_dir.glob("*.csv")))
    for idx, path in enumerate(csv_files, start=1):
        rel = rel_to(path, raw_root)
        top = top_dir_for(path, raw_root)
        row: Dict[str, object] = {
            "relative_path": rel,
            "absolute_path": str(path),
            "top_dir": top,
            "subject": subject_for(path, raw_root),
            "session_stamp": session_stamp_for(path),
            "modality": infer_modality(path),
        "raw_scope": is_raw_scope(path, raw_root),
        "raw_sensor_scope": is_raw_sensor_scope(path, raw_root),
        "derived_file": is_derived_file(path, raw_root),
        "size_bytes": path.stat().st_size,
            "mtime": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
            "sha256": "",
            "hash_status": "not_requested",
        }
        should_hash = hash_mode == "all" or (hash_mode == "raw" and row["raw_scope"])
        if should_hash:
            try:
                row["sha256"] = sha256_file(path)
                row["hash_status"] = "ok"
            except Exception as exc:  # noqa: BLE001
                row["hash_status"] = f"error: {type(exc).__name__}: {exc}"
        rows.append(row)
        if idx % 50 == 0 or idx == len(csv_files):
            log(f"[inventory] {idx}/{len(csv_files)} CSV files scanned")
    return rows


def build_subject_session_matrix(
    inventory_rows: Sequence[Dict[str, object]],
    timestamp_rows: Sequence[Dict[str, object]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    time_by_path = {row["relative_path"]: row for row in timestamp_rows}
    grouped: Dict[Tuple[str, str], Dict[str, object]] = {}

    for inv in inventory_rows:
        if not inv.get("raw_sensor_scope"):
            continue
        key = (str(inv.get("subject") or "unknown"), str(inv.get("session_stamp") or "unknown"))
        if key not in grouped:
            grouped[key] = {
                "subject": key[0],
                "session_stamp": key[1],
                "vehicle_file_count": 0,
                "physio_file_count": 0,
                "eeg_file_count": 0,
                "vehicle_paths": [],
                "physio_paths": [],
                "eeg_paths": [],
                "vehicle_time_min": "",
                "vehicle_time_max": "",
                "physio_time_min": "",
                "physio_time_max": "",
                "eeg_time_min": "",
                "eeg_time_max": "",
            }
        entry = grouped[key]
        modality = str(inv.get("modality") or "unknown")
        rel = str(inv["relative_path"])
        if modality in {"vehicle", "physio", "eeg"}:
            entry[f"{modality}_file_count"] = int(entry[f"{modality}_file_count"]) + 1
            entry[f"{modality}_paths"].append(rel)
            trow = time_by_path.get(rel)
            if trow:
                tmin = trow.get("time_min", "")
                tmax = trow.get("time_max", "")
                if tmin != "":
                    old = entry[f"{modality}_time_min"]
                    entry[f"{modality}_time_min"] = tmin if old == "" else min(float(old), float(tmin))
                if tmax != "":
                    old = entry[f"{modality}_time_max"]
                    entry[f"{modality}_time_max"] = tmax if old == "" else max(float(old), float(tmax))

    matrix_rows: List[Dict[str, object]] = []
    overlap_rows: List[Dict[str, object]] = []
    for key in sorted(grouped):
        entry = grouped[key]
        for modality in ["vehicle", "physio", "eeg"]:
            entry[f"{modality}_available"] = int(entry[f"{modality}_file_count"]) > 0
            entry[f"{modality}_paths"] = " | ".join(entry[f"{modality}_paths"])
        entry["all_three_modalities_available"] = (
            entry["vehicle_available"] and entry["physio_available"] and entry["eeg_available"]
        )
        matrix_rows.append(entry.copy())

        available_ranges = []
        for modality in ["vehicle", "physio", "eeg"]:
            tmin = entry[f"{modality}_time_min"]
            tmax = entry[f"{modality}_time_max"]
            if tmin != "" and tmax != "":
                available_ranges.append((modality, float(tmin), float(tmax)))
        if len(available_ranges) >= 2:
            overlap_start = max(r[1] for r in available_ranges)
            overlap_end = min(r[2] for r in available_ranges)
            union_start = min(r[1] for r in available_ranges)
            union_end = max(r[2] for r in available_ranges)
            overlap = max(0.0, overlap_end - overlap_start)
            union = max(1.0e-12, union_end - union_start)
            status = "overlap_ok" if overlap > 0 else "no_overlap_or_time_unit_mismatch"
        else:
            overlap_start = overlap_end = overlap = ""
            union = ""
            status = "insufficient_modalities"
        overlap_rows.append(
            {
                "subject": entry["subject"],
                "session_stamp": entry["session_stamp"],
                "modalities_with_time": ",".join(r[0] for r in available_ranges),
                "overlap_start_raw": overlap_start,
                "overlap_end_raw": overlap_end,
                "overlap_duration_raw": overlap,
                "union_duration_raw": union,
                "overlap_ratio_raw": "" if union == "" else float(overlap) / float(union),
                "status": status,
            }
        )
    return matrix_rows, overlap_rows


def build_sampling_report(timestamp_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped = defaultdict(list)
    for row in timestamp_rows:
        hz = row.get("inferred_sampling_hz", "")
        if hz == "":
            continue
        try:
            grouped[(row.get("top_dir", ""), row.get("modality", ""))].append(float(hz))
        except Exception:
            pass
    out = []
    for (top, modality), values in sorted(grouped.items()):
        arr = np.asarray(values, dtype="float64")
        out.append(
            {
                "top_dir": top,
                "modality": modality,
                "file_count": int(arr.size),
                "hz_median": float(np.median(arr)),
                "hz_min": float(np.min(arr)),
                "hz_max": float(np.max(arr)),
                "hz_mean": float(np.mean(arr)),
                "hz_std": float(np.std(arr)),
            }
        )
    return out


def build_leakage_risk_report(project_root: Path) -> List[Dict[str, object]]:
    old_builder = project_root / "02_code" / "final_code" / "dataset" / "build_event_dataset_v2_pad_mask_multipeak.py"
    old_training = project_root / "02_code" / "final_code" / "model" / "training" / "run_event_conditioned_trajectory_baseline.py"
    return [
        {
            "risk_id": "L01",
            "risk_area": "旧事件锚点",
            "severity": "high",
            "current_evidence": f"旧样本构建脚本存在 anchor peak 选择逻辑，参考文件：{old_builder.as_posix()}",
            "why_it_matters": "如果锚点是方向盘/横摆/侧倾响应峰值，而不是事件原因发生时刻，预测任务可能变成从响应中段预测响应后段。",
            "required_action": "阶段 2 重新定义事件锚点来源；保留旧锚点为历史对照和失败样本定位，不作为默认真相。",
        },
        {
            "risk_id": "L02",
            "risk_area": "输入/标签窗口",
            "severity": "high",
            "current_evidence": "旧脚本中可见 WINDOW_PRE=2.0、WINDOW_POST=2.0；训练脚本中也存在 FUTURE_SEC/输入历史窗口假设。",
            "why_it_matters": "若输入窗口跨过真实事件发生点或包含主响应动作，车辆/生理特征会混入未来标签信息。",
            "required_action": "samples_master 中逐样本记录 input_start/input_end/label_start/label_end，并明确 causal setting。",
        },
        {
            "risk_id": "L03",
            "risk_area": "EMG 动作结果泄漏",
            "severity": "high",
            "current_evidence": "肌电与方向盘动作高度相关，旧流程曾显示 EMG 候选有效，但原始窗口未重新审计。",
            "why_it_matters": "事件后 EMG 可能已经包含手臂动作结果，不能直接证明驾驶员内部状态先验有效。",
            "required_action": "将 EMG 分为事件前输入、早期观察后预测剩余轨迹、上限分析三类，报告中显式标注。",
        },
        {
            "risk_id": "L04",
            "risk_area": "标准化/基线校正",
            "severity": "high",
            "current_evidence": f"旧训练入口和特征脚本需要复核 train-only fit，参考入口：{old_training.as_posix()}",
            "why_it_matters": "如果标准化、PCA、风格统计或生理基线使用了测试集，就会虚增风格/生理收益。",
            "required_action": "阶段 2 数据版本卡中记录每个统计量的 fit 范围；阶段 3 以后所有 scaler 必须只在训练集 fit。",
        },
        {
            "risk_id": "L05",
            "risk_area": "随机切分与被试泄漏",
            "severity": "medium",
            "current_evidence": "旧流程中存在随机样本切分历史争论；新流程目标要求 session/subject-level 可行性说明。",
            "why_it_matters": "随机切分可能让同一被试或同一记录的相近事件同时进入训练和测试。",
            "required_action": "阶段 2 同时输出 random/session/subject split_table；阶段 4/5 不能只用随机切分声明风格或生理有效。",
        },
        {
            "risk_id": "L06",
            "risk_area": "200Hz 对齐与重采样",
            "severity": "medium",
            "current_evidence": "原始目录中同时存在原始 CSV、处理后 200Hz、生理 reclean、对齐后车辆生理目录。",
            "why_it_matters": "处理后数据可能已经引入插值、裁剪或跨模态对齐假设；新流程需要先从原始时间戳确认。",
            "required_action": "阶段 1 使用原始 CSV 审计时间轴；处理后目录只用于核对，不直接作为可信输入。",
        },
        {
            "risk_id": "L07",
            "risk_area": "评价指标不足",
            "severity": "medium",
            "current_evidence": "旧 RMSE 排名与预测图物理问题存在冲突。",
            "why_it_matters": "小 RMSE 波动不能证明方向、幅值、尾段回正、反向修正和困难样本被真正改善。",
            "required_action": "阶段 3 开始固定物理指标和固定预测图协议；RMSE 只能作为指标之一。",
        },
    ]


def load_font(size: int = 16) -> ImageFont.ImageFont:
    candidates = [
        Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "arial.ttf",
        Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "msyh.ttc",
    ]
    for candidate in candidates:
        if candidate.exists():
            try:
                return ImageFont.truetype(str(candidate), size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def draw_bar_chart(path: Path, title: str, labels: List[str], values: List[float]) -> None:
    width, height = 1200, 720
    margin_left, margin_right, margin_top, margin_bottom = 260, 60, 80, 90
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = load_font(15)
    title_font = load_font(22)
    draw.text((margin_left, 25), title, fill=(20, 20, 20), font=title_font)
    if not values:
        draw.text((margin_left, margin_top), "No data", fill=(180, 0, 0), font=font)
        img.save(path)
        return
    max_v = max(values) if max(values) > 0 else 1
    bar_h = max(18, min(40, (height - margin_top - margin_bottom) // max(1, len(values))))
    gap = max(6, bar_h // 3)
    scale_w = width - margin_left - margin_right
    for i, (label, value) in enumerate(zip(labels, values)):
        y = margin_top + i * (bar_h + gap)
        if y + bar_h > height - margin_bottom:
            break
        draw.text((20, y), label[:32], fill=(40, 40, 40), font=font)
        bar_w = int(scale_w * value / max_v)
        color = (53, 117, 159) if i % 2 == 0 else (80, 150, 111)
        draw.rectangle((margin_left, y, margin_left + bar_w, y + bar_h), fill=color)
        draw.text((margin_left + bar_w + 8, y), f"{value:g}", fill=(20, 20, 20), font=font)
    img.save(path)


def draw_timeline_chart(path: Path, rows: Sequence[Dict[str, object]], title: str, max_rows: int = 32) -> None:
    selected = [r for r in rows if r.get("status") != "insufficient_modalities"][:max_rows]
    width, height = 1400, max(500, 90 + len(selected) * 28)
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = load_font(14)
    title_font = load_font(22)
    draw.text((40, 24), title, fill=(20, 20, 20), font=title_font)
    if not selected:
        draw.text((40, 90), "No overlap rows with two or more modalities.", fill=(180, 0, 0), font=font)
        img.save(path)
        return
    starts, ends = [], []
    for r in selected:
        try:
            starts.append(float(r["overlap_start_raw"]))
            ends.append(float(r["overlap_end_raw"]))
        except Exception:
            pass
    if not starts or not ends:
        img.save(path)
        return
    x0, x1 = min(starts), max(ends)
    if x1 <= x0:
        x1 = x0 + 1.0
    left, right = 330, width - 80
    for i, r in enumerate(selected):
        y = 85 + i * 28
        label = f"{r.get('subject','')} {r.get('session_stamp','')}"
        draw.text((30, y - 8), label[:36], fill=(40, 40, 40), font=font)
        try:
            a = float(r["overlap_start_raw"])
            b = float(r["overlap_end_raw"])
            xa = left + int((a - x0) / (x1 - x0) * (right - left))
            xb = left + int((b - x0) / (x1 - x0) * (right - left))
            color = (66, 135, 75) if r.get("status") == "overlap_ok" else (190, 90, 40)
            draw.line((left, y, right, y), fill=(220, 220, 220), width=2)
            draw.rectangle((xa, y - 6, max(xa + 2, xb), y + 6), fill=color)
        except Exception:
            draw.text((left, y - 8), "time range unavailable", fill=(160, 0, 0), font=font)
    img.save(path)


def draw_histogram(path: Path, title: str, values: Sequence[float]) -> None:
    clean = np.asarray([v for v in values if np.isfinite(v)], dtype="float64")
    width, height = 1000, 650
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = load_font(14)
    title_font = load_font(22)
    draw.text((45, 25), title, fill=(20, 20, 20), font=title_font)
    if clean.size == 0:
        draw.text((45, 90), "No numeric data", fill=(180, 0, 0), font=font)
        img.save(path)
        return
    counts, bins = np.histogram(clean, bins=min(30, max(5, int(np.sqrt(clean.size)))))
    left, top, right, bottom = 90, 90, width - 50, height - 90
    max_count = max(1, int(counts.max()))
    bar_w = (right - left) / len(counts)
    for i, count in enumerate(counts):
        x_a = left + i * bar_w
        x_b = left + (i + 1) * bar_w - 2
        y_a = bottom - (count / max_count) * (bottom - top)
        draw.rectangle((x_a, y_a, x_b, bottom), fill=(64, 120, 164))
    draw.line((left, bottom, right, bottom), fill=(30, 30, 30), width=2)
    draw.line((left, top, left, bottom), fill=(30, 30, 30), width=2)
    draw.text((left, bottom + 12), f"min={clean.min():.4g}", fill=(40, 40, 40), font=font)
    draw.text((right - 180, bottom + 12), f"max={clean.max():.4g}", fill=(40, 40, 40), font=font)
    draw.text((left, bottom + 38), f"median={np.median(clean):.4g}, n={clean.size}", fill=(40, 40, 40), font=font)
    img.save(path)


def draw_waveform(path: Path, csv_path: Path, columns: Sequence[str], title: str, encoding: str) -> None:
    width, height = 1300, 720
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = load_font(14)
    title_font = load_font(21)
    draw.text((45, 24), title, fill=(20, 20, 20), font=title_font)
    try:
        cols = list(dict.fromkeys([c for c in columns if c]))
        sample = pd.read_csv(csv_path, encoding=encoding, nrows=3000, low_memory=False, on_bad_lines="skip")
        sample.columns = [str(c).strip().replace("\ufeff", "") for c in sample.columns]
        cols = [c for c in cols if c in sample.columns][:5]
        if not cols:
            draw.text((45, 90), "No requested waveform columns found.", fill=(180, 0, 0), font=font)
            img.save(path)
            return
        left, top, right, bottom = 80, 100, width - 80, height - 90
        colors = [(40, 95, 160), (180, 85, 45), (75, 140, 85), (130, 90, 160), (40, 140, 150)]
        draw.rectangle((left, top, right, bottom), outline=(220, 220, 220))
        for idx, col in enumerate(cols):
            y = pd.to_numeric(sample[col], errors="coerce").to_numpy(dtype="float64", copy=False)
            valid = np.isfinite(y)
            if valid.sum() < 2:
                continue
            y2 = y.copy()
            med = np.nanmedian(y2)
            q1, q99 = np.nanpercentile(y2, [1, 99])
            denom = q99 - q1 if q99 > q1 else np.nanstd(y2)
            if not np.isfinite(denom) or denom <= 0:
                denom = 1.0
            norm = np.clip((y2 - med) / denom, -2, 2)
            xs = np.linspace(left, right, len(norm))
            y_center = top + (idx + 0.5) * (bottom - top) / len(cols)
            amp = (bottom - top) / (len(cols) * 2.6)
            points = [
                (float(x), float(y_center - n * amp))
                for x, n, ok in zip(xs, norm, valid)
                if ok
            ]
            if len(points) >= 2:
                draw.line(points, fill=colors[idx % len(colors)], width=2)
            draw.text((left, int(y_center - amp - 16)), col[:55], fill=colors[idx % len(colors)], font=font)
        draw.text((left, bottom + 15), csv_path.name, fill=(80, 80, 80), font=font)
    except Exception as exc:  # noqa: BLE001
        draw.text((45, 90), f"Waveform draw failed: {type(exc).__name__}: {exc}", fill=(180, 0, 0), font=font)
    img.save(path)


def make_figures(paths: Dict[str, Path], inventory_rows, timestamp_rows, overlap_rows, schema_rows) -> None:
    figures = paths["figures"]
    by_top = defaultdict(int)
    for row in inventory_rows:
        by_top[str(row.get("top_dir") or "unknown")] += 1
    ordered = sorted(by_top.items(), key=lambda kv: kv[1], reverse=True)
    draw_bar_chart(
        figures / "csv_count_by_top_dir.png",
        "CSV count by source directory",
        [k for k, _ in ordered],
        [float(v) for _, v in ordered],
    )

    draw_timeline_chart(
        figures / "modality_overlap_timeline_sample.png",
        overlap_rows,
        "Raw modality overlap sample by subject/session",
    )

    for modality in ["vehicle", "physio", "eeg"]:
        values = []
        for row in timestamp_rows:
            if row.get("modality") == modality and row.get("inferred_sampling_hz") != "":
                try:
                    values.append(float(row["inferred_sampling_hz"]))
                except Exception:
                    pass
        draw_histogram(figures / f"sampling_rate_hist_{modality}.png", f"Inferred sampling Hz: {modality}", values)

    schema_by_path = {row["relative_path"]: row for row in schema_rows}
    for modality, signal_cols, filename in [
        ("vehicle", VEHICLE_SIGNAL_HINTS, "raw_waveform_vehicle_sample.png"),
        ("physio", PHYSIO_SIGNAL_HINTS, "raw_waveform_physio_sample.png"),
        ("eeg", [f"LSLOutletStreamName-EEG|channel{i}" for i in range(4)], "raw_waveform_eeg_sample.png"),
    ]:
        candidates = [
            row
            for row in inventory_rows
            if row.get("raw_scope") and row.get("modality") == modality and schema_by_path.get(row["relative_path"])
        ]
        if candidates:
            inv = candidates[0]
            schema = schema_by_path[inv["relative_path"]]
            draw_waveform(
                figures / filename,
                Path(str(inv["absolute_path"])),
                signal_cols,
                f"Raw waveform sample: {modality}",
                str(schema.get("encoding") or "utf-8-sig"),
            )


def pct(x: float) -> str:
    try:
        return f"{float(x) * 100:.2f}%"
    except Exception:
        return ""


def summarize_counts(rows: Sequence[Dict[str, object]], key: str) -> str:
    counts = defaultdict(int)
    for row in rows:
        counts[str(row.get(key) or "unknown")] += 1
    parts = [f"{name}: {count}" for name, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    return "；".join(parts)


def build_old_flow_docs(paths: Dict[str, Path]) -> None:
    notes = paths["notes"]
    text = f"""# 阶段 0：旧流程冻结与重建准则

更新时间：{now_text()}

## 当前决定

旧流程不再作为新流程的默认真相。旧结果只保留三类用途：

1. 参考材料：帮助理解旧的车辆历史基线、连续风格迹象、生理/脑电候选路线、困难样本现象和模型失败模式。
2. 历史对照：后续新流程模型必须和旧 E2/E5A/E10C/G13/G14 等结果公平比较，但不能因为旧结果较好就继承旧数据处理假设。
3. 失败样本库：旧流程暴露出的错侧、幅值不足、尾段漂移、反向修正、多段修正和 G11 困难样本要转化成固定检查集。

## 可以参考的旧流程内容

- 纯车辆历史和事件上下文可以形成强基线，这一点值得保留为对照。
- 连续驾驶风格在旧实验中出现过增益迹象，但必须在新样本、无泄漏、分被试/置乱条件下重新验证。
- EMG 在旧实验中是非脑电生理信号里最强候选，但必须重审其窗口是否包含事件后动作结果。
- EEG 直接推理输入表现不稳，旧流程更支持“训练期教师/辅助任务/响应类型判断”的候选角色，但不能直接升级为结论。
- 旧预测图和 G11 困难样本归因有价值，应作为新评价体系的固定坏样本来源。
- 粗细双头、多候选轨迹、响应类型等结构探索只作为建模参考，不作为新流程必须继承的结构。

## 不能默认相信的旧流程内容

- 旧事件锚点是否对应真实原因发生时刻。
- 旧 200Hz 对齐和跨模态同步是否正确。
- 旧输入窗口、标签窗口、生理窗口是否严格因果。
- 旧标准化、PCA、连续风格提取、生理基线校正是否只在训练集 fit。
- 旧随机切分是否能证明泛化。
- 旧生理状态标签是否真的代表驾驶员内部状态。
- 旧 EMG 是否没有动作结果泄漏。
- 旧 EEG 教师收益是否来自有效知识而不是数据处理或评价偏差。
- 旧 RMSE 排名是否代表真实物理预测质量。

## 新流程硬性规则

- 先完成原始数据审计，再构建样本，再训练模型。
- 每个样本必须追溯到原始文件、原始时间戳、事件锚点、输入窗口、标签窗口和质量标记。
- 所有 scaler、PCA、风格统计、生理基线和阈值都必须记录 fit 范围，默认只允许训练集 fit。
- 评价必须包含方向、幅值、错侧、尾段、峰值时间、反向修正、多段修正、困难样本和分被试表现。
- 风格和生理的有效性必须经过强车辆基线、置乱/错位对照、分被试验证和物理指标共同确认。
- 在阶段 1 完成前，不宣称连续风格、生理、EMG、EEG 教师或任何新模型路线已经可靠有效。
"""
    (notes / "stage00_old_flow_freeze_and_rules_cn.md").write_text(text, encoding="utf-8")


def build_markdown_reports(
    paths: Dict[str, Path],
    inventory_rows: Sequence[Dict[str, object]],
    schema_rows: Sequence[Dict[str, object]],
    timestamp_rows: Sequence[Dict[str, object]],
    matrix_rows: Sequence[Dict[str, object]],
    overlap_rows: Sequence[Dict[str, object]],
    signal_rows: Sequence[Dict[str, object]],
    eeg_rows: Sequence[Dict[str, object]],
    sampling_rows: Sequence[Dict[str, object]],
) -> None:
    reports = paths["reports"]
    notes = paths["notes"]
    figures = paths["figures"]
    tables = paths["tables"]

    all_csv_count = len(inventory_rows)
    raw_csv_count = sum(1 for r in inventory_rows if r.get("raw_scope"))
    raw_sensor_csv_count = sum(1 for r in inventory_rows if r.get("raw_sensor_scope"))
    derived_under_raw_dirs = raw_csv_count - raw_sensor_csv_count
    raw_vehicle = sum(1 for r in inventory_rows if r.get("raw_sensor_scope") and r.get("modality") == "vehicle")
    raw_physio = sum(1 for r in inventory_rows if r.get("raw_sensor_scope") and r.get("modality") == "physio")
    raw_eeg = sum(1 for r in inventory_rows if r.get("raw_sensor_scope") and r.get("modality") == "eeg")
    all_three = sum(1 for r in matrix_rows if r.get("all_three_modalities_available"))
    overlap_ok = sum(1 for r in overlap_rows if r.get("status") == "overlap_ok")
    time_problem = [
        r for r in timestamp_rows
        if r.get("raw_sensor_scope")
        and (
            int(float(r.get("negative_dt_count") or 0)) > 0
            or int(float(r.get("large_gap_count") or 0)) > 0
            or float(r.get("time_valid_rate") or 0) < 0.999
        )
    ]
    low_valid_signals = [
        r for r in signal_rows
        if r.get("raw_sensor_scope") and r.get("valid_rate") != "" and float(r.get("valid_rate") or 0) < 0.95
    ]
    eeg_near_constant = [r for r in eeg_rows if r.get("raw_sensor_scope") and str(r.get("near_constant")) == "True"]

    summary = f"""# 阶段 1 原始数据审计总结

更新时间：{now_text()}

## 审计范围

- 原始数据根目录：`{paths["raw_root"].as_posix()}`
- 清单和深度审计范围：只覆盖 `原始车辆数据/<被试名>/*.csv`、`原始生理数据/<被试名>/*.csv`、`原始脑电数据/<被试名>/*.csv`。
- 明确不纳入：顶层全量记录文件、压缩包、处理后目录、对齐后目录，以及 `physio_features_v2_10Hz` 等派生特征目录。
- 本次未使用服务器，未读取服务器密码文件。

## 核心数量

- 本次纳入审计 CSV 文件数：{all_csv_count}
- 原始目录范围 CSV 文件数：{raw_csv_count}
- 原始传感器 CSV 文件数：{raw_sensor_csv_count}
- 原始目录内派生特征/处理后 CSV 文件数：{derived_under_raw_dirs}
- 原始车辆 CSV：{raw_vehicle}
- 原始生理 CSV：{raw_physio}
- 原始脑电 CSV：{raw_eeg}
- 被试/记录组合数：{len(matrix_rows)}
- 三模态齐全的组合数：{all_three}
- 至少两模态有可计算时间重叠且 overlap>0 的组合数：{overlap_ok}

## 主要发现

1. 原始数据路径存在且可扫描；车辆、生理、脑电三类原始 CSV 都能定位。
2. 已为所有 CSV 生成清单和哈希；后续样本 manifest 可以引用文件路径与 SHA256。
3. `原始生理数据` 目录内混有 10Hz 派生特征文件，已用 `raw_sensor_scope` 和 `derived_file` 分开标记，不能把这些派生特征当成原始传感器数据。
4. 时间戳初审已完成，存在 {len(time_problem)} 个原始传感器文件需要重点复核连续性、重复点、gap 或时间解析率。
5. 信号质量初审发现 {len(low_valid_signals)} 个原始传感器被抽查信号有效率低于 95%。
6. EEG 初审发现 {len(eeg_near_constant)} 个通道/文件组合接近常数，需在阶段 2/5 前结合伪迹规则复核。
7. 旧事件锚点只能作为历史参考；阶段 2 必须重新定义事件锚点、输入窗口、标签窗口和 causal setting。

## 暂定判断

当前结果支持继续进入“阶段 2：事件锚点与样本清单重建”的数据映射工作，但不支持直接训练新模型。继续条件是阶段 2 能为每个候选样本写清楚原始文件、时间范围、可用模态、质量标记和泄漏风险。

## 关键产物

- 文件清单：`{(tables / "raw_file_inventory.csv").as_posix()}`
- 字段报告：`{(tables / "raw_schema_report.csv").as_posix()}`
- 被试/记录/模态矩阵：`{(tables / "subject_session_modality_matrix.csv").as_posix()}`
- 时间连续性报告：`{(tables / "timestamp_continuity_report.csv").as_posix()}`
- 采样率报告：`{(tables / "sampling_rate_report.csv").as_posix()}`
- 模态重叠报告：`{(tables / "modality_overlap_report.csv").as_posix()}`
- 信号质量报告：`{(tables / "signal_quality_report.csv").as_posix()}`
- EEG 初审报告：`{(tables / "eeg_artifact_report.csv").as_posix()}`
- 泄漏风险报告：`{(tables / "leakage_risk_report.csv").as_posix()}`
- 审计图目录：`{figures.as_posix()}`
"""
    (reports / "raw_data_audit_summary_cn.md").write_text(summary, encoding="utf-8")

    user_summary = f"""# 阶段 1 用户查看版总结：原始数据审计

更新时间：{now_text()}

## 这个阶段为什么做

旧流程已经证明继续堆模型会遇到物理解释问题：有些预测趋势像，但方向、幅值、尾段回正、反向修正和困难样本并不可靠。因此本阶段先回到原始 CSV，检查数据本身、时间轴和跨模态同步是否值得继续。

## 这个阶段检查了什么

- 只扫描三个原始目录下被试名文件夹内的 CSV，并给每个纳入文件生成 SHA256 哈希。
- 对原始车辆、原始生理、原始脑电 CSV 读取字段、行数、时间范围、缺失率和时间戳连续性。
- 按被试和记录号整理车辆/生理/脑电是否齐全。
- 初步计算不同模态的时间重叠。
- 抽查车辆、生理和脑电关键波形，生成采样率分布图。
- 列出后续最容易造成“看起来有效但其实泄漏”的风险点。

## 目前发现了什么

- 能定位到原始车辆 CSV {raw_vehicle} 个、原始生理 CSV {raw_physio} 个、原始脑电 CSV {raw_eeg} 个。
- 原始目录范围总计 {raw_csv_count} 个 CSV，其中原始传感器 CSV {raw_sensor_csv_count} 个，派生特征/处理后 CSV {derived_under_raw_dirs} 个。
- 三模态齐全的被试/记录组合有 {all_three} 个。
- 当前至少两模态有正时间重叠的组合有 {overlap_ok} 个。
- 时间连续性、信号质量和 EEG 通道质量中都有需要复核的条目，不能直接跳到训练。

## 哪些结果可信

- 文件是否存在、文件大小、修改时间和 SHA256 哈希是可追溯的。
- 字段名、行数、时间范围、采样间隔初值是从原始 CSV 重新读取的。
- “旧锚点不能默认相信”这一判断可信，因为旧代码确实存在按响应峰值选锚点的逻辑，必须重新定义事件因果起点。

## 哪些结果还不能下结论

- 不能说连续风格一定有效。
- 不能说生理数据一定有效。
- 不能说 EMG 的旧收益没有动作结果泄漏。
- 不能说 EEG 教师一定有效。
- 不能说车辆/生理/脑电已经完全同步。
- 不能说 2 秒预测窗口一定覆盖了完整方向盘响应。

## 下一阶段是否可以继续

可以继续到阶段 2，但只能继续做“事件锚点与样本清单重建”，不能直接训练模型。阶段 2 的核心是把每个样本的锚点、输入窗口、标签窗口、模态可用性和泄漏风险写清楚。

## 推荐优先查看

- `{(reports / "raw_data_audit_summary_cn.md").as_posix()}`
- `{(tables / "subject_session_modality_matrix.csv").as_posix()}`
- `{(tables / "modality_overlap_report.csv").as_posix()}`
- `{(tables / "leakage_risk_report.csv").as_posix()}`
- `{(figures / "modality_overlap_timeline_sample.png").as_posix()}`
- `{(figures / "raw_waveform_vehicle_sample.png").as_posix()}`
- `{(figures / "raw_waveform_physio_sample.png").as_posix()}`
- `{(figures / "raw_waveform_eeg_sample.png").as_posix()}`
"""
    (reports / "stage01_user_summary_cn.md").write_text(user_summary, encoding="utf-8")

    status = f"""# R2E-Steering 项目总进度看板

更新时间：{now_text()}

## 当前阶段

阶段 1：原始数据审计已完成本地第一轮；下一步进入阶段 2 的事件锚点与样本清单重建。

## 当前正在做什么

把原始 CSV 的文件清单、字段、时间戳、采样率、模态重叠、信号质量和泄漏风险整理成可追溯产物。

## 已完成什么

- 阶段 0 旧流程冻结说明已生成。
- 新流程目录结构已建立。
- 三个原始目录下被试名文件夹内 CSV 清单和哈希已生成。
- 原始车辆/生理/脑电深度审计表已生成。
- 阶段 1 用户查看版中文总结已生成。

## 正在运行什么任务

当前没有后台审计或训练任务在运行。

## 服务器是否在运行

本阶段未使用服务器；未读取服务器密码文件。服务器状态未主动检查。

## 最近一次结果

- 本次纳入审计 CSV：{all_csv_count}
- 原始范围 CSV：{raw_csv_count}
- 原始车辆/生理/脑电：{raw_vehicle}/{raw_physio}/{raw_eeg}
- 三模态齐全组合：{all_three}
- overlap>0 组合：{overlap_ok}

## 当前最大风险

旧事件锚点和旧窗口定义不能直接继承；EMG 可能存在事件后动作结果泄漏；跨模态时间单位和重叠仍需在阶段 2 逐样本确认。

## 下一步准备做什么

1. 读取原始车辆事件线索和旧事件文件，重建候选事件锚点。
2. 生成 `samples_master.csv/jsonl` 的第一版字段设计。
3. 明确输入窗口、标签窗口和 causal setting。
4. 生成 split_table 和 dataset_version_card。

## 用户可以优先查看哪些文件

- `{(reports / "stage01_user_summary_cn.md").as_posix()}`
- `{(reports / "raw_data_audit_summary_cn.md").as_posix()}`
- `{(notes / "stage00_old_flow_freeze_and_rules_cn.md").as_posix()}`
- `{(tables / "leakage_risk_report.csv").as_posix()}`
- `{(tables / "subject_session_modality_matrix.csv").as_posix()}`
"""
    (notes / "PROJECT_STATUS_CN.md").write_text(status, encoding="utf-8")

    queue = f"""# 当前任务队列

更新时间：{now_text()}

## 正在做任务

- 阶段 2：事件锚点与样本清单重建准备。

## 已完成任务

- 建立新流程目录结构。
- 初始化透明化文件。
- 冻结旧流程并生成阶段 0 规则说明。
- 扫描原始 CSV 和处理后 CSV。
- 生成文件清单、哈希、字段报告、时间戳报告、采样率报告、模态重叠报告、信号质量报告、EEG 初审报告和泄漏风险报告。
- 生成阶段 1 用户查看版总结。

## 待做任务

- 阶段 2：重建事件锚点来源。
- 阶段 2：设计并生成 `samples_master.csv/jsonl`。
- 阶段 2：生成 split_table 和 dataset_version_card。
- 阶段 2：判断是否能进入无学习基线和强车辆基线。

## 阻塞任务

- 正式模型训练被阶段 2 阻塞：没有可追溯、无泄漏样本清单前不能训练。

## 可并行任务

- 旧事件文件检索。
- 原始车辆候选事件波形查看。
- 生理/脑电质量异常样例整理。
- 样本字段协议设计。

## 需要服务器的任务

- 暂无。后续全量重采样、样本构建和模型训练可能需要服务器。

## 不需要服务器的任务

- 当前阶段 2 的锚点规则设计、样本 manifest 字段设计和本地小规模验证。
"""
    (notes / "TASK_QUEUE_CN.md").write_text(queue, encoding="utf-8")

    server = f"""# 服务器运行记录

更新时间：{now_text()}

## 连接命令格式

需要连接服务器时，只能记录不含密码的 SSH 命令格式，例如：

`ssh -p <port> <user>@<host>`

禁止在本文件或任何项目文件中写入服务器密码。

## 当前状态

- 本阶段未使用服务器。
- 未读取服务器指令与密码文件。
- 当前没有已知后台服务器任务在运行。
- GPU/显存状态：未检查，因为本阶段不需要服务器。

## 运行记录

| 启动时间 | 关闭时间 | 运行任务 | screen/nohup 名称 | 远程项目路径 | 远程日志路径 | 本地拉回路径 | 是否还在跑 | GPU/显存摘要 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| - | - | 阶段 1 本地原始数据审计 | - | - | - | - | 否 | 未使用服务器 |
"""
    (notes / "SERVER_RUNS_CN.md").write_text(server, encoding="utf-8")

    artifact_index = f"""# 阶段产物索引

更新时间：{now_text()}

## 阶段 0：旧流程冻结与重建准则

- 用户/老师可读说明：`{(notes / "stage00_old_flow_freeze_and_rules_cn.md").as_posix()}`
- 旧流程参考定位：`04_project_logs/reports/progress/experiment_registry.md`、`04_project_logs/reports/physio_to_g14_progress_review_20260511/`
- GPTPro 原始数据重建建议：`{(reports / "gptpro_replies" / "20260512_rebuild_steering_reply_summary_cn.md").as_posix()}`

## 阶段 1：原始数据审计

- 用户查看版总结：`{(reports / "stage01_user_summary_cn.md").as_posix()}`
- 审计中文总结：`{(reports / "raw_data_audit_summary_cn.md").as_posix()}`
- 文件清单：`{(tables / "raw_file_inventory.csv").as_posix()}`
- 字段报告：`{(tables / "raw_schema_report.csv").as_posix()}`
- 被试/记录/模态矩阵：`{(tables / "subject_session_modality_matrix.csv").as_posix()}`
- 时间连续性报告：`{(tables / "timestamp_continuity_report.csv").as_posix()}`
- 采样率报告：`{(tables / "sampling_rate_report.csv").as_posix()}`
- 模态重叠报告：`{(tables / "modality_overlap_report.csv").as_posix()}`
- 信号质量报告：`{(tables / "signal_quality_report.csv").as_posix()}`
- EEG 初审报告：`{(tables / "eeg_artifact_report.csv").as_posix()}`
- 泄漏风险报告：`{(tables / "leakage_risk_report.csv").as_posix()}`
- 审计脚本入口：`{(paths["audit"] / "scripts" / "raw_csv_audit.py").as_posix()}`
- 审计图目录：`{figures.as_posix()}`
- 审计运行日志：`{(paths["logs"] / "raw_csv_audit.log").as_posix()}`

## 服务器日志

- 本阶段未使用服务器。

## 重要 Git commit

- 待提交。

## 适合用户/老师直接查看的材料

1. `{(reports / "stage01_user_summary_cn.md").as_posix()}`
2. `{(notes / "stage00_old_flow_freeze_and_rules_cn.md").as_posix()}`
3. `{(figures / "modality_overlap_timeline_sample.png").as_posix()}`
4. `{(figures / "raw_waveform_vehicle_sample.png").as_posix()}`
5. `{(tables / "leakage_risk_report.csv").as_posix()}`
"""
    (notes / "ARTIFACT_INDEX_CN.md").write_text(artifact_index, encoding="utf-8")

    daily = notes / "daily_logs" / f"{datetime.now().strftime('%Y-%m-%d')}.md"
    with daily.open("a", encoding="utf-8") as f:
        f.write(
            f"""

## {now_text()} 阶段 0/1 本地原始数据审计

- 为什么做：用户要求停止在旧流程上继续堆模型，先从原始 CSV 重建无泄漏、可追溯证据链。
- 做了什么：建立目录和透明化文件；冻结旧流程；只扫描三个原始目录下被试名文件夹内 CSV；对原始车辆/生理/脑电做字段、时间、采样率、模态重叠和信号质量审计；生成泄漏风险报告和用户查看版总结。
- 输入：`{paths["raw_root"].as_posix()}` 下 `原始车辆数据/<被试名>/*.csv`、`原始生理数据/<被试名>/*.csv`、`原始脑电数据/<被试名>/*.csv`；旧流程代码和历史报告仅作为风险参考。
- 输出：`{tables.as_posix()}`、`{figures.as_posix()}`、`{reports.as_posix()}`、`{notes.as_posix()}`。
- 当前结果：本次纳入审计 CSV {all_csv_count} 个，原始范围 CSV {raw_csv_count} 个，原始车辆/生理/脑电 {raw_vehicle}/{raw_physio}/{raw_eeg} 个，三模态齐全组合 {all_three} 个。
- 遇到问题：旧锚点和旧窗口定义不能默认相信；时间连续性和信号质量存在待复核条目。
- 是否需要用户决策：暂不需要。下一步应进入阶段 2 样本清单和锚点重建，不应直接训练。
"""
        )


def run(args: argparse.Namespace) -> None:
    paths = repo_paths(Path(__file__))
    ensure_dirs([paths["tables"], paths["figures"], paths["logs"], paths["reports"], paths["notes"], paths["daily_logs"]])
    log_path = paths["logs"] / "raw_csv_audit.log"

    def log(message: str) -> None:
        line = f"{now_text()} {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    if not paths["raw_root"].exists():
        raise FileNotFoundError(paths["raw_root"])

    log("[start] R2E-Steering raw CSV audit")
    log(f"[paths] raw_root={paths['raw_root']}")
    build_old_flow_docs(paths)

    inventory_rows = build_inventory(paths["raw_root"], args.hash_mode, log)
    write_csv(paths["tables"] / "raw_file_inventory.csv", inventory_rows)
    log(f"[write] raw_file_inventory.csv rows={len(inventory_rows)}")

    raw_files = [Path(str(r["absolute_path"])) for r in inventory_rows if r.get("raw_scope")]
    schema_rows: List[Dict[str, object]] = []
    timestamp_rows: List[Dict[str, object]] = []
    signal_rows: List[Dict[str, object]] = []
    eeg_rows: List[Dict[str, object]] = []
    error_rows: List[Dict[str, object]] = []

    for idx, path in enumerate(raw_files, start=1):
        try:
            schema, timestamp, signals, eeg = audit_one_csv(path, paths["raw_root"], args.chunksize)
            schema_rows.append(schema)
            timestamp_rows.append(timestamp)
            signal_rows.extend(signals)
            eeg_rows.extend(eeg)
        except Exception as exc:  # noqa: BLE001
            rel = rel_to(path, paths["raw_root"])
            log(f"[error] {rel}: {type(exc).__name__}: {exc}")
            error_rows.append(
                {
                    "relative_path": rel,
                    "top_dir": top_dir_for(path, paths["raw_root"]),
                    "subject": subject_for(path, paths["raw_root"]),
                    "session_stamp": session_stamp_for(path),
                    "modality": infer_modality(path),
                    "raw_sensor_scope": is_raw_sensor_scope(path, paths["raw_root"]),
                    "derived_file": is_derived_file(path, paths["raw_root"]),
                    "parse_status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        if idx % 10 == 0 or idx == len(raw_files):
            log(f"[deep-audit] {idx}/{len(raw_files)} raw-scope CSV files processed")

    schema_rows.extend(error_rows)
    write_csv(paths["tables"] / "raw_schema_report.csv", schema_rows)
    write_csv(paths["tables"] / "timestamp_continuity_report.csv", timestamp_rows)
    write_csv(paths["tables"] / "signal_quality_report.csv", signal_rows)
    write_csv(paths["tables"] / "eeg_artifact_report.csv", eeg_rows)

    sampling_rows = build_sampling_report(timestamp_rows)
    matrix_rows, overlap_rows = build_subject_session_matrix(inventory_rows, timestamp_rows)
    leakage_rows = build_leakage_risk_report(paths["project_root"])
    write_csv(paths["tables"] / "sampling_rate_report.csv", sampling_rows)
    write_csv(paths["tables"] / "subject_session_modality_matrix.csv", matrix_rows)
    write_csv(paths["tables"] / "modality_overlap_report.csv", overlap_rows)
    write_csv(paths["tables"] / "leakage_risk_report.csv", leakage_rows)
    log("[write] core stage 1 tables complete")

    make_figures(paths, inventory_rows, timestamp_rows, overlap_rows, schema_rows)
    log("[write] audit figures complete")

    build_markdown_reports(
        paths,
        inventory_rows,
        schema_rows,
        timestamp_rows,
        matrix_rows,
        overlap_rows,
        signal_rows,
        eeg_rows,
        sampling_rows,
    )
    log("[write] markdown reports and transparency files complete")
    log("[done] raw CSV audit finished")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hash-mode",
        choices=["all", "raw", "none"],
        default="all",
        help="Which CSV files receive full SHA256 hashes.",
    )
    parser.add_argument("--chunksize", type=int, default=200_000)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
