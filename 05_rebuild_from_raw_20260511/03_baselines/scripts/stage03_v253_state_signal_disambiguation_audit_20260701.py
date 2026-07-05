#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v253a state-signal disambiguation audit。

本轮目标：
- 承接 v252 的结论：一部分差样本属于“锚点前输入相似，但锚点后真实未来分叉”；
- 检查驾驶风格和生理信号是否能降低这种未来分叉；
- 本轮不训练预测模型，不改 v250/v251/v252 的模型，不做 test-based 选择；
- 只做近邻检索审计：在固定 vehicle 输入的基础上，加入因果状态特征后，同 delay train 近邻的未来分叉是否下降。

状态信号边界：
- 驾驶风格：从当前 raw vehicle CSV 重新提取 last60_guard3，窗口为 [observation_s - 63, observation_s - 3]，
  不接触直接输入窗口最后 3 秒，也不接触标签未来。
- 生理信号：从 1Hz 生理特征表提取 pre5_pre2 和 pre2_0，均不超过 observation_s。
- 旧 stage04 style 表与当前 v252 样本 event_uid/sample_id 不匹配，本轮不直接使用旧表。

解释边界：
- 若加入状态信号后 bad_top10 的 neighbor future divergence 明显下降，说明该状态信号对多未来消歧有价值；
- 若下降不明显或同 subject/recording 近邻率显著升高，则不能直接宣称生理/风格有效。
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

V252_SCRIPT = BASELINES / "scripts" / "stage03_v252_input_similarity_future_divergence_20260701.py"
STYLE_OLD_TABLE = REBUILD / "04_style" / "stage04_continuous_style_protocol_v0_1" / "tables" / "style_feature_candidate_wide.csv"
PHYSIO_1HZ = REBUILD / "06_physio_processing" / "physio_subject_collection_v1_20260603" / "tables" / "physio_features_1hz.csv"
PHYSIO_INVENTORY = REBUILD / "06_physio_processing" / "physio_subject_collection_v1_20260603" / "tables" / "physio_recording_inventory.csv"

OUT = BASELINES / "v253_state_signal_disambiguation_audit_20260701"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
ZIP_PATH = BASELINES / "v253_state_signal_disambiguation_audit_20260701_pack.zip"

SEED = 253
AUX_WEIGHTS = [0.25, 0.50]
KEY_GROUPS = [
    "vehicle_only",
    "vehicle_plus_style_w0.25",
    "vehicle_plus_style_w0.50",
    "vehicle_plus_physio_recent_w0.25",
    "vehicle_plus_physio_recent_w0.50",
    "vehicle_plus_physio_guarded_w0.50",
    "vehicle_plus_style_physio_w0.50",
]

STYLE_SIGNAL_MAP = {
    "speed_kmh": "zx1|v_km/h",
    "steering": "zx|SteeringWheel",
    "ay": "zx|ay",
    "yaw_rate": "zx|vyaw",
    "roll": "zx|roll",
    "brake": "zx|BrakePedal",
    "accelerator": "zx|AcceleratorPedal",
    "lane_curvature": "zx1|lanecurvatureXY",
    "lateral_distance": "zx1|lateraldistance",
}

PHYSIO_SIGNALS = [
    "HR_bpm",
    "EMG_RMS",
    "RESP_BPM",
    "RESP_Amplitude",
    "EDA_Tonic",
    "EDA_Phasic",
    "ECG_filt200",
    "EMG_filt200",
    "RESP_filt200",
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
    """按路径导入前序脚本，复用 v252 的数据和审计函数。"""

    if not path.exists():
        raise FileNotFoundError(f"缺少脚本：{path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法导入脚本：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V252 = import_module_from_path("stage03_v252_input_similarity_future_divergence_for_v253", V252_SCRIPT)


def ensure_dirs() -> None:
    """创建输出目录。"""

    for folder in (TABLES, FIGURES, REPORTS, LOGS):
        folder.mkdir(parents=True, exist_ok=True)


def clean_out_dir() -> None:
    """只清理 v253a 自己的输出。"""

    if OUT.exists():
        shutil.rmtree(OUT)
    ensure_dirs()
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """CSV 统一使用 utf-8-sig，便于中文 Windows 环境打开。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def file_sha256(path: Path) -> str:
    """计算文件 SHA256。"""

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finite_stats(values: np.ndarray, prefix: str) -> Dict[str, float]:
    """计算一个窗口内的稳健统计；全空时返回 NaN。"""

    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            f"{prefix}_valid_ratio": 0.0,
            f"{prefix}_mean": math.nan,
            f"{prefix}_std": math.nan,
            f"{prefix}_p10": math.nan,
            f"{prefix}_p50": math.nan,
            f"{prefix}_p90": math.nan,
            f"{prefix}_abs_mean": math.nan,
            f"{prefix}_abs_p95": math.nan,
            f"{prefix}_rms": math.nan,
        }
    return {
        f"{prefix}_valid_ratio": float(arr.size / max(1, len(values))),
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_p10": float(np.quantile(arr, 0.10)),
        f"{prefix}_p50": float(np.quantile(arr, 0.50)),
        f"{prefix}_p90": float(np.quantile(arr, 0.90)),
        f"{prefix}_abs_mean": float(np.mean(np.abs(arr))),
        f"{prefix}_abs_p95": float(np.quantile(np.abs(arr), 0.95)),
        f"{prefix}_rms": float(np.sqrt(np.mean(np.square(arr)))),
    }


def slope_feature(times: np.ndarray, values: np.ndarray) -> float:
    """用首末有限值估计简单斜率，避免小窗口线性拟合不稳。"""

    mask = np.isfinite(times) & np.isfinite(values)
    if int(mask.sum()) < 2:
        return math.nan
    t = times[mask]
    v = values[mask]
    dt = float(t[-1] - t[0])
    if abs(dt) < 1e-9:
        return math.nan
    return float((v[-1] - v[0]) / dt)


def session_stamp_from_recording(recording: str) -> str:
    """从 Entity_Recording_xxx 里取 session stamp。"""

    text = str(recording)
    return text.replace("Entity_Recording_", "")


def load_vehicle_recording(path: Path) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, bool]]:
    """
    读取 raw vehicle CSV 的必要列，并转成相对秒。

    这里读取原始车辆源是为了重新为当前 v252 样本构造风格特征；
    旧 style 表样本口径不匹配，不能直接使用。
    """

    header = pd.read_csv(path, nrows=0).columns.tolist()
    usecols = ["StorageTime"] + [col for col in STYLE_SIGNAL_MAP.values() if col in header]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    t = pd.to_datetime(df["StorageTime"], errors="coerce")
    if t.notna().sum() == 0:
        rel_s = np.arange(len(df), dtype=float) * 0.005
    else:
        first = t.dropna().iloc[0]
        rel_s = (t - first).dt.total_seconds().to_numpy(dtype=float)
    order = np.argsort(rel_s)
    rel_s = rel_s[order]

    values: Dict[str, np.ndarray] = {}
    available: Dict[str, bool] = {}
    for name, col in STYLE_SIGNAL_MAP.items():
        if col in df.columns:
            values[name] = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)[order]
            available[name] = True
        else:
            values[name] = np.full(len(df), np.nan, dtype=float)
            available[name] = False
    return rel_s, values, available


def window_slice(times: np.ndarray, start_s: float, end_s: float) -> slice:
    """按时间范围返回切片。"""

    left = int(np.searchsorted(times, start_s, side="left"))
    right = int(np.searchsorted(times, end_s, side="right"))
    return slice(left, right)


def build_style_features(manifest: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """为当前 rolling sample 重新提取 last60_guard3 驾驶风格特征。"""

    rows: List[Dict[str, object]] = []
    rec_rows: List[Dict[str, object]] = []
    cache: Dict[str, Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, bool]]] = {}

    for i, row in manifest.iterrows():
        raw_path = Path(str(row["raw_vehicle_csv"]))
        obs = float(row["observation_s"])
        start_s = max(0.0, obs - 63.0)
        end_s = max(0.0, obs - 3.0)
        out: Dict[str, object] = {
            "row_index": int(i),
            "style_window": "last60_guard3",
            "style_start_s": start_s,
            "style_end_s": end_s,
            "style_uses_post_observation": bool(end_s > obs + 1e-9),
            "style_overlaps_direct_input": bool(end_s > obs - 3.0 + 1e-9),
            "style_file_exists": raw_path.exists(),
        }
        if not raw_path.exists() or end_s <= start_s:
            out.update({"style_row_count": 0, "style_duration_s": 0.0, "style_status": "missing_or_too_early"})
            rows.append(out)
            continue

        key = str(raw_path)
        if key not in cache:
            try:
                cache[key] = load_vehicle_recording(raw_path)
                rec_rows.append({"raw_vehicle_csv": key, "style_load_status": "ok"})
            except Exception as exc:  # noqa: BLE001
                rec_rows.append({"raw_vehicle_csv": key, "style_load_status": f"failed:{type(exc).__name__}:{exc}"})
                cache[key] = (np.array([], dtype=float), {}, {})
        times, values, available = cache[key]
        if len(times) == 0:
            out.update({"style_row_count": 0, "style_duration_s": 0.0, "style_status": "load_failed"})
            rows.append(out)
            continue

        sl = window_slice(times, start_s, end_s)
        win_t = times[sl]
        out["style_row_count"] = int(len(win_t))
        out["style_duration_s"] = float(win_t[-1] - win_t[0]) if len(win_t) >= 2 else 0.0
        out["style_status"] = "ok" if len(win_t) >= 100 else "short_window"
        for name in STYLE_SIGNAL_MAP:
            vals = values[name][sl]
            out[f"style_{name}_available"] = bool(available.get(name, False))
            out.update(finite_stats(vals, f"style_{name}"))
            out[f"style_{name}_slope"] = slope_feature(win_t, vals)
        if len(win_t) >= 2:
            dt = np.diff(win_t)
            steer = values["steering"][sl]
            speed = values["speed_kmh"][sl]
            for name, vals in [("steering_rate", steer), ("speed_rate", speed)]:
                if len(vals) >= 2:
                    diff = np.diff(vals)
                    rate = np.divide(diff, dt, out=np.full_like(diff, np.nan, dtype=float), where=np.abs(dt) > 1e-9)
                    out.update(finite_stats(rate, f"style_{name}"))
        rows.append(out)

    return pd.DataFrame(rows), pd.DataFrame(rec_rows)


def load_physio_groups() -> Dict[Tuple[str, str], pd.DataFrame]:
    """读取 1Hz 生理特征并按 subject/session_stamp 分组。"""

    usecols = ["subject", "session_stamp", "time_bin_s"] + PHYSIO_SIGNALS
    df = pd.read_csv(PHYSIO_1HZ, usecols=usecols, encoding="utf-8-sig")
    df["time_bin_s"] = pd.to_numeric(df["time_bin_s"], errors="coerce")
    out: Dict[Tuple[str, str], pd.DataFrame] = {}
    for key, g in df.groupby(["subject", "session_stamp"], dropna=False):
        out[(str(key[0]), str(key[1]))] = g.sort_values("time_bin_s").reset_index(drop=True)
    return out


def build_physio_features(manifest: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """从 1Hz 生理表为当前 rolling sample 提取因果窗口特征。"""

    groups = load_physio_groups()
    inv = pd.read_csv(PHYSIO_INVENTORY, encoding="utf-8-sig")
    inv_keys = set(zip(inv["subject"].astype(str), inv["session_stamp"].astype(str)))
    rows: List[Dict[str, object]] = []

    for i, row in manifest.iterrows():
        subject = str(row["subject"])
        session = session_stamp_from_recording(str(row["recording"]))
        obs = float(row["observation_s"])
        key = (subject, session)
        g = groups.get(key)
        out: Dict[str, object] = {
            "row_index": int(i),
            "physio_recording_in_inventory": bool(key in inv_keys),
            "physio_recording_has_1hz": bool(g is not None and not g.empty),
            "physio_uses_post_observation": False,
        }
        if g is None or g.empty:
            out["physio_status"] = "missing_recording"
            rows.append(out)
            continue

        times = pd.to_numeric(g["time_bin_s"], errors="coerce").to_numpy(dtype=float)
        out["physio_status"] = "ok"
        windows = {
            "physio_guard_pre5_pre2": (max(0.0, obs - 5.0), max(0.0, obs - 2.0)),
            "physio_recent_pre2_0": (max(0.0, obs - 2.0), obs),
        }
        window_stats: Dict[str, Dict[str, float]] = {}
        for win_name, (start_s, end_s) in windows.items():
            sl = window_slice(times, start_s, end_s)
            win_t = times[sl]
            out[f"{win_name}_start_s"] = start_s
            out[f"{win_name}_end_s"] = end_s
            out[f"{win_name}_rows"] = int(len(win_t))
            out[f"{win_name}_duration_s"] = float(win_t[-1] - win_t[0]) if len(win_t) >= 2 else 0.0
            for sig in PHYSIO_SIGNALS:
                vals = pd.to_numeric(g[sig].iloc[sl], errors="coerce").to_numpy(dtype=float)
                stats = finite_stats(vals, f"{win_name}_{sig}")
                stats[f"{win_name}_{sig}_slope"] = slope_feature(win_t, vals)
                out.update(stats)
                window_stats[f"{win_name}_{sig}"] = stats

        # recent - guarded 差值只使用锚点前窗口，作为状态变化线索。
        for sig in PHYSIO_SIGNALS:
            recent_mean = out.get(f"physio_recent_pre2_0_{sig}_mean", math.nan)
            guard_mean = out.get(f"physio_guard_pre5_pre2_{sig}_mean", math.nan)
            recent_std = out.get(f"physio_recent_pre2_0_{sig}_std", math.nan)
            guard_std = out.get(f"physio_guard_pre5_pre2_{sig}_std", math.nan)
            out[f"physio_delta_recent_minus_guard_{sig}_mean"] = float(recent_mean - guard_mean) if np.isfinite(recent_mean) and np.isfinite(guard_mean) else math.nan
            out[f"physio_delta_recent_minus_guard_{sig}_std"] = float(recent_std - guard_std) if np.isfinite(recent_std) and np.isfinite(guard_std) else math.nan
        rows.append(out)

    return pd.DataFrame(rows), inv


def old_style_match_audit(manifest: pd.DataFrame) -> pd.DataFrame:
    """证明旧 style 表与当前样本口径不匹配，不能直接复用。"""

    if not STYLE_OLD_TABLE.exists():
        return pd.DataFrame([{"check": "old_style_table_exists", "value": False}])
    style = pd.read_csv(STYLE_OLD_TABLE, encoding="utf-8-sig", usecols=lambda c: c in {"sample_id", "event_uid", "subject", "session_stamp", "anchor_time_rel_s"})
    rows = [{"check": "old_style_table_rows", "value": int(len(style))}]
    for key in ["sample_id", "event_uid"]:
        if key in manifest.columns and key in style.columns:
            inter = set(manifest[key].astype(str)) & set(style[key].astype(str))
            rows.append({"check": f"{key}_intersection_count", "value": int(len(inter))})
    mf = manifest.copy()
    mf["session_stamp"] = mf["recording"].astype(str).map(session_stamp_from_recording)
    mf["anchor_round"] = pd.to_numeric(mf["original_anchor_s"], errors="coerce").round(3)
    st = style.copy()
    st["anchor_round"] = pd.to_numeric(st["anchor_time_rel_s"], errors="coerce").round(3)
    joined = mf.merge(
        st[["subject", "session_stamp", "anchor_round"]].drop_duplicates(),
        on=["subject", "session_stamp", "anchor_round"],
        how="left",
        indicator=True,
    )
    rows.append({"check": "subject_session_anchor_round_match_rows", "value": int(joined["_merge"].eq("both").sum())})
    return pd.DataFrame(rows)


def standardize_aux_features(
    df: pd.DataFrame,
    train_mask: np.ndarray,
    include_prefixes: Tuple[str, ...],
) -> Tuple[np.ndarray, pd.DataFrame]:
    """训练集拟合 median/std，对辅助特征做填充和标准化。"""

    numeric_cols = []
    for col in df.columns:
        if col == "row_index":
            continue
        if not any(col.startswith(prefix) for prefix in include_prefixes):
            continue
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            numeric_cols.append(col)

    rows: List[Dict[str, object]] = []
    mats: List[np.ndarray] = []
    for col in numeric_cols:
        raw = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        train_values = raw[train_mask & np.isfinite(raw)]
        if train_values.size < 5:
            continue
        med = float(np.median(train_values))
        std = float(np.std(train_values))
        if not np.isfinite(std) or std < 1e-9:
            continue
        missing = ~np.isfinite(raw)
        z = (np.where(missing, med, raw) - med) / std
        mats.append(z.astype(np.float32))
        if missing.mean() > 0:
            mats.append(missing.astype(np.float32))
        rows.append(
            {
                "feature_name": col,
                "train_finite_n": int(train_values.size),
                "all_missing_rate": float(missing.mean()),
                "train_median": med,
                "train_std": std,
                "added_missing_indicator": bool(missing.mean() > 0),
            }
        )

    if not mats:
        return np.zeros((len(df), 0), dtype=np.float32), pd.DataFrame(rows)
    return np.stack(mats, axis=1).astype(np.float32), pd.DataFrame(rows)


def augment(base_x: np.ndarray, aux: np.ndarray, weight: float) -> np.ndarray:
    """拼接辅助状态特征，并用 weight 控制辅助块贡献。"""

    if aux.shape[1] == 0:
        return base_x.copy()
    return np.concatenate([base_x.astype(np.float32), (aux.astype(np.float32) * float(weight))], axis=1)


def add_neighbor_identity_rates(audit: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """统计近邻是否更多来自同 subject / same recording，辅助判断身份代理风险。"""

    if audit.empty:
        return audit
    subjects = manifest["subject"].astype(str).to_numpy()
    recordings = manifest["recording"].astype(str).to_numpy()
    same_subject_rates = []
    same_recording_rates = []
    for _, row in audit.iterrows():
        qi = int(row["row_index"])
        neigh = V252.parse_neighbor_indices(str(row["neighbor_row_indices"]), k=V252.K_NEIGHBORS)
        if not neigh:
            same_subject_rates.append(math.nan)
            same_recording_rates.append(math.nan)
            continue
        same_subject_rates.append(float(np.mean(subjects[neigh] == subjects[qi])))
        same_recording_rates.append(float(np.mean(recordings[neigh] == recordings[qi])))
    out = audit.copy()
    out["neighbor_same_subject_rate"] = same_subject_rates
    out["neighbor_same_recording_rate"] = same_recording_rates
    return out


def run_feature_group_audits(
    manifest: pd.DataFrame,
    base_x: np.ndarray,
    style_x: np.ndarray,
    physio_recent_x: np.ndarray,
    physio_guard_x: np.ndarray,
    y_true: np.ndarray,
    pred_v241: np.ndarray,
    pred_v250: np.ndarray,
    sample_metrics: pd.DataFrame,
    valid_mask: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """对不同状态特征组合重复 v252 近邻未来分叉审计。"""

    group_to_x: Dict[str, np.ndarray] = {"vehicle_only": base_x}
    for w in AUX_WEIGHTS:
        group_to_x[f"vehicle_plus_style_w{w:.2f}"] = augment(base_x, style_x, w)
        group_to_x[f"vehicle_plus_physio_recent_w{w:.2f}"] = augment(base_x, physio_recent_x, w)
    group_to_x["vehicle_plus_physio_guarded_w0.50"] = augment(base_x, physio_guard_x, 0.50)
    both_aux = np.concatenate([style_x, physio_recent_x], axis=1) if style_x.shape[1] or physio_recent_x.shape[1] else np.zeros((len(base_x), 0), dtype=np.float32)
    group_to_x["vehicle_plus_style_physio_w0.50"] = augment(base_x, both_aux, 0.50)

    audit_frames: List[pd.DataFrame] = []
    summary_frames: List[pd.DataFrame] = []
    corr_frames: List[pd.DataFrame] = []
    overlap_frames: List[pd.DataFrame] = []

    for group_name in KEY_GROUPS:
        x = group_to_x[group_name]
        audit, _ = V252.compute_neighbor_audit(
            manifest=manifest,
            x_flat=x,
            y_true=y_true,
            pred_v241=pred_v241,
            pred_v250=pred_v250,
            sample_metrics=sample_metrics,
            valid_mask=valid_mask,
        )
        audit = add_neighbor_identity_rates(audit, manifest)
        audit.insert(0, "feature_group", group_name)
        summary = V252.summarize_by_bucket_delay(audit)
        summary.insert(0, "feature_group", group_name)
        corr = V252.error_ambiguity_correlations(audit)
        corr.insert(0, "feature_group", group_name)
        overlap = V252.overlap_table(audit)
        overlap.insert(0, "feature_group", group_name)
        audit_frames.append(audit)
        summary_frames.append(summary)
        corr_frames.append(corr)
        overlap_frames.append(overlap)

    return (
        pd.concat(audit_frames, ignore_index=True),
        pd.concat(summary_frames, ignore_index=True),
        pd.concat(corr_frames, ignore_index=True),
        pd.concat(overlap_frames, ignore_index=True),
    )


def key_comparison(summary: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    """生成与 vehicle_only 相比的关键指标变化表。"""

    keep = summary[
        summary["bucket"].isin(["all", "bad_top10_v250", "strong_steer", "observe_later_like"])
        & summary["delay_ms"].isin(["all_delays", "0", "600", "1000"])
    ].copy()
    id_summary = (
        audit.groupby("feature_group", dropna=False)
        .agg(
            neighbor_same_subject_rate_mean=("neighbor_same_subject_rate", "mean"),
            neighbor_same_recording_rate_mean=("neighbor_same_recording_rate", "mean"),
        )
        .reset_index()
    )
    keep = keep.merge(id_summary, on="feature_group", how="left")

    base = keep[keep["feature_group"].eq("vehicle_only")].copy()
    base_cols = [
        "bucket",
        "delay_ms",
        "neighbor_future_pairwise_rmse_mean",
        "neighbor_future_to_query_mean_rmse",
        "tail_rmse_v250_mean",
        "high_neighbor_divergence_q75_rate",
    ]
    base = base[base_cols].rename(
        columns={
            "neighbor_future_pairwise_rmse_mean": "base_pairwise",
            "neighbor_future_to_query_mean_rmse": "base_query_neighbor",
            "high_neighbor_divergence_q75_rate": "base_high_divergence_rate",
        }
    )
    merged = keep.merge(base, on=["bucket", "delay_ms"], how="left")
    merged["delta_pairwise_vs_vehicle_only"] = merged["neighbor_future_pairwise_rmse_mean"] - merged["base_pairwise"]
    merged["delta_query_neighbor_vs_vehicle_only"] = merged["neighbor_future_to_query_mean_rmse"] - merged["base_query_neighbor"]
    merged["delta_high_divergence_rate_vs_vehicle_only"] = merged["high_neighbor_divergence_q75_rate"] - merged["base_high_divergence_rate"]
    return merged


def plot_key_metrics(compare: pd.DataFrame) -> Path:
    """绘制 bad_top10_v250 all-delay 的消歧对比。"""

    path = FIGURES / "v253a_state_signal_badtop10_disambiguation.png"
    sub = compare[compare["bucket"].eq("bad_top10_v250") & compare["delay_ms"].eq("all_delays")].copy()
    if sub.empty:
        return path
    sub["label"] = sub["feature_group"].str.replace("vehicle_plus_", "+", regex=False).str.replace("vehicle_only", "vehicle", regex=False)
    x = np.arange(len(sub))
    fig, ax = plt.subplots(figsize=(14, 5.8))
    ax.bar(x - 0.18, sub["neighbor_future_pairwise_rmse_mean"], width=0.36, label="近邻之间未来分叉", color="#4C78A8")
    ax.bar(x + 0.18, sub["neighbor_future_to_query_mean_rmse"], width=0.36, label="query vs 近邻未来差距", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(sub["label"], rotation=25, ha="right")
    ax.set_ylabel("future RMSE / divergence")
    ax.set_title("v253a: 加入驾驶风格/生理后，bad_top10 近邻未来分叉是否下降")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_delta_metrics(compare: pd.DataFrame) -> Path:
    """绘制相对 vehicle_only 的 delta，负数代表消歧改善。"""

    path = FIGURES / "v253a_state_signal_delta_vs_vehicle_only.png"
    sub = compare[
        compare["bucket"].isin(["all", "bad_top10_v250", "strong_steer", "observe_later_like"])
        & compare["delay_ms"].eq("all_delays")
        & ~compare["feature_group"].eq("vehicle_only")
    ].copy()
    if sub.empty:
        return path
    labels = sub["feature_group"].drop_duplicates().tolist()
    buckets = ["all", "bad_top10_v250", "strong_steer", "observe_later_like"]
    x = np.arange(len(labels))
    width = 0.18
    fig, ax = plt.subplots(figsize=(15, 6))
    for i, bucket in enumerate(buckets):
        vals = (
            sub[sub["bucket"].eq(bucket)]
            .set_index("feature_group")
            .reindex(labels)["delta_query_neighbor_vs_vehicle_only"]
            .to_numpy(dtype=float)
        )
        ax.bar(x + (i - 1.5) * width, vals, width=width, label=bucket)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("vehicle_plus_", "+") for s in labels], rotation=25, ha="right")
    ax.set_ylabel("query-vs-neighbor future RMSE delta（负数=更可辨识）")
    ax.set_title("v253a: 状态信号相对 vehicle-only 的消歧变化")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(
    feature_audit: pd.DataFrame,
    style_match: pd.DataFrame,
    style_summary: pd.DataFrame,
    physio_summary: pd.DataFrame,
    compare: pd.DataFrame,
    corr: pd.DataFrame,
    figures: List[Path],
) -> None:
    """写中文报告。"""

    lines: List[str] = []
    lines.append("# v253a 驾驶风格/生理信号消歧审计")
    lines.append("")
    lines.append("## 本轮问题")
    lines.append("")
    lines.append("v252 已经证明一部分差样本属于“锚点前输入相似，但锚点后真实未来分叉”。本轮检查驾驶风格和生理信号是否能让近邻未来更集中。")
    lines.append("")
    lines.append("## 固定边界")
    lines.append("")
    lines.append("- 不训练新预测模型，不改 v250/v251/v252。")
    lines.append("- 不使用 test 选择模型；所有结果只作为状态信号是否值得进入下一步模型的审计证据。")
    lines.append("- 旧 stage04 style 表与当前 v252 样本不匹配，本轮只记录其不可直接复用，不直接使用。")
    lines.append("- 驾驶风格从当前 raw vehicle 重新提取 `last60_guard3`，窗口截止到 `observation_s - 3s`。")
    lines.append("- 生理信号从 1Hz 表提取 `pre5_pre2` 和 `pre2_0`，窗口均不超过 `observation_s`。")
    lines.append("")
    lines.append("## 旧 style 表匹配审计")
    lines.append("")
    lines.append(style_match.to_markdown(index=False))
    lines.append("")
    lines.append("## 特征可用性")
    lines.append("")
    lines.append(feature_audit.to_markdown(index=False))
    lines.append("")
    lines.append("## 驾驶风格提取摘要")
    lines.append("")
    lines.append(style_summary.to_markdown(index=False))
    lines.append("")
    lines.append("## 生理提取摘要")
    lines.append("")
    lines.append(physio_summary.to_markdown(index=False))
    lines.append("")
    lines.append("## 关键对比")
    lines.append("")
    key = compare[
        compare["bucket"].isin(["all", "bad_top10_v250", "strong_steer", "observe_later_like"])
        & compare["delay_ms"].isin(["all_delays", "0"])
    ].copy()
    show_cols = [
        "feature_group",
        "bucket",
        "delay_ms",
        "n",
        "neighbor_future_pairwise_rmse_mean",
        "neighbor_future_to_query_mean_rmse",
        "delta_query_neighbor_vs_vehicle_only",
        "high_neighbor_divergence_q75_rate",
        "neighbor_same_subject_rate_mean",
        "neighbor_same_recording_rate_mean",
    ]
    lines.append(key[show_cols].to_markdown(index=False))
    lines.append("")
    lines.append("## 相关性摘要")
    lines.append("")
    corr_key = corr[
        corr["subset"].eq("all_delays")
        & corr["x_metric"].isin(["neighbor_future_pairwise_rmse_mean", "neighbor_future_to_query_mean_rmse", "neighbor_input_distance_mean"])
        & corr["y_metric"].eq("tail_rmse_v250")
    ].copy()
    lines.append(corr_key.to_markdown(index=False))
    lines.append("")
    lines.append("## 判读方式")
    lines.append("")
    lines.append("- `delta_query_neighbor_vs_vehicle_only < 0`：加入状态信号后，同输入近邻的真实未来更接近 query，说明有消歧价值。")
    lines.append("- 如果 delta 很小或为正，说明该状态信号当前表示没有帮助消歧。")
    lines.append("- 如果 same subject / same recording rate 明显升高，需要警惕状态信号只是把近邻检索推向身份或 session 匹配。")
    lines.append("")
    lines.append("## 关键图")
    lines.append("")
    for fig in figures:
        lines.append(f"- `{fig.relative_to(OUT)}`")
    lines.append("")
    (REPORTS / "v253_state_signal_disambiguation_audit_cn.md").write_text("\n".join(lines), encoding="utf-8")


def write_input_hashes() -> None:
    """记录关键输入哈希。"""

    paths = [V252_SCRIPT, STYLE_OLD_TABLE, PHYSIO_1HZ, PHYSIO_INVENTORY]
    rows = []
    for path in paths:
        if path.exists():
            rows.append({"path": str(path), "sha256": file_sha256(path), "bytes": int(path.stat().st_size)})
    write_csv(pd.DataFrame(rows), LOGS / "input_file_hashes.csv")


def write_file_inventory() -> None:
    """记录输出文件清单。"""

    rows = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            rows.append({"relative_path": str(path.relative_to(OUT)), "bytes": int(path.stat().st_size)})
    write_csv(pd.DataFrame(rows), LOGS / "file_inventory.csv")


def make_zip() -> str | None:
    """打包 v253a 关键产物。"""

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(Path(__file__), arcname=f"scripts/{Path(__file__).name}")
        for folder in [TABLES, FIGURES, REPORTS, LOGS]:
            for path in folder.rglob("*"):
                if path.is_file():
                    zf.write(path, arcname=str(path.relative_to(OUT)))
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        return zf.testzip()


def build_guardrail(
    split_check: pd.DataFrame,
    style_df: pd.DataFrame,
    physio_df: pd.DataFrame,
    zip_test: str | None,
) -> Dict[str, object]:
    """生成本轮约束检查。"""

    cross = int(split_check["same_event_uid_cross_split"].sum()) if "same_event_uid_cross_split" in split_check.columns else 0
    return {
        "pass": bool(
            cross == 0
            and zip_test is None
            and not bool(style_df["style_uses_post_observation"].fillna(False).any())
            and not bool(style_df["style_overlaps_direct_input"].fillna(False).any())
            and not bool(physio_df["physio_uses_post_observation"].fillna(False).any())
        ),
        "same_event_uid_cross_split_count": cross,
        "retrained_model": False,
        "test_used_for_model_selection": False,
        "old_style_table_used_as_feature": False,
        "style_window": "last60_guard3_[observation_s-63,observation_s-3]",
        "physio_windows": ["pre5_pre2", "pre2_0"],
        "style_uses_post_observation_any": bool(style_df["style_uses_post_observation"].fillna(False).any()),
        "style_overlaps_direct_input_any": bool(style_df["style_overlaps_direct_input"].fillna(False).any()),
        "physio_uses_post_observation_any": bool(physio_df["physio_uses_post_observation"].fillna(False).any()),
        "zip_testzip": zip_test,
    }


def main() -> None:
    clean_out_dir()
    print("[v253a] state-signal disambiguation audit")
    print("[v253a] no model training; fixed v250/v252 vehicle input")

    loaded = V252.load_fixed_inputs()
    data = loaded["data"]
    manifest = data.manifest.copy()
    base_x = loaded["x_flat"].astype(np.float32)
    y_true = loaded["y_true"]
    pred_v241 = loaded["pred_v241"]
    pred_v250 = loaded["pred_v250"]
    sample_metrics = loaded["sample_metrics"]
    valid_mask = loaded["valid_mask"]
    split_check = loaded["split_check"]
    train_mask = manifest["split"].astype(str).eq("train").to_numpy()

    print("[v253a] audit old style table matching")
    style_match = old_style_match_audit(manifest)

    print("[v253a] build current driving-style features from raw vehicle")
    style_df, style_recording_log = build_style_features(manifest)
    print("[v253a] build current physio features from 1Hz table")
    physio_df, physio_inventory = build_physio_features(manifest)

    style_x, style_scaler = standardize_aux_features(style_df, train_mask, include_prefixes=("style_",))
    physio_recent_x, physio_recent_scaler = standardize_aux_features(
        physio_df,
        train_mask,
        include_prefixes=("physio_recent_pre2_0_", "physio_delta_recent_minus_guard_", "physio_recording_"),
    )
    physio_guard_x, physio_guard_scaler = standardize_aux_features(
        physio_df,
        train_mask,
        include_prefixes=("physio_guard_pre5_pre2_", "physio_recording_"),
    )

    feature_audit = pd.DataFrame(
        [
            {"feature_block": "vehicle_base_v250_minimal", "n_features": int(base_x.shape[1]), "source": "v250_minimal_lateral7 hist+road+phase"},
            {"feature_block": "driving_style_last60_guard3", "n_features": int(style_x.shape[1]), "source": "raw vehicle recomputed for current v252 samples"},
            {"feature_block": "physio_recent_pre2_0_and_delta", "n_features": int(physio_recent_x.shape[1]), "source": "physio_features_1hz.csv causal pre-observation"},
            {"feature_block": "physio_guard_pre5_pre2", "n_features": int(physio_guard_x.shape[1]), "source": "physio_features_1hz.csv guarded baseline"},
        ]
    )

    style_summary = pd.DataFrame(
        [
            {
                "n_samples": int(len(style_df)),
                "ok_rate": float(style_df["style_status"].eq("ok").mean()),
                "short_or_missing_rate": float((~style_df["style_status"].eq("ok")).mean()),
                "mean_row_count": float(style_df["style_row_count"].mean()),
                "post_observation_any": bool(style_df["style_uses_post_observation"].fillna(False).any()),
                "overlap_direct_input_any": bool(style_df["style_overlaps_direct_input"].fillna(False).any()),
            }
        ]
    )
    physio_summary = pd.DataFrame(
        [
            {
                "n_samples": int(len(physio_df)),
                "recording_inventory_match_rate": float(physio_df["physio_recording_in_inventory"].mean()),
                "has_1hz_rate": float(physio_df["physio_recording_has_1hz"].mean()),
                "recent_pre2_0_rows_mean": float(physio_df["physio_recent_pre2_0_rows"].mean()),
                "guard_pre5_pre2_rows_mean": float(physio_df["physio_guard_pre5_pre2_rows"].mean()),
                "post_observation_any": bool(physio_df["physio_uses_post_observation"].fillna(False).any()),
            }
        ]
    )

    print("[v253a] run neighbor divergence audits")
    audit, summary, corr, overlap = run_feature_group_audits(
        manifest=manifest,
        base_x=base_x,
        style_x=style_x,
        physio_recent_x=physio_recent_x,
        physio_guard_x=physio_guard_x,
        y_true=y_true,
        pred_v241=pred_v241,
        pred_v250=pred_v250,
        sample_metrics=sample_metrics,
        valid_mask=valid_mask,
    )
    compare = key_comparison(summary, audit)

    print("[v253a] write outputs")
    write_csv(style_match, TABLES / "v253a_old_style_match_audit.csv")
    write_csv(style_df, TABLES / "v253a_current_style_features_last60_guard3.csv")
    write_csv(style_recording_log, LOGS / "v253a_style_recording_load_log.csv")
    write_csv(physio_df, TABLES / "v253a_current_physio_features_1hz.csv")
    write_csv(feature_audit, TABLES / "v253a_feature_block_audit.csv")
    write_csv(style_summary, TABLES / "v253a_style_feature_summary.csv")
    write_csv(physio_summary, TABLES / "v253a_physio_feature_summary.csv")
    write_csv(style_scaler, TABLES / "v253a_style_train_scaler.csv")
    write_csv(physio_recent_scaler, TABLES / "v253a_physio_recent_train_scaler.csv")
    write_csv(physio_guard_scaler, TABLES / "v253a_physio_guard_train_scaler.csv")
    write_csv(audit, TABLES / "v253a_neighbor_divergence_by_feature_group.csv")
    write_csv(summary, TABLES / "v253a_summary_by_feature_group_bucket_delay.csv")
    write_csv(corr, TABLES / "v253a_error_ambiguity_correlation_by_feature_group.csv")
    write_csv(overlap, TABLES / "v253a_high_ambiguity_error_overlap_by_feature_group.csv")
    write_csv(compare, TABLES / "v253a_key_comparison_vs_vehicle_only.csv")
    write_csv(split_check, TABLES / "v253a_split_integrity_check.csv")

    figures = [plot_key_metrics(compare), plot_delta_metrics(compare)]
    write_input_hashes()
    write_file_inventory()
    write_report(feature_audit, style_match, style_summary, physio_summary, compare, corr, figures)
    write_file_inventory()
    zip_test = make_zip()
    guardrail = build_guardrail(split_check, style_df, physio_df, zip_test)
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    if not bool(guardrail["pass"]):
        raise AssertionError("v253a guardrail 未通过：\n" + json.dumps(guardrail, ensure_ascii=False, indent=2))
    write_file_inventory()

    key = compare[
        compare["bucket"].eq("bad_top10_v250")
        & compare["delay_ms"].eq("all_delays")
        & compare["feature_group"].isin(KEY_GROUPS)
    ].copy()
    base = key[key["feature_group"].eq("vehicle_only")].iloc[0]
    best = key.sort_values("neighbor_future_to_query_mean_rmse").iloc[0]
    print(
        "[v253a] bad_top10 vehicle query_neighbor={:.6f}; best_group={} query_neighbor={:.6f} delta={:.6f}".format(
            float(base["neighbor_future_to_query_mean_rmse"]),
            str(best["feature_group"]),
            float(best["neighbor_future_to_query_mean_rmse"]),
            float(best["delta_query_neighbor_vs_vehicle_only"]),
        )
    )
    print(f"[v253a] report={REPORTS / 'v253_state_signal_disambiguation_audit_cn.md'}")
    print(f"[v253a] zip={ZIP_PATH}")


if __name__ == "__main__":
    main()
