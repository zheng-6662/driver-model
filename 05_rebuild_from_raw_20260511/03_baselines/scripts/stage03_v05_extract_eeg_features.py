#!/usr/bin/env python
"""Extract v0.5-aligned EEG features from cleaned EEG recordings.

This script intentionally does not use the old roll-peak EEG feature tables.
It reads the v0.5 manifest, maps each sample to its recording-level cleaned EEG
FIF file, and extracts EEG features from a causal pre-anchor window.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import mne
import numpy as np
import pandas as pd


BANDS = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

ROI = {
    "frontal": ["Fp1", "Fp2", "AF3", "AF4", "F3", "Fz", "F7", "F8", "FC1", "FC5", "FC6"],
    "temporal": ["T7", "T8"],
    "occipital": ["O1", "O2", "Oz", "PO3", "PO4"],
    "central": ["C3", "C4", "Cz", "CP1", "CP2", "CP5", "CP6"],
    "parietal": ["P3", "P4", "P7", "P8", "Pz"],
}

EPS = 1e-12

COMPAT_FEATURES = [
    "Frontal_alpha_asym",
    "Occipital_ta_beta",
    "Frontal_ta_beta",
    "Temporal_ta_beta",
    "Occipital_alpha_abs",
    "Temporal_gamma_rel",
    "Occipital_gamma_rel",
    "Frontal_gamma_rel",
]


@dataclass
class Paths:
    root: Path
    manifest: Path
    output_dir: Path
    subject_root: Path
    raw_eeg_root: Path


def parse_args() -> argparse.Namespace:
    root = Path.cwd()
    default_manifest = (
        root
        / "05_rebuild_from_raw_20260511"
        / "03_processed_datasets"
        / "stage03_v05_server_aligned_subject_oldflow_fair09"
        / "tables"
        / "oldflow_fair09_vehicle_only_server_aligned_v05_subject_split_manifest.csv"
    )
    default_output = (
        root
        / "05_rebuild_from_raw_20260511"
        / "03_baselines"
        / "stage03_v05_eeg_features"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--manifest", type=Path, default=default_manifest)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--history-sec", type=float, default=2.0)
    parser.add_argument("--baseline-start-sec", type=float, default=8.0)
    parser.add_argument("--baseline-end-sec", type=float, default=4.0)
    parser.add_argument("--limit-samples", type=int, default=0)
    parser.add_argument("--limit-recordings", type=int, default=0)
    parser.add_argument("--feature-prefix", type=str, default="eeg_pre2s")
    parser.add_argument("--skip-features", action="store_true")
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> Paths:
    root = args.root.resolve()
    return Paths(
        root=root,
        manifest=args.manifest if args.manifest.is_absolute() else root / args.manifest,
        output_dir=args.output_dir if args.output_dir.is_absolute() else root / args.output_dir,
        subject_root=root / "01_datasets" / "多模态数据" / "被试数据集合",
        raw_eeg_root=root / "01_datasets" / "数据预处理" / "原始脑电数据",
    )


def recording_prefix(recording_id: str) -> str:
    rid = str(recording_id)
    if rid.startswith("Entity_Recording_"):
        return rid
    return f"Entity_Recording_{rid}"


def clean_eeg_path(paths: Paths, subj: str, recording_id: str) -> Path:
    prefix = recording_prefix(recording_id)
    return (
        paths.subject_root
        / str(subj)
        / "eeg_clean"
        / f"{prefix}_eeg_raw_clean_resamp200_ica_final_qc.fif"
    )


def raw_eeg_path(paths: Paths, subj: str, recording_id: str) -> Path:
    prefix = recording_prefix(recording_id)
    return paths.raw_eeg_root / str(subj) / f"{prefix}_eeg.csv"


def vehicle_path_from_manifest(paths: Paths, row: pd.Series) -> Optional[Path]:
    for col in ["raw_vehicle_file_before_cleaning", "vehicle_file"]:
        if col not in row or pd.isna(row[col]):
            continue
        p = str(row[col])
        p = p.replace("/root/autodl-tmp/data_process", str(paths.root).replace("\\", "/"))
        candidate = Path(p)
        if candidate.exists():
            return candidate
    return None


def read_first_storage_time(csv_path: Path) -> Optional[pd.Timestamp]:
    try:
        df = pd.read_csv(csv_path, usecols=["StorageTime"], nrows=20)
        s = df["StorageTime"].dropna()
        if len(s) == 0:
            return None
        return pd.to_datetime(s.iloc[0])
    except Exception:
        return None


def infer_raw_schema(csv_path: Path) -> Dict[str, object]:
    out: Dict[str, object] = {
        "raw_eeg_exists": csv_path.exists(),
        "raw_eeg_size_mb": np.nan,
        "raw_eeg_col_count": np.nan,
        "raw_eeg_channel_count": np.nan,
        "raw_eeg_accel_count": np.nan,
        "raw_eeg_first_storage_time": "",
    }
    if not csv_path.exists():
        return out
    out["raw_eeg_size_mb"] = round(csv_path.stat().st_size / 1024 / 1024, 3)
    try:
        df = pd.read_csv(csv_path, nrows=5)
        cols = list(df.columns)
        out["raw_eeg_col_count"] = len(cols)
        out["raw_eeg_channel_count"] = sum("EEG|channel" in c for c in cols)
        out["raw_eeg_accel_count"] = sum("Accelerometer|channel" in c for c in cols)
        if "StorageTime" in df.columns:
            vals = df["StorageTime"].dropna()
            if len(vals):
                out["raw_eeg_first_storage_time"] = str(pd.to_datetime(vals.iloc[0]))
    except Exception as exc:
        out["raw_eeg_schema_error"] = repr(exc)
    return out


def build_recording_inventory(paths: Paths, manifest: pd.DataFrame, limit_recordings: int = 0) -> pd.DataFrame:
    rows = []
    recs = manifest[["subj", "recording_id"]].drop_duplicates().sort_values(["subj", "recording_id"])
    if limit_recordings:
        recs = recs.head(limit_recordings)
    for _, rec in recs.iterrows():
        subj = str(rec["subj"])
        rid = str(rec["recording_id"])
        fif = clean_eeg_path(paths, subj, rid)
        raw_csv = raw_eeg_path(paths, subj, rid)
        row = {
            "subj": subj,
            "recording_id": rid,
            "clean_eeg_fif": str(fif),
            "clean_eeg_exists": fif.exists(),
            "raw_eeg_csv": str(raw_csv),
        }
        row.update(infer_raw_schema(raw_csv))
        if fif.exists():
            try:
                raw = mne.io.read_raw_fif(fif, preload=False, verbose="ERROR")
                row.update(
                    {
                        "clean_sfreq": float(raw.info["sfreq"]),
                        "clean_nchan": int(raw.info["nchan"]),
                        "clean_duration_s": round(raw.n_times / raw.info["sfreq"], 3),
                        "clean_bad_count": len(raw.info.get("bads", [])),
                        "clean_bad_list": ";".join(raw.info.get("bads", [])),
                        "clean_ch_names": ";".join(raw.ch_names),
                    }
                )
            except Exception as exc:
                row["clean_read_error"] = repr(exc)
        rows.append(row)
    return pd.DataFrame(rows)


def bandpower(psd: np.ndarray, freqs: np.ndarray, fmin: float, fmax: float) -> float:
    idx = (freqs >= fmin) & (freqs < fmax)
    if not np.any(idx):
        return np.nan
    return float(np.trapezoid(psd[idx], freqs[idx]))


def safe_mean(values: np.ndarray, ch_names: List[str], wanted: Iterable[str]) -> float:
    idx = [ch_names.index(ch) for ch in wanted if ch in ch_names]
    if not idx:
        return np.nan
    return float(np.nanmean(values[idx]))


def pick_asym_pair(ch_names: List[str]) -> Tuple[Optional[str], Optional[str], str]:
    for right, left, tag in [("F4", "F3", "F4F3"), ("AF4", "AF3", "AF4AF3"), ("F8", "F7", "F8F7")]:
        if right in ch_names and left in ch_names:
            return right, left, tag
    return None, None, "NA"


def compute_window_features(
    raw: mne.io.BaseRaw,
    start_s: float,
    end_s: float,
    prefix: str,
) -> Tuple[Dict[str, float], str]:
    sfreq = float(raw.info["sfreq"])
    start_idx = max(0, int(round(start_s * sfreq)))
    end_idx = min(raw.n_times, int(round(end_s * sfreq)))
    out: Dict[str, float] = {
        f"{prefix}_window_start_s": start_idx / sfreq,
        f"{prefix}_window_end_s": end_idx / sfreq,
        f"{prefix}_window_len_s": (end_idx - start_idx) / sfreq,
    }
    if end_idx - start_idx < max(64, int(0.5 * sfreq)):
        return out, "window_too_short"

    ch_all = list(raw.ch_names)
    needed = sorted(set(sum(ROI.values(), []) + ["F3", "F4", "AF3", "AF4", "F7", "F8"]))
    picks = [ch_all.index(ch) for ch in needed if ch in ch_all]
    actual_ch = [ch_all[i] for i in picks]
    if len(picks) < 4:
        return out, "too_few_channels"

    data = raw.get_data(picks=picks, start=start_idx, stop=end_idx)
    finite_ratio = float(np.isfinite(data).mean())
    out[f"{prefix}_finite_ratio"] = finite_ratio
    if finite_ratio < 0.95:
        return out, "too_many_nonfinite"
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    nper = int(min(256, data.shape[1]))
    if nper < 64:
        return out, "window_too_short"
    psd, freqs = mne.time_frequency.psd_array_welch(
        data,
        sfreq=sfreq,
        fmin=4.0,
        fmax=45.0,
        n_fft=nper,
        n_per_seg=nper,
        n_overlap=max(0, nper // 2),
        average="mean",
        verbose=False,
    )

    bp: Dict[str, np.ndarray] = {}
    for band, (f1, f2) in BANDS.items():
        bp[band] = np.array([bandpower(psd[i], freqs, f1, f2) for i in range(psd.shape[0])])

    total = bp["theta"] + bp["alpha"] + bp["beta"] + bp["gamma"] + EPS
    for roi_name, roi_channels in ROI.items():
        for band in BANDS:
            val = safe_mean(bp[band], actual_ch, roi_channels)
            out[f"{prefix}_{roi_name}_{band}_abs"] = val
            out[f"{prefix}_{roi_name}_{band}_log"] = float(np.log(val + EPS)) if np.isfinite(val) else np.nan
        theta = safe_mean(bp["theta"], actual_ch, roi_channels)
        alpha = safe_mean(bp["alpha"], actual_ch, roi_channels)
        beta = safe_mean(bp["beta"], actual_ch, roi_channels)
        gamma = safe_mean(bp["gamma"], actual_ch, roi_channels)
        tot = safe_mean(total, actual_ch, roi_channels)
        out[f"{prefix}_{roi_name}_theta_alpha_over_beta"] = (theta + alpha) / (beta + EPS) if np.isfinite(theta + alpha + beta) else np.nan
        out[f"{prefix}_{roi_name}_theta_beta"] = theta / (beta + EPS) if np.isfinite(theta + beta) else np.nan
        out[f"{prefix}_{roi_name}_alpha_beta"] = alpha / (beta + EPS) if np.isfinite(alpha + beta) else np.nan
        out[f"{prefix}_{roi_name}_gamma_rel"] = gamma / (tot + EPS) if np.isfinite(gamma + tot) else np.nan

    right, left, tag = pick_asym_pair(actual_ch)
    if right is not None and left is not None:
        ridx = actual_ch.index(right)
        lidx = actual_ch.index(left)
        asym = np.log(bp["alpha"][ridx] + EPS) - np.log(bp["alpha"][lidx] + EPS)
    else:
        asym = np.nan
    out[f"{prefix}_frontal_alpha_asym_{tag}"] = float(asym) if np.isfinite(asym) else np.nan

    # Compatibility columns matching old event-level EEG names.
    out["Frontal_alpha_asym"] = out[f"{prefix}_frontal_alpha_asym_{tag}"]
    out["Occipital_ta_beta"] = out[f"{prefix}_occipital_theta_alpha_over_beta"]
    out["Frontal_ta_beta"] = out[f"{prefix}_frontal_theta_alpha_over_beta"]
    out["Temporal_ta_beta"] = out[f"{prefix}_temporal_theta_alpha_over_beta"]
    out["Occipital_alpha_abs"] = out[f"{prefix}_occipital_alpha_abs"]
    out["Temporal_gamma_rel"] = out[f"{prefix}_temporal_gamma_rel"]
    out["Occipital_gamma_rel"] = out[f"{prefix}_occipital_gamma_rel"]
    out["Frontal_gamma_rel"] = out[f"{prefix}_frontal_gamma_rel"]

    return out, "ok"


def add_baseline_delta(row: Dict[str, object], prefix: str, base_prefix: str) -> None:
    for name in COMPAT_FEATURES:
        c = row.get(name, np.nan)
        b = row.get(f"{base_prefix}_{name}", np.nan)
        row[f"{prefix}_minus_baseline_{name}"] = c - b if np.isfinite(c) and np.isfinite(b) else np.nan


def extract_features(
    paths: Paths,
    manifest: pd.DataFrame,
    history_sec: float,
    baseline_start_sec: float,
    baseline_end_sec: float,
    limit_samples: int,
    limit_recordings: int,
    feature_prefix: str,
) -> pd.DataFrame:
    rows = []
    grouped = list(manifest.groupby(["subj", "recording_id"], sort=True))
    if limit_recordings:
        grouped = grouped[:limit_recordings]
    processed_samples = 0
    for (subj, rid), group in grouped:
        fif = clean_eeg_path(paths, str(subj), str(rid))
        if not fif.exists():
            for _, sample in group.iterrows():
                rows.append(base_sample_row(sample, fif, "missing_eeg_fif"))
            continue
        try:
            raw = mne.io.read_raw_fif(fif, preload=False, verbose="ERROR")
        except Exception as exc:
            for _, sample in group.iterrows():
                rows.append(base_sample_row(sample, fif, f"read_error:{exc!r}"))
            continue
        for _, sample in group.iterrows():
            if limit_samples and processed_samples >= limit_samples:
                return pd.DataFrame(rows)
            processed_samples += 1
            anchor_s = float(sample["anchor_s"])
            row = base_sample_row(sample, fif, "pending")
            row["eeg_sfreq"] = float(raw.info["sfreq"])
            row["eeg_nchan"] = int(raw.info["nchan"])
            row["eeg_duration_s"] = raw.n_times / float(raw.info["sfreq"])
            if anchor_s < history_sec:
                row["eeg_status"] = "not_enough_pre_anchor"
                rows.append(row)
                continue
            if anchor_s > row["eeg_duration_s"]:
                row["eeg_status"] = "anchor_beyond_eeg_duration"
                rows.append(row)
                continue

            feats, status = compute_window_features(
                raw,
                anchor_s - history_sec,
                anchor_s,
                feature_prefix,
            )
            row.update(feats)
            row["eeg_status"] = status

            base_status = "not_requested"
            if anchor_s >= baseline_start_sec and baseline_start_sec > baseline_end_sec:
                base_feats, base_status = compute_window_features(
                    raw,
                    anchor_s - baseline_start_sec,
                    anchor_s - baseline_end_sec,
                    "eeg_base",
                )
                base_compat = {}
                for k, v in base_feats.items():
                    if k in COMPAT_FEATURES:
                        base_compat[f"eeg_base_{k}"] = v
                row.update({k: v for k, v in base_feats.items() if k not in COMPAT_FEATURES})
                row.update(base_compat)
                add_baseline_delta(row, feature_prefix, "eeg_base")
            else:
                base_status = "not_enough_pre_baseline"
            row["eeg_baseline_status"] = base_status
            rows.append(row)
    return pd.DataFrame(rows)


def base_sample_row(sample: pd.Series, fif: Path, status: str) -> Dict[str, object]:
    cols = [
        "protocol_version",
        "sample_key",
        "subj",
        "split",
        "file",
        "recording_id",
        "event_idx",
        "episode_id",
        "anchor_s",
        "anchor_idx",
        "event_type",
        "road_type_anchor",
        "v05_source_group",
        "v04_label",
        "v04_label_cn",
    ]
    out = {c: sample[c] for c in cols if c in sample.index}
    out["clean_eeg_fif"] = str(fif)
    out["eeg_status"] = status
    return out


def write_report(
    out_dir: Path,
    manifest: pd.DataFrame,
    inventory: pd.DataFrame,
    features: Optional[pd.DataFrame],
    args: argparse.Namespace,
) -> None:
    report = out_dir / "stage03_v05_eeg_feature_extraction_report_cn.md"
    lines: List[str] = []
    lines.append("# v0.5 脑电数据审计与特征提取说明\n")
    lines.append("## 本轮结论\n")
    lines.append("- 原始脑电 CSV 用于确认字段、时间戳和原始通道结构。")
    lines.append("- 建议建模优先使用已经完成清洗、重采样和 ICA 处理的 `*_eeg_raw_clean_resamp200_ica_final_qc.fif`。")
    lines.append("- 旧脑电特征表是按横滚峰值前 2 秒提取，不能直接用于 v0.5 新锚点。")
    lines.append("- 本脚本按 v0.5 manifest 的 `anchor_s` 重新提取锚点前脑电特征，默认窗口为 `[anchor_s-2s, anchor_s)`，不使用未来标签窗口。\n")
    lines.append("## 数据概况\n")
    lines.append(f"- v0.5 manifest 样本数：{len(manifest)}")
    lines.append(f"- v0.5 记录数：{manifest[['subj', 'recording_id']].drop_duplicates().shape[0]}")
    lines.append(f"- 有清洗脑电 FIF 的记录数：{int(inventory['clean_eeg_exists'].sum())}/{len(inventory)}")
    if "raw_eeg_exists" in inventory:
        lines.append(f"- 有原始脑电 CSV 的记录数：{int(inventory['raw_eeg_exists'].sum())}/{len(inventory)}")
    if "clean_sfreq" in inventory:
        sf = inventory["clean_sfreq"].dropna().value_counts().to_dict()
        lines.append(f"- 清洗后脑电采样率分布：{sf}")
    lines.append("\n## 原始脑电字段\n")
    lines.append("- 典型原始 CSV 包含 `ID`、`StorageTime`、32 个 `LSLOutletStreamName-EEG|channelX` 脑电通道，以及 3 个 `Accelerometer` 加速度通道。")
    lines.append("- 旧预处理代码会丢弃开头 EEG 全 NaN 行、插值少量 NaN、用 `StorageTime` 估计真实采样率、1-40Hz 带通、50Hz 工频滤波、平均参考、ICA 清理，并重采样到 200Hz。")
    lines.append("\n## 特征设计\n")
    lines.append("- 频段：theta 4-8Hz、alpha 8-13Hz、beta 13-30Hz、gamma 30-45Hz。")
    lines.append("- 区域：额叶、颞叶、枕叶、中央区、顶叶。")
    lines.append("- 输出包括各区域频段功率、对数功率、theta+alpha/beta、theta/beta、alpha/beta、gamma 相对功率、额叶 alpha 不对称，以及和旧流程兼容的 8 个核心特征。")
    lines.append("- 同时输出窗口长度、有限值比例、通道数、清洗文件路径和状态字段，方便排查缺失或异常样本。")
    if features is not None and len(features):
        lines.append("\n## 提取结果\n")
        lines.append(f"- 已输出特征样本数：{len(features)}")
        lines.append("- `eeg_status` 分布：")
        for k, v in features["eeg_status"].value_counts(dropna=False).items():
            lines.append(f"  - {k}: {v}")
        if "split" in features.columns:
            lines.append("- 分 split 可用情况：")
            table = features.assign(ok=features["eeg_status"].eq("ok")).groupby("split")["ok"].agg(["sum", "count"])
            for split, r in table.iterrows():
                lines.append(f"  - {split}: ok {int(r['sum'])}/{int(r['count'])}")
    lines.append("\n## 下一步建议\n")
    lines.append("1. 先把本特征表接入 v0.5 机制实验的可用性检查。")
    lines.append("2. 如果 `eeg_status=ok` 覆盖 train/val/test 都足够，再跑 `车辆+脑电`、`车辆+连续风格+脑电` 和脑电教师版本。")
    lines.append("3. 如果覆盖不足，先查缺失记录是否没有清洗 FIF，不能直接用旧 roll-peak 特征补。")
    lines.append("4. 如果要使用 `anchor_s` 后 0.5 秒早期动作窗口，必须单独命名为 early-window 特征，不能和严格 pre-anchor 特征混在一起。")
    report.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = paths.output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(paths.manifest)
    inventory = build_recording_inventory(paths, manifest, args.limit_recordings)
    inventory.to_csv(tables_dir / "v05_eeg_recording_inventory.csv", index=False, encoding="utf-8-sig")

    features: Optional[pd.DataFrame] = None
    if not args.skip_features:
        manifest_for_extract = manifest
        if args.limit_samples:
            # keep group order stable but only extract the requested number.
            manifest_for_extract = manifest.copy()
        features = extract_features(
            paths,
            manifest_for_extract,
            history_sec=args.history_sec,
            baseline_start_sec=args.baseline_start_sec,
            baseline_end_sec=args.baseline_end_sec,
            limit_samples=args.limit_samples,
            limit_recordings=args.limit_recordings,
            feature_prefix=args.feature_prefix,
        )
        features.to_csv(
            tables_dir / f"v05_eeg_features_pre_anchor_hist{int(args.history_sec)}s.csv",
            index=False,
            encoding="utf-8-sig",
        )
        summary = (
            features.groupby(["split", "eeg_status"], dropna=False)
            .size()
            .reset_index(name="n")
            if "split" in features.columns
            else features.groupby(["eeg_status"], dropna=False).size().reset_index(name="n")
        )
        summary.to_csv(tables_dir / "v05_eeg_feature_availability_summary.csv", index=False, encoding="utf-8-sig")

    write_report(paths.output_dir, manifest, inventory, features, args)
    print(f"Saved EEG audit/features to: {paths.output_dir}")


if __name__ == "__main__":
    main()
