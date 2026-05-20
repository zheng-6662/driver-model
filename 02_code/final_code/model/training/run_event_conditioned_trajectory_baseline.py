from __future__ import annotations

import argparse
import os
import random
import time
from glob import glob
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from event_conditioned_eval_support import (
    annotate_event_meta,
    build_primary_selection_bundle,
    structure_aware_selection_key,
)
from event_conditioned_baseline_model import (
    EventConditionedDataset,
    EventConditionedTrajectoryModel,
    RESPONSE_CANDIDATE_CLASS_KEY,
    build_event_schema_targets,
    build_event_teacher_from_batch,
    build_response_type_targets,
    compute_event_loss,
    compute_response_type_loss,
    count_parameters,
    masked_mse,
    subset_array_dict,
)
from future_steer_speed_subjectsplit_masked import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LR,
    FUTURE_LEN,
    RESULT_ROOT,
    _make_sample,
    normalize_inputs,
    save_json,
)


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[3]
PROTOCOL_DIR = THIS_DIR / "protocol_allphase_control_v2_context_full2s"
DEFAULT_MANIFEST = PROTOCOL_DIR / "sample_manifest.csv"
DEFAULT_DRIVER_STYLE_VECTOR_CSV = PROJECT_ROOT / "04_project_logs" / "reports" / "style_probe_artifacts" / "driver_style_vectors.csv"
DEFAULT_V05_EEG_FEATURE_TABLE = (
    PROJECT_ROOT
    / "05_rebuild_from_raw_20260511"
    / "03_baselines"
    / "stage03_v05_eeg_features"
    / "tables"
    / "v05_eeg_features_pre_anchor_hist2s.csv"
)
RUN_ROOT = RESULT_ROOT.parent / "event_conditioned_runs"

# Tail amplitude penalty
TAIL_START = 200          # step index where tail begins (1.0 s at 200 Hz)
W_TAIL_AMP = 0.3          # penalty weight; adjust after seeing results
TEACHER_BASE_NAMES = [
    "hr",
    "eda_tonic",
    "eda_phasic",
    "emg_rms",
    "alpha_asym",
    "occ_ta_beta",
    "frontal_ta_beta",
    "temporal_ta_beta",
    "occ_alpha_abs",
    "temporal_gamma_rel",
    "occ_gamma_rel",
    "frontal_gamma_rel",
]
EEG_FEAT_KEYS = [
    "Occipital_ta_beta",
    "Frontal_ta_beta",
    "Temporal_ta_beta",
    "Occipital_alpha_abs",
    "Temporal_gamma_rel",
    "Occipital_gamma_rel",
    "Frontal_gamma_rel",
]
EEG_HIST_SEC = 2
EPS = 1e-6
PHYSIO_CURRENT_SAMPLES = 600
PHYSIO_BASELINE_GAP_SAMPLES = 600
PHYSIO_BASELINE_MAX_SAMPLES = 6000
PHYSIO_BASELINE_MIN_SAMPLES = 600
PHYSIO_ONLY_DIM = 4
_V05_EEG_FEATURE_CACHE: dict[str, Any] | None = None


def _basename_from_any_path(raw_path: str | Path) -> str:
    return str(raw_path).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def resolve_data_file_path(raw_path: str | Path, subject: str | None = None, kind: str | None = None) -> Path:
    """Resolve manifest paths across local Windows and remote Linux project roots."""
    path = Path(str(raw_path))
    if path.exists():
        return path
    raw = str(raw_path).replace("\\", "/")
    candidates: list[Path] = []
    marker = "data_process/"
    if marker in raw:
        candidates.append(PROJECT_ROOT / raw.split(marker, 1)[1])
    marker = "01_datasets/"
    if marker in raw:
        rel = raw.split(marker, 1)[1]
        candidates.append(PROJECT_ROOT / "01_datasets" / rel)
        candidates.append(PROJECT_ROOT.parent / "01_datasets" / rel)
    marker = "datasetprocess/"
    if marker in raw:
        rel = raw.split(marker, 1)[1]
        candidates.append(PROJECT_ROOT / "01_datasets" / rel)
        candidates.append(PROJECT_ROOT.parent / "01_datasets" / rel)
    candidates.extend([PROJECT_ROOT / raw, Path.cwd() / raw])
    if subject and kind:
        filename = _basename_from_any_path(raw_path)
        pattern = str(PROJECT_ROOT / "01_datasets" / "**" / str(subject) / str(kind) / filename)
        candidates.extend(Path(match) for match in sorted(glob(pattern, recursive=True)))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def set_determinism(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _sample_by_split(meta_df: pd.DataFrame, split: str, n_keep: int | None, seed: int) -> pd.DataFrame:
    split_df = meta_df[meta_df["split"].astype(str) == split]
    if n_keep is None or n_keep <= 0 or len(split_df) <= n_keep:
        return split_df
    return split_df.sample(n=n_keep, random_state=seed)


def normalize_subject_id(value: Any) -> str:
    return str(value).strip().lower()


def _subject_from_row(row: pd.Series) -> str:
    if "subj" in row and pd.notna(row["subj"]):
        return normalize_subject_id(row["subj"])
    if "subject_id" in row and pd.notna(row["subject_id"]):
        return normalize_subject_id(row["subject_id"])
    vehicle_file = str(row.get("vehicle_file", ""))
    parts = Path(vehicle_file).parts
    return normalize_subject_id(parts[-3] if len(parts) >= 3 else "unknown")


def find_col(cols: list[str], candidates: list[str]) -> str | None:
    lower_map = {str(c).lower(): c for c in cols}
    for cand in candidates:
        key = str(cand).lower()
        if key in lower_map:
            return lower_map[key]
    return None


def safe_nanmean(values: np.ndarray, default: float = np.nan) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float(default)
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return float(default)
    return float(valid.mean())


def safe_nanstd(values: np.ndarray, default: float = np.nan) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float(default)
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return float(default)
    return float(valid.std())


def recording_prefix_from_vehicle_file(vehicle_file: str) -> str:
    name = os.path.basename(str(vehicle_file))
    for suffix in [
        "_vehicle_aligned_cleaned_roadtype_labeled.csv",
        "_vehicle_aligned_cleaned.csv",
        "_vehicle.csv",
        ".csv",
    ]:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return os.path.splitext(name)[0].replace("_vehicle_aligned_cleaned", "").replace("_vehicle", "")


def infer_physio_file(vehicle_file: str) -> str | None:
    subj_dir = os.path.dirname(os.path.dirname(str(vehicle_file)))
    physio_dir = os.path.join(subj_dir, "physio")
    if not os.path.isdir(physio_dir):
        return None
    prefix = recording_prefix_from_vehicle_file(vehicle_file)
    matches = glob(os.path.join(physio_dir, prefix + "*physio*.csv"))
    if matches:
        return matches[0]
    fallback = glob(os.path.join(physio_dir, "*.csv"))
    return fallback[0] if fallback else None


def infer_eeg_event_feature_file(vehicle_file: str) -> str | None:
    subj_dir = os.path.dirname(os.path.dirname(str(vehicle_file)))
    eeg_dir = os.path.join(subj_dir, "eeg_clean")
    if not os.path.isdir(eeg_dir):
        return None
    prefix = recording_prefix_from_vehicle_file(vehicle_file)
    suffix = f"_eeg_event_features_rollpeak_hist{int(EEG_HIST_SEC)}s.csv"
    matches = glob(os.path.join(eeg_dir, prefix + "*" + suffix))
    return matches[0] if matches else None


def build_eeg_feat_map(eeg_event_csv: str | None) -> dict[int, np.ndarray]:
    if eeg_event_csv is None or not os.path.exists(eeg_event_csv):
        return {}
    df = pd.read_csv(eeg_event_csv)
    if "event_row_index" not in df.columns:
        return {}
    asym_cols = [c for c in df.columns if str(c).startswith("Frontal_alpha_asym_")]
    asym_col = asym_cols[0] if asym_cols else None
    df = df.set_index("event_row_index")
    out: dict[int, np.ndarray] = {}
    for idx, row in df.iterrows():
        feats: list[float] = []
        feats.append(float(row[asym_col]) if asym_col is not None and asym_col in row else np.nan)
        for name in EEG_FEAT_KEYS:
            feats.append(float(row[name]) if name in row else np.nan)
        out[int(idx)] = np.asarray(feats, dtype=np.float32)
    return out


def _eeg_feature_vector_from_row(row: pd.Series, prefix: str = "") -> np.ndarray | None:
    names = ["Frontal_alpha_asym", *EEG_FEAT_KEYS]
    values: list[float] = []
    for name in names:
        col = f"{prefix}{name}" if prefix else name
        if col not in row.index:
            return None
        try:
            values.append(float(row[col]))
        except Exception:
            values.append(np.nan)
    arr = np.asarray(values, dtype=np.float32)
    return arr if np.isfinite(arr).any() else None


def build_v05_eeg_feature_cache(feature_csv: str | Path = DEFAULT_V05_EEG_FEATURE_TABLE) -> dict[str, Any]:
    global _V05_EEG_FEATURE_CACHE
    csv_path = str(feature_csv)
    if _V05_EEG_FEATURE_CACHE is not None and _V05_EEG_FEATURE_CACHE.get("source_file") == csv_path:
        return _V05_EEG_FEATURE_CACHE
    cache: dict[str, Any] = {
        "source_file": csv_path,
        "available": False,
        "by_sample_key": {},
        "delta_by_sample_key": {},
        "by_record_event": {},
        "delta_by_record_event": {},
        "row_count": 0,
        "ok_count": 0,
    }
    path = Path(csv_path)
    if not path.exists():
        _V05_EEG_FEATURE_CACHE = cache
        return cache
    df = pd.read_csv(path)
    if df.empty:
        _V05_EEG_FEATURE_CACHE = cache
        return cache
    ok_df = df[df.get("eeg_status", "ok").astype(str).eq("ok")].copy() if "eeg_status" in df.columns else df.copy()
    by_sample: dict[str, np.ndarray] = {}
    delta_by_sample: dict[str, np.ndarray] = {}
    by_record_event: dict[tuple[str, str, int], np.ndarray] = {}
    delta_by_record_event: dict[tuple[str, str, int], np.ndarray] = {}
    for _, feat_row in ok_df.iterrows():
        current = _eeg_feature_vector_from_row(feat_row)
        delta = _eeg_feature_vector_from_row(feat_row, prefix="eeg_pre2s_minus_baseline_")
        sample_key = str(feat_row.get("sample_key", "") or "").strip()
        if current is not None and sample_key:
            by_sample[sample_key] = current
        if delta is not None and sample_key:
            delta_by_sample[sample_key] = delta
        try:
            record_key = (
                normalize_subject_id(str(feat_row.get("subj", ""))),
                str(feat_row.get("recording_id", "")),
                int(feat_row.get("event_idx", -999999)),
            )
        except Exception:
            record_key = ("", "", -999999)
        if current is not None and record_key[0] and record_key[1] and record_key[2] != -999999:
            by_record_event[record_key] = current
        if delta is not None and record_key[0] and record_key[1] and record_key[2] != -999999:
            delta_by_record_event[record_key] = delta
    cache.update(
        {
            "available": bool(by_sample or by_record_event),
            "by_sample_key": by_sample,
            "delta_by_sample_key": delta_by_sample,
            "by_record_event": by_record_event,
            "delta_by_record_event": delta_by_record_event,
            "row_count": int(len(df)),
            "ok_count": int(len(ok_df)),
        }
    )
    _V05_EEG_FEATURE_CACHE = cache
    return cache


def get_v05_eeg_features_for_row(row: pd.Series) -> tuple[np.ndarray | None, np.ndarray | None]:
    cache = build_v05_eeg_feature_cache()
    if not cache.get("available", False):
        return None, None
    sample_key = str(row.get("sample_key", "") or "").strip()
    current = cache["by_sample_key"].get(sample_key) if sample_key else None
    delta = cache["delta_by_sample_key"].get(sample_key) if sample_key else None
    if current is not None or delta is not None:
        return current, delta
    try:
        record_key = (
            _subject_from_row(row),
            str(row.get("recording_id", "")),
            int(row.get("event_idx", -999999)),
        )
    except Exception:
        return None, None
    return cache["by_record_event"].get(record_key), cache["delta_by_record_event"].get(record_key)


def _physio_base_columns(df_p: pd.DataFrame) -> tuple[str, str, str, str] | None:
    cols = df_p.columns.tolist()
    col_hr = find_col(cols, ["HR", "HR_bpm", "hr", "hr_bpm"])
    col_tonic = find_col(cols, ["EDA_Tonic", "eda_tonic", "Tonic"])
    col_phasic = find_col(cols, ["EDA_Phasic", "eda_phasic", "Phasic"])
    col_emg = find_col(cols, ["EMG_RMS", "emg_rms", "EMG"])
    if col_hr is None or col_tonic is None or col_phasic is None or col_emg is None:
        return None
    return col_hr, col_tonic, col_phasic, col_emg


def _physio_segment_means(seg: pd.DataFrame, columns: tuple[str, str, str, str]) -> np.ndarray:
    return np.asarray(
        [safe_nanmean(seg[col].to_numpy(dtype=np.float64)) for col in columns],
        dtype=np.float32,
    )


def extract_physio_window_means(df_p: pd.DataFrame | None, anchor_idx: int) -> np.ndarray | None:
    if df_p is None or len(df_p) < anchor_idx or anchor_idx - PHYSIO_CURRENT_SAMPLES < 0:
        return None
    columns = _physio_base_columns(df_p)
    if columns is None:
        return None
    seg = df_p.iloc[anchor_idx - PHYSIO_CURRENT_SAMPLES: anchor_idx]
    if seg.empty:
        return None
    return _physio_segment_means(seg, columns)


def extract_physio_local_delta(df_p: pd.DataFrame | None, anchor_idx: int) -> np.ndarray | None:
    if df_p is None or len(df_p) < anchor_idx or anchor_idx - PHYSIO_CURRENT_SAMPLES < 0:
        return None
    columns = _physio_base_columns(df_p)
    if columns is None:
        return None
    current_seg = df_p.iloc[anchor_idx - PHYSIO_CURRENT_SAMPLES: anchor_idx]
    baseline_end = anchor_idx - PHYSIO_BASELINE_GAP_SAMPLES
    if current_seg.empty or baseline_end <= 0:
        return None
    baseline_start = max(0, baseline_end - PHYSIO_BASELINE_MAX_SAMPLES)
    baseline_seg = df_p.iloc[baseline_start:baseline_end]
    if len(baseline_seg) < PHYSIO_BASELINE_MIN_SAMPLES:
        return None
    current_mean = _physio_segment_means(current_seg, columns)
    baseline_mean = _physio_segment_means(baseline_seg, columns)
    baseline_std = np.asarray(
        [safe_nanstd(baseline_seg[col].to_numpy(dtype=np.float64)) for col in columns],
        dtype=np.float32,
    )
    valid = np.isfinite(current_mean) & np.isfinite(baseline_mean) & np.isfinite(baseline_std)
    baseline_std = baseline_std.copy()
    baseline_std[baseline_std < EPS] = 1.0
    delta = np.full((4,), np.nan, dtype=np.float32)
    delta[valid] = ((current_mean[valid] - baseline_mean[valid]) / baseline_std[valid]).astype(np.float32)
    return delta if np.isfinite(delta).any() else None


def compute_eeg_prior_event_delta(eeg_map: dict[int, np.ndarray], event_idx: int, min_prev: int = 2) -> np.ndarray | None:
    current = eeg_map.get(int(event_idx))
    if current is None:
        return None
    prev_keys = [key for key in sorted(eeg_map) if int(key) < int(event_idx)]
    if not prev_keys:
        return None
    prev = np.stack([np.asarray(eeg_map[key], dtype=np.float32) for key in prev_keys], axis=0)
    current_arr = np.asarray(current, dtype=np.float32)
    delta = np.full_like(current_arr, np.nan, dtype=np.float32)
    for col in range(current_arr.shape[0]):
        values = prev[:, col]
        valid_values = values[np.isfinite(values)]
        if valid_values.size < int(min_prev) or not np.isfinite(current_arr[col]):
            continue
        sd = float(valid_values.std())
        if sd < EPS:
            sd = 1.0
        delta[col] = float((current_arr[col] - valid_values.mean()) / sd)
    return delta if np.isfinite(delta).any() else None


def fit_pca_projection(train_x: np.ndarray, out_dim: int) -> dict[str, np.ndarray | int]:
    x_full = np.asarray(train_x, dtype=np.float64)
    if x_full.ndim != 2:
        raise ValueError(f"Expected 2D train_x for PCA, got shape={x_full.shape}")
    valid_mask = np.isfinite(x_full).all(axis=0)
    if not np.any(valid_mask):
        raise ValueError("No valid feature dims available for PCA")
    x = x_full[:, valid_mask]
    mean = np.mean(x, axis=0, keepdims=True)
    xc = x - mean
    _, singular_values, vt = np.linalg.svd(xc, full_matrices=False)
    requested_dim = max(1, int(out_dim))
    rank_dim = min(requested_dim, vt.shape[0])
    basis = np.zeros((x.shape[1], requested_dim), dtype=np.float32)
    if rank_dim > 0:
        basis[:, :rank_dim] = vt[:rank_dim].T.astype(np.float32)
    denom = max(1, x.shape[0] - 1)
    explained_variance = (singular_values ** 2) / float(denom)
    total_variance = float(np.sum(explained_variance))
    explained_variance_ratio = np.zeros((requested_dim,), dtype=np.float32)
    if total_variance > 1e-12:
        explained_variance_ratio[:rank_dim] = (explained_variance[:rank_dim] / total_variance).astype(np.float32)
    return {
        "valid_mask": valid_mask.astype(bool),
        "mean": mean.reshape(-1).astype(np.float32),
        "basis": basis,
        "requested_dim": int(requested_dim),
        "rank_dim": int(rank_dim),
        "explained_variance_ratio": explained_variance_ratio,
    }


def apply_pca_projection(x: np.ndarray, pca_params: dict[str, Any]) -> np.ndarray:
    x_full = np.asarray(x, dtype=np.float32)
    valid_mask = np.asarray(pca_params["valid_mask"], dtype=bool)
    x_valid = x_full[:, valid_mask]
    mean = np.asarray(pca_params["mean"], dtype=np.float32).reshape(1, -1)
    basis = np.asarray(pca_params["basis"], dtype=np.float32)
    return ((x_valid - mean) @ basis).astype(np.float32)


def pca_top_loadings(feature_names: list[str], pca_params: dict[str, Any], component_names: list[str], top_n: int = 8) -> list[dict[str, Any]]:
    valid_mask = np.asarray(pca_params["valid_mask"], dtype=bool)
    valid_names = [name for name, keep in zip(feature_names, valid_mask) if keep]
    basis = np.asarray(pca_params["basis"], dtype=np.float32)
    out: list[dict[str, Any]] = []
    for j, comp_name in enumerate(component_names):
        weights = basis[:, j]
        order = np.argsort(-np.abs(weights))[:top_n]
        out.append(
            {
                "component": comp_name,
                "top_features": [
                    {
                        "feature": valid_names[int(i)],
                        "loading": float(weights[int(i)]),
                        "abs_loading": float(abs(weights[int(i)])),
                    }
                    for i in order
                ],
            }
        )
    return out


def compute_teacher_state_old_ac(base_feat_z: np.ndarray) -> np.ndarray:
    hr = base_feat_z[:, 0]
    tonic = base_feat_z[:, 1]
    phasic = base_feat_z[:, 2]
    emg = base_feat_z[:, 3]
    alpha_asym = base_feat_z[:, 4]
    occ_ta = base_feat_z[:, 5]
    fr_ta = base_feat_z[:, 6]
    te_ta = base_feat_z[:, 7]
    occ_aabs = base_feat_z[:, 8]
    te_g = base_feat_z[:, 9]
    oc_g = base_feat_z[:, 10]
    fr_g = base_feat_z[:, 11]
    gamma_mean = (te_g + oc_g + fr_g) / 3.0
    ta_mean = (occ_ta + fr_ta + te_ta) / 3.0
    arousal = 0.70 * hr + 0.40 * tonic + 0.80 * phasic + 0.30 * gamma_mean - 0.30 * occ_aabs + 0.10 * alpha_asym
    control = 0.70 * emg + 0.50 * ta_mean
    return np.stack([arousal, control], axis=1).astype(np.float32)


def compute_semantic_driver_state(base_feat_z: np.ndarray, finite_mask: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    hr = base_feat_z[:, 0]
    tonic = base_feat_z[:, 1]
    phasic = base_feat_z[:, 2]
    emg = base_feat_z[:, 3]
    occ_ta = base_feat_z[:, 5]
    fr_ta = base_feat_z[:, 6]
    te_ta = base_feat_z[:, 7]
    occ_alpha_abs = base_feat_z[:, 8]
    te_gamma = base_feat_z[:, 9]
    occ_gamma = base_feat_z[:, 10]
    fr_gamma = base_feat_z[:, 11]
    ta_mean = (occ_ta + fr_ta + te_ta) / 3.0
    gamma_mean = (te_gamma + occ_gamma + fr_gamma) / 3.0

    driver_arousal = 0.30 * hr + 0.25 * tonic + 0.30 * phasic + 0.10 * emg + 0.05 * gamma_mean
    driver_workload = 0.30 * fr_ta + 0.25 * te_ta + 0.20 * occ_ta + 0.15 * phasic + 0.10 * hr
    driver_fatigue_risk = 0.30 * occ_alpha_abs + 0.25 * occ_ta + 0.20 * te_ta + 0.15 * fr_ta - 0.10 * hr
    driver_control_tension = 0.70 * emg + 0.30 * ta_mean

    if finite_mask is None:
        physio_valid_ratio = np.ones_like(hr, dtype=np.float32)
        eeg_valid_ratio = np.ones_like(hr, dtype=np.float32)
    else:
        mask = np.asarray(finite_mask, dtype=np.float32)
        physio_valid_ratio = mask[:, :4].mean(axis=1).astype(np.float32)
        eeg_valid_ratio = mask[:, 4:].mean(axis=1).astype(np.float32)

    state = np.stack(
        [
            driver_arousal,
            driver_workload,
            driver_fatigue_risk,
            driver_control_tension,
            physio_valid_ratio,
            eeg_valid_ratio,
        ],
        axis=1,
    ).astype(np.float32)
    meta = {
        "state_formulas": {
            "driver_arousal": "0.30*hr + 0.25*eda_tonic + 0.30*eda_phasic + 0.10*emg_rms + 0.05*gamma_mean",
            "driver_workload": "0.30*frontal_ta_beta + 0.25*temporal_ta_beta + 0.20*occ_ta_beta + 0.15*eda_phasic + 0.10*hr",
            "driver_fatigue_risk": "0.30*occ_alpha_abs + 0.25*occ_ta_beta + 0.20*temporal_ta_beta + 0.15*frontal_ta_beta - 0.10*hr",
            "driver_control_tension": "0.70*emg_rms + 0.30*mean(occ_ta_beta, frontal_ta_beta, temporal_ta_beta)",
            "physio_valid_ratio": "mean(valid(hr, eda_tonic, eda_phasic, emg_rms))",
            "eeg_valid_ratio": "mean(valid(eeg_feature_1..8))",
        },
        "state_interpretation": {
            "driver_arousal": "literature-guided arousal / stress activation proxy",
            "driver_workload": "literature-guided cognitive workload proxy",
            "driver_fatigue_risk": "literature-guided low-vigilance / fatigue-risk proxy",
            "driver_control_tension": "muscle / control tension proxy",
            "physio_valid_ratio": "physiology modality reliability",
            "eeg_valid_ratio": "EEG modality reliability",
        },
    }
    return state, meta


def build_teacher_state(
    base_feat_z: np.ndarray,
    mode: str,
    state_dim: int,
    fit_indices: list[int],
    finite_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if mode == "raw_hr_only":
        cols = [0]
        z_raw = np.asarray(base_feat_z[:, cols], dtype=np.float32)
        component_names = [f"physio_raw_{TEACHER_BASE_NAMES[i]}" for i in cols]
        return z_raw, {"mode": mode, "raw_dim": int(z_raw.shape[1]), "component_names": component_names, "eeg_removed_for_control": True, "physio_signal_group": "hr"}
    if mode == "raw_eda_only":
        cols = [1, 2]
        z_raw = np.asarray(base_feat_z[:, cols], dtype=np.float32)
        component_names = [f"physio_raw_{TEACHER_BASE_NAMES[i]}" for i in cols]
        return z_raw, {"mode": mode, "raw_dim": int(z_raw.shape[1]), "component_names": component_names, "eeg_removed_for_control": True, "physio_signal_group": "eda"}
    if mode == "raw_emg_only":
        cols = [3]
        z_raw = np.asarray(base_feat_z[:, cols], dtype=np.float32)
        component_names = [f"physio_raw_{TEACHER_BASE_NAMES[i]}" for i in cols]
        return z_raw, {"mode": mode, "raw_dim": int(z_raw.shape[1]), "component_names": component_names, "eeg_removed_for_control": True, "physio_signal_group": "emg"}
    if mode == "raw_physio_no_eeg":
        z_raw = np.asarray(base_feat_z[:, :PHYSIO_ONLY_DIM], dtype=np.float32)
        component_names = [f"physio_raw_{name}" for name in TEACHER_BASE_NAMES[:PHYSIO_ONLY_DIM]]
        return z_raw, {"mode": mode, "raw_dim": int(z_raw.shape[1]), "component_names": component_names, "eeg_removed_for_control": True}
    if mode == "raw_eeg_only":
        z_raw = np.asarray(base_feat_z[:, PHYSIO_ONLY_DIM:], dtype=np.float32)
        component_names = [f"eeg_raw_{name}" for name in TEACHER_BASE_NAMES[PHYSIO_ONLY_DIM:]]
        return z_raw, {"mode": mode, "raw_dim": int(z_raw.shape[1]), "component_names": component_names, "physio_removed_for_control": True}
    if mode == "raw_physio":
        z_raw = np.asarray(base_feat_z, dtype=np.float32)
        component_names = [f"physio_raw_{name}" for name in TEACHER_BASE_NAMES]
        return z_raw, {"mode": mode, "raw_dim": int(z_raw.shape[1]), "component_names": component_names}
    if mode == "old_ac":
        z_raw = compute_teacher_state_old_ac(base_feat_z)
        return z_raw.astype(np.float32), {"mode": mode, "raw_dim": int(z_raw.shape[1]), "component_names": ["A", "C"]}
    if mode == "semantic_driver_state":
        z_raw, semantic_meta = compute_semantic_driver_state(base_feat_z, finite_mask=finite_mask)
        component_names = [
            "driver_arousal",
            "driver_workload",
            "driver_fatigue_risk",
            "driver_control_tension",
            "physio_valid_ratio",
            "eeg_valid_ratio",
        ]
        semantic_meta.update({"mode": mode, "raw_dim": int(z_raw.shape[1]), "component_names": component_names})
        return z_raw.astype(np.float32), semantic_meta
    if mode == "pca_latent":
        fit_idx = np.asarray(fit_indices, dtype=np.int64)
        if fit_idx.size == 0:
            raise ValueError("fit_indices for teacher state PCA cannot be empty")
        fit_dim = max(1, min(int(state_dim), base_feat_z.shape[1], len(fit_idx)))
        pca_params = fit_pca_projection(np.asarray(base_feat_z, dtype=np.float32)[fit_idx], fit_dim)
        z_raw = apply_pca_projection(base_feat_z, pca_params)
        component_names = [f"teacher_state_{i + 1}" for i in range(int(z_raw.shape[1]))]
        return z_raw.astype(np.float32), {
            "mode": mode,
            "raw_dim": int(z_raw.shape[1]),
            "component_names": component_names,
            "pca_valid_mask": np.asarray(pca_params["valid_mask"], dtype=int).tolist(),
            "pca_mean": np.asarray(pca_params["mean"], dtype=np.float32).tolist(),
            "pca_basis": np.asarray(pca_params["basis"], dtype=np.float32).tolist(),
            "pca_explained_variance_ratio": np.asarray(pca_params["explained_variance_ratio"], dtype=np.float32).tolist(),
            "pca_top_loadings": pca_top_loadings(TEACHER_BASE_NAMES, pca_params, component_names),
        }
    raise ValueError(f"Unsupported teacher_state_mode={mode!r}")


def subset_manifest(
    manifest_df: pd.DataFrame,
    max_train_samples: int | None,
    max_val_samples: int | None,
    max_test_samples: int | None,
    seed: int,
) -> pd.DataFrame:
    if not {"train", "val", "test"}.issubset(set(manifest_df["split"].astype(str).unique())):
        return manifest_df.reset_index(drop=True)
    out = pd.concat(
        [
            _sample_by_split(manifest_df, "train", max_train_samples, seed + 11),
            _sample_by_split(manifest_df, "val", max_val_samples, seed + 13),
            _sample_by_split(manifest_df, "test", max_test_samples, seed + 17),
        ],
        axis=0,
        ignore_index=True,
    )
    return out.reset_index(drop=True)


def build_sample_bundle_from_manifest(
    manifest_path: str | Path,
    max_train_samples: int | None,
    max_val_samples: int | None,
    max_test_samples: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, int]:
    meta_df = pd.read_csv(manifest_path)
    meta_df = subset_manifest(meta_df, max_train_samples, max_val_samples, max_test_samples, seed=seed)

    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    curve_list: list[np.ndarray] = []
    ctx_list: list[np.ndarray] = []
    mask_list: list[np.ndarray] = []
    keep_rows: list[int] = []
    dropped = 0

    for i, row in meta_df.iterrows():
        try:
            row_work = row.copy()
            subject = _subject_from_row(row_work)
            if "vehicle_file" in row_work:
                row_work["vehicle_file"] = str(resolve_data_file_path(row_work["vehicle_file"], subject=subject, kind="vehicle"))
            if "event_file" in row_work:
                row_work["event_file"] = str(resolve_data_file_path(row_work["event_file"], subject=subject, kind="event"))
            x_win, y_seq, curve_future, ctx, future_mask = _make_sample(row_work)
        except Exception:
            dropped += 1
            continue
        x_list.append(x_win)
        y_list.append(y_seq)
        curve_list.append(curve_future)
        ctx_list.append(ctx)
        mask_list.append(future_mask)
        keep_rows.append(i)

    if not x_list:
        raise RuntimeError("No valid samples were built from manifest; check manifest path and data files.")

    kept_meta = meta_df.iloc[keep_rows].reset_index(drop=True).copy()
    return (
        np.stack(x_list).astype(np.float32),
        np.stack(y_list).astype(np.float32),
        np.stack(curve_list).astype(np.float32),
        np.stack(ctx_list).astype(np.float32),
        np.stack(mask_list).astype(np.float32),
        kept_meta,
        dropped,
    )


def standardize_feature_pool(
    pool: np.ndarray,
    train_idx: list[int],
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    train = pool[np.asarray(train_idx, dtype=np.int64)]
    finite_count = np.isfinite(train).sum(axis=0)
    missing_count = (~np.isfinite(train)).sum(axis=0)
    all_missing_mask = finite_count == 0
    mu = np.zeros((pool.shape[1],), dtype=np.float32)
    sd = np.ones((pool.shape[1],), dtype=np.float32)
    valid_stat_mask = ~all_missing_mask
    if np.any(valid_stat_mask):
        mu[valid_stat_mask] = np.nanmean(train[:, valid_stat_mask], axis=0).astype(np.float32)
        sd[valid_stat_mask] = np.nanstd(train[:, valid_stat_mask], axis=0).astype(np.float32)
    sd[sd < EPS] = EPS
    filled = pool.copy()
    bad = ~np.isfinite(filled)
    if np.any(bad):
        rows, cols = np.where(bad)
        filled[rows, cols] = mu[cols]
    z = ((filled - mu.reshape(1, -1)) / sd.reshape(1, -1)).astype(np.float32)
    stats: list[dict[str, Any]] = []
    for i, name in enumerate(feature_names):
        stats.append(
            {
                "index": int(i),
                "name": name,
                "finite_count": int(finite_count[i]),
                "missing_count": int(missing_count[i]),
                "valid_ratio": float(finite_count[i] / max(1, len(train_idx))),
                "all_missing": bool(all_missing_mask[i]),
                "mean": float(mu[i]),
                "std": float(sd[i]),
            }
        )
    return z, np.isfinite(pool), mu, sd, stats


def remove_eeg_from_teacher_pool(pool: np.ndarray) -> np.ndarray:
    out = np.asarray(pool, dtype=np.float32).copy()
    if out.shape[1] > PHYSIO_ONLY_DIM:
        out[:, PHYSIO_ONLY_DIM:] = np.nan
    return out


def remove_physio_from_teacher_pool(pool: np.ndarray) -> np.ndarray:
    out = np.asarray(pool, dtype=np.float32).copy()
    out[:, :PHYSIO_ONLY_DIM] = np.nan
    return out


def keep_only_teacher_signal(pool: np.ndarray, signal_group: str) -> np.ndarray:
    out = np.asarray(pool, dtype=np.float32).copy()
    keep_by_group = {
        "hr": [0],
        "eda": [1, 2],
        "emg": [3],
        "eeg": list(range(PHYSIO_ONLY_DIM, out.shape[1])),
    }
    if signal_group not in keep_by_group:
        raise ValueError(f"Unsupported teacher signal group={signal_group!r}")
    keep = set(int(i) for i in keep_by_group[signal_group])
    for idx in range(out.shape[1]):
        if idx not in keep:
            out[:, idx] = np.nan
    return out


def build_teacher_base_pool(meta_df: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    physio_cache: dict[str, pd.DataFrame | None] = {}
    eeg_cache: dict[str, dict[int, np.ndarray]] = {}
    v05_eeg_cache = build_v05_eeg_feature_cache()
    rows: list[np.ndarray] = []
    physio_available = 0
    eeg_available = 0
    eeg_v05_available = 0
    eeg_legacy_available = 0
    for _, row in meta_df.iterrows():
        subject = _subject_from_row(row)
        vehicle_file = str(resolve_data_file_path(str(row["vehicle_file"]), subject=subject, kind="vehicle"))
        context_vehicle_text = str(row.get("raw_vehicle_file_before_cleaning", "") or "").strip()
        context_vehicle_file = (
            str(resolve_data_file_path(context_vehicle_text, subject=subject, kind="vehicle"))
            if context_vehicle_text and context_vehicle_text.lower() != "nan"
            else vehicle_file
        )
        anchor_idx = int(row["anchor_idx"])
        event_idx = int(row.get("event_idx", -1))
        if context_vehicle_file not in physio_cache:
            physio_file = infer_physio_file(context_vehicle_file)
            physio_cache[context_vehicle_file] = pd.read_csv(physio_file) if physio_file is not None and os.path.exists(physio_file) else None
        if context_vehicle_file not in eeg_cache:
            eeg_cache[context_vehicle_file] = build_eeg_feat_map(infer_eeg_event_feature_file(context_vehicle_file))
        phys4 = extract_physio_window_means(physio_cache[context_vehicle_file], anchor_idx)
        v05_eeg8, _ = get_v05_eeg_features_for_row(row)
        legacy_eeg8 = eeg_cache[context_vehicle_file].get(event_idx)
        eeg8 = v05_eeg8 if v05_eeg8 is not None else legacy_eeg8
        if phys4 is None:
            phys4 = np.full((4,), np.nan, dtype=np.float32)
        else:
            physio_available += 1
        if eeg8 is None:
            eeg8 = np.full((8,), np.nan, dtype=np.float32)
        else:
            eeg_available += 1
            if v05_eeg8 is not None:
                eeg_v05_available += 1
            elif legacy_eeg8 is not None:
                eeg_legacy_available += 1
        rows.append(np.concatenate([phys4, eeg8], axis=0).astype(np.float32))
    base_pool = np.stack(rows, axis=0).astype(np.float32)
    meta = {
        "base_feature_names": TEACHER_BASE_NAMES,
        "sample_count": int(len(meta_df)),
        "physio_available_count": int(physio_available),
        "eeg_available_count": int(eeg_available),
        "eeg_v05_available_count": int(eeg_v05_available),
        "eeg_legacy_available_count": int(eeg_legacy_available),
        "eeg_v05_feature_table": str(v05_eeg_cache.get("source_file", "")),
        "eeg_v05_feature_table_available": bool(v05_eeg_cache.get("available", False)),
        "eeg_v05_feature_table_ok_rows": int(v05_eeg_cache.get("ok_count", 0)),
        "physio_file_count": int(sum(1 for value in physio_cache.values() if value is not None)),
        "eeg_file_count": int(sum(1 for value in eeg_cache.values() if value)),
    }
    return base_pool, meta


def build_teacher_base_and_local_delta_pool(meta_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    physio_cache: dict[str, pd.DataFrame | None] = {}
    eeg_cache: dict[str, dict[int, np.ndarray]] = {}
    v05_eeg_cache = build_v05_eeg_feature_cache()
    current_rows: list[np.ndarray] = []
    delta_rows: list[np.ndarray] = []
    physio_available = 0
    eeg_available = 0
    physio_delta_available = 0
    eeg_delta_available = 0
    eeg_v05_available = 0
    eeg_legacy_available = 0
    eeg_v05_delta_available = 0
    eeg_legacy_delta_available = 0
    for _, row in meta_df.iterrows():
        subject = _subject_from_row(row)
        vehicle_file = str(resolve_data_file_path(str(row["vehicle_file"]), subject=subject, kind="vehicle"))
        context_vehicle_text = str(row.get("raw_vehicle_file_before_cleaning", "") or "").strip()
        context_vehicle_file = (
            str(resolve_data_file_path(context_vehicle_text, subject=subject, kind="vehicle"))
            if context_vehicle_text and context_vehicle_text.lower() != "nan"
            else vehicle_file
        )
        anchor_idx = int(row["anchor_idx"])
        event_idx = int(row.get("event_idx", -1))
        if context_vehicle_file not in physio_cache:
            physio_file = infer_physio_file(context_vehicle_file)
            physio_cache[context_vehicle_file] = pd.read_csv(physio_file) if physio_file is not None and os.path.exists(physio_file) else None
        if context_vehicle_file not in eeg_cache:
            eeg_cache[context_vehicle_file] = build_eeg_feat_map(infer_eeg_event_feature_file(context_vehicle_file))

        phys4 = extract_physio_window_means(physio_cache[context_vehicle_file], anchor_idx)
        phys_delta4 = extract_physio_local_delta(physio_cache[context_vehicle_file], anchor_idx)
        v05_eeg8, v05_eeg_delta8 = get_v05_eeg_features_for_row(row)
        legacy_eeg8 = eeg_cache[context_vehicle_file].get(event_idx)
        legacy_eeg_delta8 = compute_eeg_prior_event_delta(eeg_cache[context_vehicle_file], event_idx)
        eeg8 = v05_eeg8 if v05_eeg8 is not None else legacy_eeg8
        eeg_delta8 = v05_eeg_delta8 if v05_eeg_delta8 is not None else legacy_eeg_delta8

        if phys4 is None:
            phys4 = np.full((4,), np.nan, dtype=np.float32)
        else:
            physio_available += 1
        if phys_delta4 is None:
            phys_delta4 = np.full((4,), np.nan, dtype=np.float32)
        else:
            physio_delta_available += 1
        if eeg8 is None:
            eeg8 = np.full((8,), np.nan, dtype=np.float32)
        else:
            eeg_available += 1
            if v05_eeg8 is not None:
                eeg_v05_available += 1
            elif legacy_eeg8 is not None:
                eeg_legacy_available += 1
        if eeg_delta8 is None:
            eeg_delta8 = np.full((8,), np.nan, dtype=np.float32)
        else:
            eeg_delta_available += 1
            if v05_eeg_delta8 is not None:
                eeg_v05_delta_available += 1
            elif legacy_eeg_delta8 is not None:
                eeg_legacy_delta_available += 1
        current_rows.append(np.concatenate([phys4, eeg8], axis=0).astype(np.float32))
        delta_rows.append(np.concatenate([phys_delta4, eeg_delta8], axis=0).astype(np.float32))

    current_pool = np.stack(current_rows, axis=0).astype(np.float32)
    delta_pool = np.stack(delta_rows, axis=0).astype(np.float32)
    meta = {
        "base_feature_names": TEACHER_BASE_NAMES,
        "sample_count": int(len(meta_df)),
        "physio_available_count": int(physio_available),
        "eeg_available_count": int(eeg_available),
        "physio_delta_available_count": int(physio_delta_available),
        "eeg_delta_available_count": int(eeg_delta_available),
        "eeg_v05_available_count": int(eeg_v05_available),
        "eeg_legacy_available_count": int(eeg_legacy_available),
        "eeg_v05_delta_available_count": int(eeg_v05_delta_available),
        "eeg_legacy_delta_available_count": int(eeg_legacy_delta_available),
        "eeg_v05_feature_table": str(v05_eeg_cache.get("source_file", "")),
        "eeg_v05_feature_table_available": bool(v05_eeg_cache.get("available", False)),
        "eeg_v05_feature_table_ok_rows": int(v05_eeg_cache.get("ok_count", 0)),
        "physio_file_count": int(sum(1 for value in physio_cache.values() if value is not None)),
        "eeg_file_count": int(sum(1 for value in eeg_cache.values() if value)),
        "physio_current_window_samples": int(PHYSIO_CURRENT_SAMPLES),
        "physio_baseline_gap_samples": int(PHYSIO_BASELINE_GAP_SAMPLES),
        "physio_baseline_max_samples": int(PHYSIO_BASELINE_MAX_SAMPLES),
        "physio_baseline_min_samples": int(PHYSIO_BASELINE_MIN_SAMPLES),
        "eeg_delta_baseline": "prior events in the same recording only",
    }
    return current_pool, delta_pool, meta


def build_semantic_driver_state_local_delta_context(
    meta_df: pd.DataFrame,
    train_idx: list[int],
    remove_eeg: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    current_pool, delta_pool, source_meta = build_teacher_base_and_local_delta_pool(meta_df)
    if remove_eeg:
        current_pool = remove_eeg_from_teacher_pool(current_pool)
        delta_pool = remove_eeg_from_teacher_pool(delta_pool)
    current_z, current_finite_mask, current_mu, current_sd, current_stats = standardize_feature_pool(current_pool, train_idx, TEACHER_BASE_NAMES)
    delta_z, delta_finite_mask, delta_mu, delta_sd, delta_stats = standardize_feature_pool(delta_pool, train_idx, TEACHER_BASE_NAMES)
    current_state, current_meta = compute_semantic_driver_state(current_z, finite_mask=current_finite_mask)
    delta_state, delta_meta = compute_semantic_driver_state(delta_z, finite_mask=delta_finite_mask)
    z_raw = np.concatenate([current_state[:, :4], delta_state[:, :4], current_state[:, 4:6], delta_state[:, 4:6]], axis=1).astype(np.float32)
    component_names = [
        "driver_arousal_current",
        "driver_workload_current",
        "driver_fatigue_risk_current",
        "driver_control_tension_current",
        "driver_arousal_delta",
        "driver_workload_delta",
        "driver_fatigue_risk_delta",
        "driver_control_tension_delta",
        "physio_valid_ratio",
        "eeg_valid_ratio",
        "physio_delta_valid_ratio",
        "eeg_delta_valid_ratio",
    ]
    z_train = z_raw[np.asarray(train_idx, dtype=np.int64)]
    z_mu = z_train.mean(axis=0).astype(np.float32)
    z_sd = z_train.std(axis=0).astype(np.float32)
    z_sd[z_sd < EPS] = EPS
    z_ctx = ((z_raw - z_mu.reshape(1, -1)) / z_sd.reshape(1, -1)).astype(np.float32)
    mode_name = "semantic_driver_state_local_delta_no_eeg" if remove_eeg else "semantic_driver_state_local_delta"
    teacher_meta = {
        "kind": "teacher_state_context",
        "mode": mode_name,
        "fit_split": "train",
        "fit_sample_count": int(len(train_idx)),
        "raw_dim": int(z_raw.shape[1]),
        "state_dim": int(z_ctx.shape[1]),
        "component_names": component_names,
        "semantic_state_source": (
            "current pre-anchor non-EEG physiology state plus local pre-anchor physiology deltas"
            if remove_eeg
            else "current pre-anchor physiology/EEG state plus local pre-anchor / prior-event deltas"
        ),
        "eeg_removed_for_control": bool(remove_eeg),
        "current_state_formulas": current_meta["state_formulas"],
        "delta_state_formulas": delta_meta["state_formulas"],
        "current_state_interpretation": current_meta["state_interpretation"],
        "delta_state_interpretation": {key: value.replace("proxy", "relative-change proxy") for key, value in delta_meta["state_interpretation"].items()},
        "z_mu": z_mu.tolist(),
        "z_sd": z_sd.tolist(),
        "current_base_mu": current_mu.tolist(),
        "current_base_sd": current_sd.tolist(),
        "delta_base_mu": delta_mu.tolist(),
        "delta_base_sd": delta_sd.tolist(),
        "current_base_missing_stats": current_stats,
        "delta_base_missing_stats": delta_stats,
        **source_meta,
    }
    return z_ctx, teacher_meta


SIGNAL_GROUP_TO_INDICES: dict[str, list[int]] = {
    "hr": [0],
    "eda": [1, 2],
    "emg": [3],
    "eeg": list(range(PHYSIO_ONLY_DIM, len(TEACHER_BASE_NAMES))),
    "all": list(range(len(TEACHER_BASE_NAMES))),
}


def _signal_feature_names(signal_group: str) -> list[str]:
    if signal_group not in SIGNAL_GROUP_TO_INDICES:
        raise ValueError(f"Unsupported signal_group={signal_group!r}")
    return [TEACHER_BASE_NAMES[i] for i in SIGNAL_GROUP_TO_INDICES[signal_group]]


def build_signal_current_delta_context(
    meta_df: pd.DataFrame,
    train_idx: list[int],
    signal_group: str,
    transform: str,
    state_dim: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if signal_group not in SIGNAL_GROUP_TO_INDICES:
        raise ValueError(f"Unsupported signal_group={signal_group!r}")
    if transform not in {"current_delta", "pca"}:
        raise ValueError(f"Unsupported signal context transform={transform!r}")

    current_pool, delta_pool, source_meta = build_teacher_base_and_local_delta_pool(meta_df)
    current_z, current_mask, current_mu, current_sd, current_stats = standardize_feature_pool(
        current_pool, train_idx, TEACHER_BASE_NAMES
    )
    delta_z, delta_mask, delta_mu, delta_sd, delta_stats = standardize_feature_pool(
        delta_pool, train_idx, TEACHER_BASE_NAMES
    )

    cols = SIGNAL_GROUP_TO_INDICES[signal_group]
    signal_names = _signal_feature_names(signal_group)
    current_part = current_z[:, cols]
    delta_part = delta_z[:, cols]

    validity_parts: list[np.ndarray] = []
    validity_names: list[str] = []
    validity_groups = ["hr", "eda", "emg", "eeg"] if signal_group == "all" else [signal_group]
    for valid_group in validity_groups:
        valid_cols = SIGNAL_GROUP_TO_INDICES[valid_group]
        validity_parts.append(current_mask[:, valid_cols].mean(axis=1, keepdims=True).astype(np.float32))
        validity_parts.append(delta_mask[:, valid_cols].mean(axis=1, keepdims=True).astype(np.float32))
        validity_names.extend([f"{valid_group}_current_valid_ratio", f"{valid_group}_delta_valid_ratio"])

    raw_parts = [current_part, delta_part, *validity_parts]
    source_component_names = (
        [f"{name}_current" for name in signal_names]
        + [f"{name}_local_delta" for name in signal_names]
        + validity_names
    )
    raw_direct = np.concatenate(raw_parts, axis=1).astype(np.float32)

    pca_meta: dict[str, Any] = {"enabled": False}
    if transform == "pca":
        fit_idx = np.asarray(train_idx, dtype=np.int64)
        out_dim = max(1, min(int(state_dim), raw_direct.shape[1], len(fit_idx)))
        pca_params = fit_pca_projection(raw_direct[fit_idx], out_dim)
        z_raw = apply_pca_projection(raw_direct, pca_params).astype(np.float32)
        component_names = [f"{signal_group}_data_driven_state_{i + 1}" for i in range(int(z_raw.shape[1]))]
        pca_meta = {
            "enabled": True,
            "requested_dim": int(pca_params["requested_dim"]),
            "rank_dim": int(pca_params["rank_dim"]),
            "explained_variance_ratio": np.asarray(pca_params["explained_variance_ratio"], dtype=np.float32).tolist(),
            "top_loadings": pca_top_loadings(source_component_names, pca_params, component_names, top_n=6),
        }
    else:
        z_raw = raw_direct
        component_names = source_component_names

    z_train = z_raw[np.asarray(train_idx, dtype=np.int64)]
    z_mu = z_train.mean(axis=0).astype(np.float32)
    z_sd = z_train.std(axis=0).astype(np.float32)
    z_sd[z_sd < EPS] = EPS
    z_ctx = ((z_raw - z_mu.reshape(1, -1)) / z_sd.reshape(1, -1)).astype(np.float32)
    mode_name = f"signal_{transform}_{signal_group}_only"
    teacher_meta = {
        "kind": "teacher_state_context",
        "mode": mode_name,
        "fit_split": "train",
        "fit_sample_count": int(len(train_idx)),
        "signal_group": signal_group,
        "kept_feature_names": signal_names,
        "transform": transform,
        "raw_dim_before_final_standardize": int(z_raw.shape[1]),
        "state_dim": int(z_ctx.shape[1]),
        "component_names": component_names,
        "source_component_names": source_component_names,
        "interpretation": (
            "current pre-anchor signal plus local pre-anchor baseline-corrected change; no manual semantic weights"
            if transform == "current_delta"
            else "train-split PCA over current pre-anchor signal, local baseline-corrected change, and validity ratios; no manual semantic weights"
        ),
        "current_window_samples": int(PHYSIO_CURRENT_SAMPLES),
        "physio_baseline_gap_samples": int(PHYSIO_BASELINE_GAP_SAMPLES),
        "physio_baseline_max_samples": int(PHYSIO_BASELINE_MAX_SAMPLES),
        "physio_baseline_min_samples": int(PHYSIO_BASELINE_MIN_SAMPLES),
        "eeg_delta_baseline": "prior events in the same recording only",
        "z_mu": z_mu.tolist(),
        "z_sd": z_sd.tolist(),
        "current_base_mu": current_mu.tolist(),
        "current_base_sd": current_sd.tolist(),
        "delta_base_mu": delta_mu.tolist(),
        "delta_base_sd": delta_sd.tolist(),
        "current_base_missing_stats": current_stats,
        "delta_base_missing_stats": delta_stats,
        "data_driven_pca": pca_meta,
        **source_meta,
    }
    return z_ctx, teacher_meta


def build_teacher_state_context(meta_df: pd.DataFrame, train_idx: list[int], mode: str, state_dim: int) -> tuple[np.ndarray, dict[str, Any]]:
    if mode == "semantic_driver_state_local_delta":
        return build_semantic_driver_state_local_delta_context(meta_df, train_idx)
    if mode == "semantic_driver_state_local_delta_no_eeg":
        return build_semantic_driver_state_local_delta_context(meta_df, train_idx, remove_eeg=True)
    current_delta_signal_modes = {
        "signal_current_delta_hr_only": "hr",
        "signal_current_delta_eda_only": "eda",
        "signal_current_delta_emg_only": "emg",
        "signal_current_delta_eeg_only": "eeg",
        "signal_current_delta_all": "all",
    }
    pca_signal_modes = {
        "signal_pca_hr_only": "hr",
        "signal_pca_eda_only": "eda",
        "signal_pca_emg_only": "emg",
        "signal_pca_eeg_only": "eeg",
        "signal_pca_all": "all",
    }
    if mode in current_delta_signal_modes:
        return build_signal_current_delta_context(
            meta_df,
            train_idx,
            signal_group=current_delta_signal_modes[mode],
            transform="current_delta",
            state_dim=state_dim,
        )
    if mode in pca_signal_modes:
        return build_signal_current_delta_context(
            meta_df,
            train_idx,
            signal_group=pca_signal_modes[mode],
            transform="pca",
            state_dim=state_dim,
        )
    semantic_signal_only_modes = {
        "semantic_driver_state_hr_only": "hr",
        "semantic_driver_state_eda_only": "eda",
        "semantic_driver_state_emg_only": "emg",
    }
    no_eeg_control = mode in {"semantic_driver_state_no_eeg", "raw_physio_no_eeg"}
    eeg_only_control = mode in {"semantic_driver_state_eeg_only", "raw_eeg_only"}
    signal_only_control = mode in semantic_signal_only_modes
    base_mode = (
        "semantic_driver_state"
        if mode in {"semantic_driver_state_no_eeg", "semantic_driver_state_eeg_only"} or signal_only_control
        else mode
    )
    base_pool, source_meta = build_teacher_base_pool(meta_df)
    if no_eeg_control:
        base_pool = remove_eeg_from_teacher_pool(base_pool)
        source_meta = {**source_meta, "eeg_removed_for_control": True, "removed_feature_names": TEACHER_BASE_NAMES[PHYSIO_ONLY_DIM:]}
    if eeg_only_control:
        base_pool = remove_physio_from_teacher_pool(base_pool)
        source_meta = {**source_meta, "physio_removed_for_control": True, "removed_feature_names": TEACHER_BASE_NAMES[:PHYSIO_ONLY_DIM]}
    if signal_only_control:
        signal_group = semantic_signal_only_modes[mode]
        base_pool = keep_only_teacher_signal(base_pool, signal_group)
        keep_names = {
            "hr": ["hr"],
            "eda": ["eda_tonic", "eda_phasic"],
            "emg": ["emg_rms"],
        }[signal_group]
        source_meta = {
            **source_meta,
            "semantic_signal_only_group": signal_group,
            "kept_feature_names": keep_names,
            "removed_feature_names": [name for name in TEACHER_BASE_NAMES if name not in keep_names],
        }
    base_z, base_finite_mask, base_mu, base_sd, base_stats = standardize_feature_pool(base_pool, train_idx, TEACHER_BASE_NAMES)
    z_raw, teacher_meta = build_teacher_state(base_z, mode=base_mode, state_dim=state_dim, fit_indices=train_idx, finite_mask=base_finite_mask)
    if mode != base_mode:
        teacher_meta = {**teacher_meta, "mode": mode, "base_mode": base_mode}
        if no_eeg_control:
            teacher_meta["eeg_removed_for_control"] = True
        if eeg_only_control:
            teacher_meta["physio_removed_for_control"] = True
        if signal_only_control:
            teacher_meta["semantic_signal_only_group"] = semantic_signal_only_modes[mode]
    z_train = z_raw[np.asarray(train_idx, dtype=np.int64)]
    z_mu = z_train.mean(axis=0).astype(np.float32)
    z_sd = z_train.std(axis=0).astype(np.float32)
    z_sd[z_sd < EPS] = EPS
    z_ctx = ((z_raw - z_mu.reshape(1, -1)) / z_sd.reshape(1, -1)).astype(np.float32)
    teacher_meta.update(
        {
            "kind": "teacher_state_context",
            "fit_split": "train",
            "fit_sample_count": int(len(train_idx)),
            "state_dim": int(z_ctx.shape[1]),
            "z_mu": z_mu.tolist(),
            "z_sd": z_sd.tolist(),
            "base_mu": base_mu.tolist(),
            "base_sd": base_sd.tolist(),
            "base_missing_stats": base_stats,
            "base_all_missing_indices": [int(i) for i, item in enumerate(base_stats) if bool(item["all_missing"])],
            "base_all_missing_names": [item["name"] for item in base_stats if bool(item["all_missing"])],
            **source_meta,
        }
    )
    return z_ctx, teacher_meta


def load_driver_style_raw_map(style_vector_csv: str | Path, include_iqr: bool) -> tuple[dict[str, np.ndarray], list[str]]:
    path = Path(style_vector_csv)
    if not path.exists():
        raise FileNotFoundError(f"Driver style vector CSV does not exist: {path}")
    df = pd.read_csv(path)
    subj_col = None
    for candidate in ["driver_id", "subject", "Subject", "subject_id", "subj"]:
        if candidate in df.columns:
            subj_col = candidate
            break
    if subj_col is None:
        raise ValueError(f"No subject/driver id column found in style vector file: {path}")
    feature_cols: list[str] = []
    numeric_cols: dict[str, pd.Series] = {}
    for col in df.columns:
        if col == subj_col or col == "session_count_total" or str(col).endswith("__count"):
            continue
        if (not include_iqr) and str(col).endswith("__iqr"):
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().any():
            feature_cols.append(str(col))
            numeric_cols[str(col)] = values.astype(np.float32)
    if not feature_cols:
        raise ValueError(f"No numeric behavior feature columns found in style vector file: {path}")
    numeric_df = pd.DataFrame(numeric_cols)
    style_map: dict[str, np.ndarray] = {}
    for i, subject in enumerate(df[subj_col].astype(str).values):
        style_map[normalize_subject_id(subject)] = numeric_df.iloc[i][feature_cols].to_numpy(dtype=np.float32, copy=True)
    return style_map, feature_cols


def _style_fit_subject_indices(meta_df: pd.DataFrame, train_idx: list[int], missing: np.ndarray) -> np.ndarray:
    selected: list[int] = []
    seen: set[str] = set()
    for idx in train_idx:
        if int(missing[int(idx)]) != 0:
            continue
        sid = _subject_from_row(meta_df.iloc[int(idx)])
        if sid in seen:
            continue
        seen.add(sid)
        selected.append(int(idx))
    if selected:
        return np.asarray(selected, dtype=np.int64)
    for idx in train_idx:
        sid = _subject_from_row(meta_df.iloc[int(idx)])
        if sid in seen:
            continue
        seen.add(sid)
        selected.append(int(idx))
    return np.asarray(selected if selected else train_idx, dtype=np.int64)


def build_driver_style_context(
    meta_df: pd.DataFrame,
    train_idx: list[int],
    style_vector_csv: str | Path,
    embed_dim: int,
    include_iqr: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    style_map, raw_feature_names = load_driver_style_raw_map(style_vector_csv, include_iqr=include_iqr)
    raw_dim = len(raw_feature_names)
    raw_rows: list[np.ndarray] = []
    missing: list[int] = []
    for _, row in meta_df.iterrows():
        values = style_map.get(_subject_from_row(row))
        if values is None:
            raw_rows.append(np.full((raw_dim,), np.nan, dtype=np.float32))
            missing.append(1)
        else:
            raw_rows.append(np.asarray(values, dtype=np.float32).reshape(-1))
            missing.append(0)
    style_raw = np.stack(raw_rows, axis=0).astype(np.float32)
    missing_arr = np.asarray(missing, dtype=np.int64)
    fit_idx = _style_fit_subject_indices(meta_df, train_idx, missing_arr)
    fit_raw = style_raw[fit_idx]
    finite_count = np.isfinite(fit_raw).sum(axis=0)
    all_missing_mask = finite_count == 0
    raw_mu = np.zeros((raw_dim,), dtype=np.float32)
    raw_sd = np.ones((raw_dim,), dtype=np.float32)
    valid_stat_mask = ~all_missing_mask
    if np.any(valid_stat_mask):
        raw_mu[valid_stat_mask] = np.nanmean(fit_raw[:, valid_stat_mask], axis=0).astype(np.float32)
        raw_sd[valid_stat_mask] = np.nanstd(fit_raw[:, valid_stat_mask], axis=0).astype(np.float32)
    raw_sd[raw_sd < EPS] = 1.0
    style_filled = style_raw.copy()
    bad = ~np.isfinite(style_filled)
    if np.any(bad):
        rows, cols = np.where(bad)
        style_filled[rows, cols] = raw_mu[cols]
    style_z = ((style_filled - raw_mu.reshape(1, -1)) / raw_sd.reshape(1, -1)).astype(np.float32)
    requested_dim = max(1, min(int(embed_dim), raw_dim))
    pca_params = fit_pca_projection(style_z[fit_idx], requested_dim)
    style_ctx = apply_pca_projection(style_z, pca_params)
    component_names = [f"style_vector_{i + 1}" for i in range(int(style_ctx.shape[1]))]
    meta = {
        "kind": "driver_style_context",
        "source_file": str(style_vector_csv),
        "transform": "train_subject_pca",
        "fit_split": "train",
        "fit_sample_count": int(len(train_idx)),
        "fit_subject_count": int(len(fit_idx)),
        "style_dim": int(style_ctx.shape[1]),
        "requested_embed_dim": int(embed_dim),
        "include_iqr": bool(include_iqr),
        "raw_dim": int(raw_dim),
        "raw_feature_names": list(raw_feature_names),
        "component_names": component_names,
        "missing_sample_count": int(missing_arr.sum()),
        "missing_subject_count": int(len({_subject_from_row(meta_df.iloc[i]) for i, miss in enumerate(missing_arr) if int(miss) != 0})),
        "raw_mean": raw_mu.tolist(),
        "raw_std": raw_sd.tolist(),
        "pca_valid_mask": np.asarray(pca_params["valid_mask"], dtype=int).tolist(),
        "pca_mean": np.asarray(pca_params["mean"], dtype=np.float32).tolist(),
        "pca_basis": np.asarray(pca_params["basis"], dtype=np.float32).tolist(),
        "pca_explained_variance_ratio": np.asarray(pca_params["explained_variance_ratio"], dtype=np.float32).tolist(),
        "pca_top_loadings": pca_top_loadings(raw_feature_names, pca_params, component_names),
    }
    return style_ctx.astype(np.float32), meta


def build_optional_context_augmentation(
    ctx_pool: np.ndarray,
    meta_df: pd.DataFrame,
    train_idx: list[int],
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    context_feature_names = ["anchor_steer", "anchor_speed", "anchor_steer_rate", "anchor_ay", "anchor_yawrate"]
    pieces = [np.asarray(ctx_pool, dtype=np.float32)]
    augmentation_meta: dict[str, Any] = {
        "enabled": False,
        "base_context_dim": int(ctx_pool.shape[1]),
        "base_context_feature_names": list(context_feature_names),
        "augmentations": [],
    }
    if bool(getattr(args, "enable_teacher_state_context", False)):
        teacher_ctx, teacher_meta = build_teacher_state_context(
            meta_df,
            train_idx,
            mode=str(getattr(args, "teacher_state_mode", "pca_latent")),
            state_dim=int(getattr(args, "teacher_state_dim", 4)),
        )
        pieces.append(teacher_ctx)
        context_feature_names.extend(list(teacher_meta["component_names"]))
        augmentation_meta["enabled"] = True
        augmentation_meta["augmentations"].append(teacher_meta)
    if bool(getattr(args, "enable_driver_style_context", False)):
        style_ctx, style_meta = build_driver_style_context(
            meta_df,
            train_idx,
            style_vector_csv=str(getattr(args, "driver_style_vector_csv", DEFAULT_DRIVER_STYLE_VECTOR_CSV)),
            embed_dim=int(getattr(args, "driver_style_embed_dim", 4)),
            include_iqr=bool(getattr(args, "driver_style_include_iqr", True)),
        )
        pieces.append(style_ctx)
        context_feature_names.extend(list(style_meta["component_names"]))
        augmentation_meta["enabled"] = True
        augmentation_meta["augmentations"].append(style_meta)
    ctx_aug = np.concatenate(pieces, axis=1).astype(np.float32)
    augmentation_meta["final_context_dim"] = int(ctx_aug.shape[1])
    augmentation_meta["context_feature_names"] = context_feature_names
    return ctx_aug, augmentation_meta


def apply_optional_context_augmentation(
    ctx_pool: np.ndarray,
    meta_df: pd.DataFrame,
    train_idx: list[int],
    args: argparse.Namespace,
    run_root: Path | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    ctx_aug, augmentation_meta = build_optional_context_augmentation(
        ctx_pool=ctx_pool,
        meta_df=meta_df,
        train_idx=train_idx,
        args=args,
    )
    if run_root is not None:
        save_json(run_root / "context_augmentation_meta.json", augmentation_meta)
        save_json(
            run_root / "context_feature_names.json",
            {
                "n_context_features": int(len(augmentation_meta["context_feature_names"])),
                "context_feature_names": augmentation_meta["context_feature_names"],
            },
        )
    return ctx_aug, augmentation_meta


def make_loader(dataset: EventConditionedDataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )


def safe_torch_save(payload: dict[str, Any], path: str | Path, attempts: int = 5) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_target = target.with_name(f"{target.name}.tmp")
    last_exc: Exception | None = None
    for attempt in range(1, int(attempts) + 1):
        try:
            if tmp_target.exists():
                tmp_target.unlink()
            with tmp_target.open("wb") as handle:
                torch.save(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            tmp_target.replace(target)
            return
        except (OSError, RuntimeError) as exc:
            last_exc = exc
            if attempt < int(attempts):
                time.sleep(0.5 * attempt)
    if last_exc is not None:
        raise last_exc


def weighted_masked_mse(
    pred: torch.Tensor,
    true: torch.Tensor,
    mask: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    sq = (pred - true) ** 2
    weight_mask = mask
    if sample_weight is not None:
        weight_mask = weight_mask * sample_weight.view(-1, 1, 1).to(dtype=mask.dtype, device=mask.device)
    denom = torch.clamp(weight_mask.sum(), min=1.0)
    return (sq * weight_mask).sum() / denom


def compute_multi_candidate_losses(
    extras: dict[str, Any],
    y_true: torch.Tensor,
    traj_mask: torch.Tensor,
    batch: dict[str, torch.Tensor] | None = None,
    target_mode: str = "oracle",
) -> dict[str, torch.Tensor]:
    candidates = extras.get("candidate_trajectories")
    logits = extras.get("candidate_logits")
    zero = y_true.new_zeros(())
    if candidates is None or logits is None:
        return {
            "min_loss": zero,
            "selector_loss": zero,
            "oracle_loss": zero,
        }
    mask = traj_mask.unsqueeze(1)
    sq = (candidates - y_true.unsqueeze(1)) ** 2
    denom = torch.clamp(mask.sum(dim=(2, 3)), min=1.0)
    per_candidate = (sq * mask).sum(dim=(2, 3)) / denom
    oracle_loss, best_idx = torch.min(per_candidate, dim=1)
    response_label_loss = zero
    response_target_idx = None
    if batch is not None and RESPONSE_CANDIDATE_CLASS_KEY in batch:
        response_target_idx = batch[RESPONSE_CANDIDATE_CLASS_KEY].long().to(device=logits.device)
        response_target_idx = torch.clamp(response_target_idx, min=0, max=int(logits.shape[1]) - 1)
        response_label_loss = per_candidate.gather(1, response_target_idx.view(-1, 1)).mean()
    mode = str(target_mode)
    if mode == "response_type" and response_target_idx is not None:
        train_candidate_loss = response_label_loss
        selector_target = response_target_idx.detach()
    elif mode == "hybrid" and response_target_idx is not None:
        train_candidate_loss = 0.5 * oracle_loss.mean() + 0.5 * response_label_loss
        selector_target = response_target_idx.detach()
    else:
        train_candidate_loss = oracle_loss.mean()
        selector_target = best_idx.detach()
    selector_loss = torch.nn.functional.cross_entropy(logits, selector_target)
    return {
        "min_loss": train_candidate_loss,
        "selector_loss": selector_loss,
        "oracle_loss": oracle_loss.mean(),
        "response_label_loss": response_label_loss,
    }


def build_response_candidate_prototypes(
    y_pool: np.ndarray,
    train_idx: list[int],
    response_targets: dict[str, np.ndarray] | None,
    norm_stats: dict[str, Any],
    *,
    num_candidates: int,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    if response_targets is None or RESPONSE_CANDIDATE_CLASS_KEY not in response_targets or int(num_candidates) <= 1:
        return None, {"enabled": False}
    y_mean = np.asarray(norm_stats["y_mean"], dtype=np.float32).reshape(1, 1, -1)
    y_std = np.asarray(norm_stats["y_std"], dtype=np.float32).reshape(1, 1, -1)
    y_norm = ((np.asarray(y_pool, dtype=np.float32) - y_mean) / y_std).astype(np.float32)
    labels = np.asarray(response_targets[RESPONSE_CANDIDATE_CLASS_KEY], dtype=np.int64)
    train_labels = labels[np.asarray(train_idx, dtype=np.int64)]
    train_y = y_norm[np.asarray(train_idx, dtype=np.int64)]
    global_proto = np.mean(train_y, axis=0).astype(np.float32)
    prototypes = np.zeros((int(num_candidates), y_norm.shape[1], y_norm.shape[2]), dtype=np.float32)
    counts: dict[str, int] = {}
    for cls in range(int(num_candidates)):
        keep = train_labels == cls
        counts[str(cls)] = int(keep.sum())
        prototypes[cls] = np.mean(train_y[keep], axis=0).astype(np.float32) if bool(keep.any()) else global_proto
    meta = {
        "enabled": True,
        "num_candidates": int(num_candidates),
        "class_meaning": {
            "0": "小幅或普通响应",
            "1": "高幅或晚峰响应",
            "2": "反向修正响应",
            "3": "多段修正或尾段跨侧响应",
        },
        "train_class_counts": counts,
        "fallback_used": [int(k) for k, v in counts.items() if int(v) <= 0],
    }
    return prototypes, meta


def compute_steer_physical_losses(
    y_hat: torch.Tensor,
    y_true_norm: torch.Tensor,
    traj_mask: torch.Tensor,
    y_mean_t: torch.Tensor,
    y_std_t: torch.Tensor,
    *,
    amp_major_threshold: float,
    amp_target_ratio: float,
    direction_threshold: float,
    direction_margin: float,
    peak_window_steps: int = -1,
    amp_peak_window_only: bool = False,
    direction_major_only: bool = False,
) -> dict[str, torch.Tensor]:
    y_hat_den = y_hat * y_std_t.view(1, 1, -1) + y_mean_t.view(1, 1, -1)
    y_true_den = y_true_norm * y_std_t.view(1, 1, -1) + y_mean_t.view(1, 1, -1)
    pred_steer = y_hat_den[:, :, 0]
    true_steer = y_true_den[:, :, 0]
    valid = traj_mask.squeeze(-1) > 0.5
    zero = torch.zeros((), dtype=torch.float32, device=y_hat.device)

    true_abs = torch.where(valid, true_steer.abs(), torch.zeros_like(true_steer))
    pred_abs = torch.where(valid, pred_steer.abs(), torch.zeros_like(pred_steer))
    true_amp = true_abs.max(dim=1).values
    true_peak_idx = torch.argmax(true_abs, dim=1)
    if int(peak_window_steps) >= 0:
        steps = torch.arange(true_steer.shape[1], device=y_hat.device).view(1, -1)
        peak_dist = (steps - true_peak_idx.view(-1, 1)).abs()
        peak_valid = valid & (peak_dist <= int(peak_window_steps))
    else:
        peak_valid = valid
    pred_amp_source = torch.where(peak_valid, pred_steer.abs(), torch.zeros_like(pred_steer)) if bool(amp_peak_window_only) else pred_abs
    pred_amp = pred_amp_source.max(dim=1).values
    amp_active = true_amp >= float(amp_major_threshold)
    if bool(amp_active.any()):
        target_amp = float(amp_target_ratio) * true_amp
        amp_loss = (torch.relu(target_amp - pred_amp)[amp_active] ** 2).mean()
    else:
        amp_loss = zero

    direction_active = peak_valid & (true_steer.abs() >= float(direction_threshold))
    if bool(direction_major_only):
        direction_active = direction_active & amp_active.view(-1, 1)
    if bool(direction_active.any()):
        true_sign = torch.sign(true_steer)
        signed_pred = pred_steer * true_sign
        direction_loss = (torch.relu(float(direction_margin) - signed_pred)[direction_active] ** 2).mean()
    else:
        direction_loss = zero

    return {"amp_loss": amp_loss, "direction_loss": direction_loss}


def compute_distill_reliability_weights(
    teacher_pred_denorm: np.ndarray,
    y_pool: np.ndarray,
    mask_pool: np.ndarray,
    *,
    amp_major_threshold: float,
    amp_min_ratio: float,
    amp_max_ratio: float,
    direction_threshold: float,
    min_weight: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    true_steer = np.asarray(y_pool[:, :, 0], dtype=np.float32)
    teacher_steer = np.asarray(teacher_pred_denorm[:, :, 0], dtype=np.float32)
    valid = np.asarray(mask_pool, dtype=np.float32) > 0.5
    true_abs = np.where(valid, np.abs(true_steer), 0.0)
    teacher_abs = np.where(valid, np.abs(teacher_steer), 0.0)
    true_amp = true_abs.max(axis=1)
    teacher_amp = teacher_abs.max(axis=1)
    true_peak_idx = np.argmax(true_abs, axis=1)
    row_idx = np.arange(true_steer.shape[0])
    true_peak = true_steer[row_idx, true_peak_idx]
    teacher_at_true_peak = teacher_steer[row_idx, true_peak_idx]
    major = true_amp >= float(amp_major_threshold)
    amp_ratio = teacher_amp / np.maximum(true_amp, EPS)
    amp_ok = (amp_ratio >= float(amp_min_ratio)) & (amp_ratio <= float(amp_max_ratio))
    true_sign = np.sign(true_peak)
    teacher_sign = np.sign(teacher_at_true_peak)
    direction_ok = (true_sign == 0) | ((teacher_sign == true_sign) & (np.abs(teacher_at_true_peak) >= float(direction_threshold)))
    reliable = (~major) | (amp_ok & direction_ok)
    weights = np.ones((true_steer.shape[0],), dtype=np.float32)
    weights[major & ~reliable] = float(min_weight)
    meta = {
        "enabled": True,
        "amp_major_threshold": float(amp_major_threshold),
        "amp_min_ratio": float(amp_min_ratio),
        "amp_max_ratio": float(amp_max_ratio),
        "direction_threshold": float(direction_threshold),
        "min_weight": float(min_weight),
        "sample_count": int(len(weights)),
        "major_count": int(major.sum()),
        "downweighted_count": int((weights < 0.999).sum()),
        "downweighted_rate": float((weights < 0.999).mean()),
        "major_downweighted_rate": float((major & (weights < 0.999)).sum() / max(int(major.sum()), 1)),
        "amp_unreliable_count": int((major & ~amp_ok).sum()),
        "direction_unreliable_count": int((major & ~direction_ok).sum()),
        "weight_mean": float(weights.mean()),
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
    }
    return weights, meta


def _normalize_x_with_stats(x_pool: np.ndarray, norm_stats: dict[str, Any]) -> np.ndarray:
    feat_mean = np.asarray(norm_stats["feat_mean"], dtype=np.float32).reshape(1, 1, -1)
    feat_std = np.asarray(norm_stats["feat_std"], dtype=np.float32).reshape(1, 1, -1)
    return ((np.asarray(x_pool, dtype=np.float32) - feat_mean) / feat_std).astype(np.float32)


def _teacher_config_value(config: dict[str, Any], key: str, default: Any) -> Any:
    value = config.get(key, default)
    return default if value is None else value


def precompute_distill_targets_from_teacher(
    teacher_checkpoint: str | Path,
    x_pool: np.ndarray,
    y_pool: np.ndarray,
    curve_pool: np.ndarray,
    base_ctx_pool: np.ndarray,
    mask_pool: np.ndarray,
    meta_df: pd.DataFrame,
    train_idx: list[int],
    event_targets: dict[str, np.ndarray],
    student_norm_stats: dict[str, np.ndarray],
    batch_size: int,
    device: str,
    enable_reliability_weighting: bool = False,
    reliability_min_weight: float = 0.25,
    reliability_amp_min_ratio: float = 0.65,
    reliability_amp_max_ratio: float = 1.60,
    reliability_direction_threshold: float = 0.05,
    reliability_amp_major_threshold: float = 0.20,
    enable_hardcase_weighting: bool = False,
    hardcase_extra_weight: float = 0.50,
    hardcase_amp_threshold: float = 0.30,
    hardcase_late_peak_threshold_s: float = 1.20,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    checkpoint_path = Path(teacher_checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"distill teacher checkpoint does not exist: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    teacher_config = dict(ckpt.get("args", {}))
    teacher_norm_stats = ckpt.get("norm_stats")
    if teacher_norm_stats is None:
        raise ValueError(f"teacher checkpoint has no norm_stats: {checkpoint_path}")
    teacher_args = argparse.Namespace(**teacher_config)
    teacher_ctx_pool, teacher_context_meta = build_optional_context_augmentation(
        ctx_pool=base_ctx_pool,
        meta_df=meta_df,
        train_idx=train_idx,
        args=teacher_args,
    )
    teacher_x_norm = _normalize_x_with_stats(x_pool, teacher_norm_stats)
    teacher_ds = EventConditionedDataset(
        X_norm=teacher_x_norm,
        y_pool=y_pool,
        curve_pool=curve_pool,
        ctx_pool=teacher_ctx_pool,
        mask_pool=mask_pool,
        norm_stats=teacher_norm_stats,
        event_targets=event_targets,
        meta_df=meta_df.reset_index(drop=True),
    )
    teacher_loader = make_loader(teacher_ds, batch_size=batch_size, shuffle=False, seed=int(_teacher_config_value(teacher_config, "seed", 2026)) + 1701)
    teacher_model = EventConditionedTrajectoryModel(
        input_dim=int(teacher_ds.src.shape[-1]),
        context_dim=int(teacher_ds.ctx.shape[-1]),
        future_len=FUTURE_LEN,
        event_bin_size=int(_teacher_config_value(teacher_config, "event_bin_size", 20)),
        d_model=int(_teacher_config_value(teacher_config, "d_model", 128)),
        nhead=int(_teacher_config_value(teacher_config, "nhead", 2)),
        enc_layers=int(_teacher_config_value(teacher_config, "enc_layers", 2)),
        dec_layers=int(_teacher_config_value(teacher_config, "dec_layers", 2)),
        ffn_dim=int(_teacher_config_value(teacher_config, "ffn_dim", 256)),
        dropout=float(_teacher_config_value(teacher_config, "dropout", 0.1)),
        event_embed_dim=int(_teacher_config_value(teacher_config, "event_embed_dim", 96)),
        out_dim=2,
        conditioning_mode=str(_teacher_config_value(teacher_config, "conditioning_mode", "vehicle_direct_coarse_fine")),
        structure_width=float(_teacher_config_value(teacher_config, "structure_width", 0.065)),
        gate_temperature=float(_teacher_config_value(teacher_config, "gate_temperature", 0.040)),
        event_residual_scale=float(_teacher_config_value(teacher_config, "event_residual_scale", 1.0)),
        enable_response_type_head=bool(_teacher_config_value(teacher_config, "enable_response_type_head", False)),
        enable_response_type_condition=bool(_teacher_config_value(teacher_config, "enable_response_type_condition", False)),
        response_type_use_context=bool(_teacher_config_value(teacher_config, "response_type_use_context", False)),
        response_type_hidden_dim=int(_teacher_config_value(teacher_config, "response_type_hidden_dim", 96)),
        num_trajectory_candidates=int(_teacher_config_value(teacher_config, "num_trajectory_candidates", 1)),
        candidate_delta_scale=float(_teacher_config_value(teacher_config, "candidate_delta_scale", 1.0)),
        candidate_base_mode=str(_teacher_config_value(teacher_config, "candidate_base_mode", "learned_delta")),
    ).to(device)
    missing, unexpected = teacher_model.load_state_dict(ckpt["model_state"], strict=False)
    if missing or unexpected:
        print(
            f"distill teacher load_state_dict(strict=False): missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )
    teacher_model.eval()
    teacher_y_mean = torch.tensor(np.asarray(teacher_norm_stats["y_mean"], dtype=np.float32), dtype=torch.float32, device=device)
    teacher_y_std = torch.tensor(np.asarray(teacher_norm_stats["y_std"], dtype=np.float32), dtype=torch.float32, device=device)
    student_y_mean = np.asarray(student_norm_stats["y_mean"], dtype=np.float32).reshape(1, 1, -1)
    student_y_std = np.asarray(student_norm_stats["y_std"], dtype=np.float32).reshape(1, 1, -1)

    pred_denorm: list[np.ndarray] = []
    with torch.no_grad():
        for raw_batch in teacher_loader:
            batch = _move_batch_to_device(raw_batch, device=device)
            y_hat, _ = teacher_model(
                src=batch["src"],
                ctx=batch["ctx"],
                curve_norm=batch["curve_norm"],
                event_teacher=None,
                privileged_event_teacher=None,
            )
            y_den = y_hat * teacher_y_std.view(1, 1, -1) + teacher_y_mean.view(1, 1, -1)
            pred_denorm.append(y_den.cpu().numpy())
    teacher_pred_denorm = np.concatenate(pred_denorm, axis=0).astype(np.float32)
    student_target_norm = ((teacher_pred_denorm - student_y_mean) / student_y_std).astype(np.float32)
    distill_weights: np.ndarray | None = None
    if bool(enable_reliability_weighting):
        distill_weights, reliability_meta = compute_distill_reliability_weights(
            teacher_pred_denorm=teacher_pred_denorm,
            y_pool=y_pool,
            mask_pool=mask_pool,
            amp_major_threshold=float(reliability_amp_major_threshold),
            amp_min_ratio=float(reliability_amp_min_ratio),
            amp_max_ratio=float(reliability_amp_max_ratio),
            direction_threshold=float(reliability_direction_threshold),
            min_weight=float(reliability_min_weight),
        )
    else:
        reliability_meta = {"enabled": False}
    hardcase_meta: dict[str, Any] = {"enabled": False}
    if bool(enable_hardcase_weighting):
        response_targets_for_weight = build_response_type_targets(
            y_pool=y_pool,
            mask_pool=mask_pool,
            amp_threshold=float(hardcase_amp_threshold),
            late_peak_threshold_s=float(hardcase_late_peak_threshold_s),
        )
        hardcase_mask = (
            (response_targets_for_weight["response_hard_case"] > 0.5)
            | (response_targets_for_weight["response_reverse"] > 0.5)
            | (response_targets_for_weight["response_multi"] > 0.5)
        )
        hardcase_multiplier = np.where(hardcase_mask, 1.0 + float(hardcase_extra_weight), 1.0).astype(np.float32)
        if distill_weights is None:
            distill_weights = hardcase_multiplier
        else:
            distill_weights = (distill_weights.astype(np.float32) * hardcase_multiplier).astype(np.float32)
        hardcase_meta = {
            "enabled": True,
            "extra_weight": float(hardcase_extra_weight),
            "amp_threshold": float(hardcase_amp_threshold),
            "late_peak_threshold_s": float(hardcase_late_peak_threshold_s),
            "hardcase_count": int(hardcase_mask.sum()),
            "weight_mean_after_hardcase": float(distill_weights.mean()),
            "weight_min_after_hardcase": float(distill_weights.min()),
            "weight_max_after_hardcase": float(distill_weights.max()),
        }
    del teacher_model
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    meta = {
        "enabled": True,
        "teacher_checkpoint": str(checkpoint_path),
        "teacher_config": teacher_config,
        "teacher_context": teacher_context_meta,
        "target_shape": list(student_target_norm.shape),
        "target_unit": "student-normalized teacher trajectory",
        "train_only": True,
        "reliability_weighting": reliability_meta,
        "hardcase_weighting": hardcase_meta,
    }
    return student_target_norm, distill_weights, meta


def _move_batch_to_device(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    float_keys = {
        "src",
        "y_true",
        "curve_norm",
        "ctx",
        "ctx_raw",
        "event_mask",
        "distill_y_soft",
        "distill_sample_weight",
        "response_high_amp",
        "response_reverse",
        "response_multi",
        "response_late_peak",
        "response_hard_case",
    }
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            if key.endswith("_has"):
                out[key] = value.to(device=device, dtype=torch.float32)
            elif key in float_keys:
                out[key] = value.to(device=device, dtype=torch.float32)
            else:
                out[key] = value.to(device=device)
    return out


def _compute_event_metrics(batch: dict[str, torch.Tensor], event_logits: dict[str, torch.Tensor]) -> dict[str, float]:
    turn_has_pred = (torch.sigmoid(event_logits["first_major_turn_onset_has_logit"]) >= 0.5).to(dtype=torch.float32)
    reversal_has_pred = (torch.sigmoid(event_logits["first_reversal_has_logit"]) >= 0.5).to(dtype=torch.float32)
    peak_idx_pred = torch.argmax(event_logits["main_peak_idx_logits"], dim=1)

    turn_has_acc = (turn_has_pred == batch["first_major_turn_onset_has"]).to(dtype=torch.float32).mean().item()
    reversal_has_acc = (reversal_has_pred == batch["first_reversal_has"]).to(dtype=torch.float32).mean().item()

    valid_peak = (batch["event_mask"].sum(dim=1) > 0)
    if valid_peak.any():
        peak_mae = (peak_idx_pred[valid_peak] - batch["main_peak_idx"][valid_peak]).abs().to(dtype=torch.float32).mean().item()
    else:
        peak_mae = 0.0
    return {
        "turn_has_acc": float(turn_has_acc),
        "reversal_has_acc": float(reversal_has_acc),
        "main_peak_idx_mae": float(peak_mae),
    }


def evaluate_epoch(
    model: EventConditionedTrajectoryModel,
    loader: DataLoader,
    meta_df: pd.DataFrame,
    split_name: str,
    seed: int,
    device: str,
    y_mean_t: torch.Tensor,
    y_std_t: torch.Tensor,
    event_loss_weight: float,
    use_privileged_teacher: bool = False,
) -> dict[str, float]:
    model.eval()
    traj_loss_total = 0.0
    event_loss_total = 0.0
    total_loss = 0.0
    rmse_steer_num = 0.0
    rmse_speed_num = 0.0
    rmse_den = 0.0
    metric_accum = {"turn_has_acc": 0.0, "reversal_has_acc": 0.0, "main_peak_idx_mae": 0.0}
    n_batch = 0
    preds: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    ctxs_raw: list[np.ndarray] = []

    with torch.no_grad():
        for raw_batch in loader:
            batch = _move_batch_to_device(raw_batch, device=device)
            privileged_teacher = None
            if use_privileged_teacher and "privileged_event_teacher" in batch:
                privileged_teacher = batch["privileged_event_teacher"]
            y_hat, extras = model(
                src=batch["src"],
                ctx=batch["ctx"],
                curve_norm=batch["curve_norm"],
                event_teacher=None,
                privileged_event_teacher=privileged_teacher,
            )
            traj_mask = batch["event_mask"].unsqueeze(-1)
            traj_loss = masked_mse(y_hat, batch["y_true"], traj_mask)
            event_breakdown = compute_event_loss(batch, extras["event_logits"])
            loss = traj_loss + event_loss_weight * event_breakdown.total

            y_hat_den = y_hat * y_std_t.view(1, 1, -1) + y_mean_t.view(1, 1, -1)
            y_true_den = batch["y_true"] * y_std_t.view(1, 1, -1) + y_mean_t.view(1, 1, -1)
            mask = batch["event_mask"]
            rmse_steer_num += float((((y_hat_den[:, :, 0] - y_true_den[:, :, 0]) ** 2) * mask).sum().item())
            rmse_speed_num += float((((y_hat_den[:, :, 1] - y_true_den[:, :, 1]) ** 2) * mask).sum().item())
            rmse_den += float(mask.sum().item())
            preds.append(y_hat_den.cpu().numpy())
            trues.append(y_true_den.cpu().numpy())
            masks.append(mask.cpu().numpy())
            ctxs_raw.append(raw_batch["ctx_raw"].cpu().numpy())

            metrics = _compute_event_metrics(batch, extras["event_logits"])
            for key, value in metrics.items():
                metric_accum[key] += value

            traj_loss_total += float(traj_loss.item())
            event_loss_total += float(event_breakdown.total.item())
            total_loss += float(loss.item())
            n_batch += 1

    denom = max(n_batch, 1)
    rmse_den = max(rmse_den, 1.0)
    pred = np.concatenate(preds, axis=0)
    true = np.concatenate(trues, axis=0)
    mask = np.concatenate(masks, axis=0)
    ctx_raw = np.concatenate(ctxs_raw, axis=0)
    selection_bundle = build_primary_selection_bundle(
        pred=pred,
        true=true,
        mask=mask,
        ctx_raw=ctx_raw,
        meta_df=meta_df,
        split_name=split_name,
        seed=seed,
    )
    out = {
        "loss": total_loss / denom,
        "traj_loss": traj_loss_total / denom,
        "event_loss": event_loss_total / denom,
        "steer_rmse": float(np.sqrt(rmse_steer_num / rmse_den)),
        "speed_rmse": float(np.sqrt(rmse_speed_num / rmse_den)),
        "turn_has_acc": metric_accum["turn_has_acc"] / denom,
        "reversal_has_acc": metric_accum["reversal_has_acc"] / denom,
        "main_peak_idx_mae": metric_accum["main_peak_idx_mae"] / denom,
    }
    out.update(
        {
            "selection_summary": selection_bundle["selection_summary"],
            "trajectory_sample_df": selection_bundle["sample_df"],
            "primary_trajectory_sample_df": selection_bundle["primary_sample_df"],
            "weighted_metrics": selection_bundle["weighted"],
            "primary_weighted_metrics": selection_bundle["primary_weighted"],
            "interaction_sample_count": int(
                (selection_bundle["sample_df"].get("interaction_slice", pd.Series([], dtype=object)).astype(str) == "interaction").sum()
            ),
        }
    )
    return out


def _compact_eval_summary(eval_metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "loss": float(eval_metrics["loss"]),
        "traj_loss": float(eval_metrics["traj_loss"]),
        "event_loss": float(eval_metrics["event_loss"]),
        "steer_rmse": float(eval_metrics["steer_rmse"]),
        "speed_rmse": float(eval_metrics["speed_rmse"]),
        "turn_has_acc": float(eval_metrics["turn_has_acc"]),
        "reversal_has_acc": float(eval_metrics["reversal_has_acc"]),
        "main_peak_idx_mae": float(eval_metrics["main_peak_idx_mae"]),
        "interaction_sample_count": int(eval_metrics.get("interaction_sample_count", 0)),
        "selection_summary": {
            key: float(value) if isinstance(value, (int, float, np.floating)) else value
            for key, value in eval_metrics["selection_summary"].items()
        },
    }


def train_one_run(args: argparse.Namespace) -> dict[str, Any]:
    set_determinism(seed=int(args.seed))
    run_name = f"{args.run_prefix}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_root = RUN_ROOT / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    max_train = args.max_train_samples
    max_val = args.max_val_samples
    max_test = args.max_test_samples
    epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    if bool(args.smoke_test):
        epochs = int(args.smoke_epochs)
        batch_size = int(args.smoke_batch_size)
        max_train = int(args.smoke_train_samples)
        max_val = int(args.smoke_val_samples)
        max_test = int(args.smoke_test_samples)

    sample_bundle = build_sample_bundle_from_manifest(
        manifest_path=args.manifest,
        max_train_samples=max_train,
        max_val_samples=max_val,
        max_test_samples=max_test,
        seed=int(args.seed),
    )
    X_pool, y_pool, curve_pool, ctx_pool, mask_pool, meta_df, dropped_count = sample_bundle
    base_ctx_pool = np.asarray(ctx_pool, dtype=np.float32)
    meta_df = annotate_event_meta(meta_df, y_pool, mask_pool)

    split_series = meta_df["split"].astype(str).reset_index(drop=True)
    train_idx = split_series.index[split_series == "train"].tolist()
    val_idx = split_series.index[split_series == "val"].tolist()
    test_idx = split_series.index[split_series == "test"].tolist()
    if not train_idx or not val_idx or not test_idx:
        raise RuntimeError("Split samples are incomplete after filtering; check manifest subset settings.")

    print(
        f"sample split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} "
        f"dropped={dropped_count} input_dim={X_pool.shape[-1]}",
        flush=True,
    )
    ctx_pool, context_augmentation_meta = apply_optional_context_augmentation(
        ctx_pool=ctx_pool,
        meta_df=meta_df,
        train_idx=train_idx,
        args=args,
        run_root=run_root,
    )

    X_norm, norm_stats = normalize_inputs(X_pool, y_pool, curve_pool, ctx_pool, train_idx)
    event_targets = build_event_schema_targets(
        y_pool=y_pool,
        mask_pool=mask_pool,
        future_len=FUTURE_LEN,
        event_bin_size=int(args.event_bin_size),
    )
    response_targets = None
    if bool(getattr(args, "enable_response_type_head", False)) or bool(getattr(args, "enable_response_type_condition", False)):
        response_targets = build_response_type_targets(
            y_pool=y_pool,
            mask_pool=mask_pool,
            amp_threshold=float(args.response_type_amp_threshold),
            late_peak_threshold_s=float(args.response_type_late_peak_threshold_s),
        )
    candidate_prototypes = None
    candidate_prototype_meta: dict[str, Any] = {"enabled": False}
    if str(getattr(args, "candidate_base_mode", "learned_delta")) == "response_prototype":
        candidate_prototypes, candidate_prototype_meta = build_response_candidate_prototypes(
            y_pool=y_pool,
            train_idx=train_idx,
            response_targets=response_targets,
            norm_stats=norm_stats,
            num_candidates=int(args.num_trajectory_candidates),
        )
        if candidate_prototypes is not None:
            proto_path = run_root / "candidate_prototypes_norm.npy"
            np.save(proto_path, candidate_prototypes.astype(np.float32))
            candidate_prototype_meta["path"] = str(proto_path)
            setattr(args, "candidate_prototype_path", str(proto_path))
        save_json(run_root / "candidate_prototype_meta.json", candidate_prototype_meta)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    distill_targets_norm: np.ndarray | None = None
    distill_weights: np.ndarray | None = None
    distillation_meta: dict[str, Any] = {"enabled": False}
    if str(getattr(args, "distill_teacher_checkpoint", "") or "").strip():
        distill_targets_norm, distill_weights, distillation_meta = precompute_distill_targets_from_teacher(
            teacher_checkpoint=str(args.distill_teacher_checkpoint),
            x_pool=X_pool,
            y_pool=y_pool,
            curve_pool=curve_pool,
            base_ctx_pool=base_ctx_pool,
            mask_pool=mask_pool,
            meta_df=meta_df,
            train_idx=train_idx,
            event_targets=event_targets,
            student_norm_stats=norm_stats,
            batch_size=batch_size,
            device=str(device),
            enable_reliability_weighting=bool(args.distill_reliability_weighting),
            reliability_min_weight=float(args.distill_reliability_min_weight),
            reliability_amp_min_ratio=float(args.distill_reliability_amp_min_ratio),
            reliability_amp_max_ratio=float(args.distill_reliability_amp_max_ratio),
            reliability_direction_threshold=float(args.distill_reliability_direction_threshold),
            reliability_amp_major_threshold=float(args.steer_amp_major_threshold),
            enable_hardcase_weighting=bool(args.distill_hardcase_weighting),
            hardcase_extra_weight=float(args.distill_hardcase_extra_weight),
            hardcase_amp_threshold=float(args.distill_hardcase_amp_threshold),
            hardcase_late_peak_threshold_s=float(args.distill_hardcase_late_peak_threshold_s),
        )
        distillation_meta.update(
            {
                "distill_weight": float(args.distill_weight),
                "distill_tail_weight": float(args.distill_tail_weight),
            }
        )
        if distill_weights is not None:
            weight_path = run_root / "distill_reliability_weights.csv"
            pd.DataFrame(
                {
                    "sample_index": np.arange(len(distill_weights), dtype=np.int64),
                    "split": meta_df["split"].astype(str).to_numpy(),
                    "distill_sample_weight": distill_weights.astype(np.float32),
                }
            ).to_csv(weight_path, index=False)
            distillation_meta["reliability_weight_path"] = str(weight_path)
        save_json(run_root / "distillation_meta.json", distillation_meta)

    train_ds = EventConditionedDataset(
        X_norm=X_norm[train_idx],
        y_pool=y_pool[train_idx],
        curve_pool=curve_pool[train_idx],
        ctx_pool=ctx_pool[train_idx],
        mask_pool=mask_pool[train_idx],
        norm_stats=norm_stats,
        event_targets=subset_array_dict(event_targets, train_idx),
        meta_df=meta_df.iloc[train_idx].reset_index(drop=True),
        distill_target_pool=None if distill_targets_norm is None else distill_targets_norm[train_idx],
        distill_weight_pool=None if distill_weights is None else distill_weights[train_idx],
        response_targets=None if response_targets is None else subset_array_dict(response_targets, train_idx),
    )
    val_ds = EventConditionedDataset(
        X_norm=X_norm[val_idx],
        y_pool=y_pool[val_idx],
        curve_pool=curve_pool[val_idx],
        ctx_pool=ctx_pool[val_idx],
        mask_pool=mask_pool[val_idx],
        norm_stats=norm_stats,
        event_targets=subset_array_dict(event_targets, val_idx),
        meta_df=meta_df.iloc[val_idx].reset_index(drop=True),
        distill_target_pool=None if distill_targets_norm is None else distill_targets_norm[val_idx],
        response_targets=None if response_targets is None else subset_array_dict(response_targets, val_idx),
    )
    test_ds = EventConditionedDataset(
        X_norm=X_norm[test_idx],
        y_pool=y_pool[test_idx],
        curve_pool=curve_pool[test_idx],
        ctx_pool=ctx_pool[test_idx],
        mask_pool=mask_pool[test_idx],
        norm_stats=norm_stats,
        event_targets=subset_array_dict(event_targets, test_idx),
        meta_df=meta_df.iloc[test_idx].reset_index(drop=True),
        distill_target_pool=None if distill_targets_norm is None else distill_targets_norm[test_idx],
        response_targets=None if response_targets is None else subset_array_dict(response_targets, test_idx),
    )

    train_loader = make_loader(train_ds, batch_size=batch_size, shuffle=True, seed=int(args.seed) + 101)
    val_loader = make_loader(val_ds, batch_size=batch_size, shuffle=False, seed=int(args.seed) + 103)
    test_loader = make_loader(test_ds, batch_size=batch_size, shuffle=False, seed=int(args.seed) + 107)

    print(
        f"training: device={device} epochs={epochs} batch_size={batch_size} "
        f"ctx_dim={ctx_pool.shape[-1]} conditioning={args.conditioning_mode}",
        flush=True,
    )

    model = EventConditionedTrajectoryModel(
        input_dim=int(train_ds.src.shape[-1]),
        context_dim=int(train_ds.ctx.shape[-1]),
        future_len=FUTURE_LEN,
        event_bin_size=int(args.event_bin_size),
        d_model=int(args.d_model),
        nhead=int(args.nhead),
        enc_layers=int(args.enc_layers),
        dec_layers=int(args.dec_layers),
        ffn_dim=int(args.ffn_dim),
        dropout=float(args.dropout),
        event_embed_dim=int(args.event_embed_dim),
        out_dim=2,
        conditioning_mode=str(args.conditioning_mode),
        structure_width=float(args.structure_width),
        gate_temperature=float(args.gate_temperature),
        event_residual_scale=float(args.event_residual_scale),
        enable_response_type_head=bool(args.enable_response_type_head),
        enable_response_type_condition=bool(args.enable_response_type_condition),
        response_type_use_context=bool(getattr(args, "response_type_use_context", False)),
        response_type_hidden_dim=int(args.response_type_hidden_dim),
        num_trajectory_candidates=int(args.num_trajectory_candidates),
        candidate_delta_scale=float(args.candidate_delta_scale),
        candidate_base_mode=str(getattr(args, "candidate_base_mode", "learned_delta")),
        candidate_prototypes=candidate_prototypes,
    ).to(device)
    if args.init_checkpoint:
        init_ckpt = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(init_ckpt["model_state"], strict=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    y_mean_t = torch.tensor(norm_stats["y_mean"], dtype=torch.float32, device=device)
    y_std_t = torch.tensor(norm_stats["y_std"], dtype=torch.float32, device=device)

    best_val = float("inf")
    best_epoch = 0
    history: list[dict[str, Any]] = []
    best_ckpt = run_root / "best_model.pt"
    legacy_best_ckpt = run_root / "best_model_legacy.pt"
    structure_best_ckpt = run_root / "best_model_structure.pt"
    teacher_rng = random.Random(int(args.seed) + 999)
    selection_mode = str(args.selection_mode)
    patience = int(args.patience)
    min_epochs = int(args.min_epochs)
    best_structure_key: tuple[float, ...] | None = None
    best_legacy_key: tuple[float, ...] | None = None
    best_structure_epoch = 0
    best_legacy_epoch = 0
    best_structure_summary: dict[str, Any] | None = None
    best_legacy_summary: dict[str, Any] | None = None
    active_best_key: tuple[float, ...] | None = None
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        traj_sum = 0.0
        event_sum = 0.0
        tail_amp_sum = 0.0
        distill_sum = 0.0
        distill_tail_sum = 0.0
        distill_weight_mean_sum = 0.0
        steer_amp_physical_sum = 0.0
        steer_direction_physical_sum = 0.0
        response_type_sum = 0.0
        multi_candidate_sum = 0.0
        candidate_selector_sum = 0.0
        response_label_candidate_sum = 0.0
        n_batch = 0
        teacher_steps = 0

        for raw_batch in train_loader:
            batch = _move_batch_to_device(raw_batch, device=device)
            use_teacher = False
            if float(args.teacher_forcing_ratio) >= 1.0:
                use_teacher = True
            elif float(args.teacher_forcing_ratio) > 0.0:
                use_teacher = teacher_rng.random() < float(args.teacher_forcing_ratio)

            teacher_events = build_event_teacher_from_batch(batch, device=device) if use_teacher else None
            if use_teacher:
                teacher_steps += 1
            privileged_teacher = None
            if bool(args.use_privileged_teacher) and "privileged_event_teacher" in batch:
                privileged_teacher = batch["privileged_event_teacher"]

            optimizer.zero_grad()
            y_hat, extras = model(
                src=batch["src"],
                ctx=batch["ctx"],
                curve_norm=batch["curve_norm"],
                event_teacher=teacher_events,
                privileged_event_teacher=privileged_teacher,
            )
            traj_mask = batch["event_mask"].unsqueeze(-1)
            traj_loss = masked_mse(y_hat, batch["y_true"], traj_mask)
            multi_candidate_losses = compute_multi_candidate_losses(
                extras=extras,
                y_true=batch["y_true"],
                traj_mask=traj_mask,
                batch=batch,
                target_mode=str(getattr(args, "multi_candidate_target_mode", "oracle")),
            )
            event_breakdown = compute_event_loss(batch, extras["event_logits"])
            response_type_loss = compute_response_type_loss(batch, extras.get("response_type_logits"))
            physical_losses = compute_steer_physical_losses(
                y_hat=y_hat,
                y_true_norm=batch["y_true"],
                traj_mask=traj_mask,
                y_mean_t=y_mean_t,
                y_std_t=y_std_t,
                amp_major_threshold=float(getattr(args, "steer_amp_major_threshold", 0.20)),
                amp_target_ratio=float(getattr(args, "steer_amp_target_ratio", 0.85)),
                direction_threshold=float(getattr(args, "steer_direction_threshold", 0.10)),
                direction_margin=float(getattr(args, "steer_direction_margin", 0.03)),
                peak_window_steps=int(getattr(args, "steer_physical_peak_window_steps", -1)),
                amp_peak_window_only=bool(getattr(args, "steer_amp_peak_window_only", False)),
                direction_major_only=bool(getattr(args, "steer_direction_major_only", False)),
            )
            # Tail amplitude penalty (steer channel only, steps >= TAIL_START).
            tail_mask = traj_mask[:, TAIL_START:, :]
            pred_amp = y_hat[:, TAIL_START:, 0:1].abs()
            true_amp = batch["y_true"][:, TAIL_START:, 0:1].abs()
            tail_amp_loss = masked_mse(pred_amp, true_amp, tail_mask)
            distill_loss = traj_loss.new_zeros(())
            distill_tail_loss = traj_loss.new_zeros(())
            distill_weight_mean = 0.0
            if "distill_y_soft" in batch:
                sample_weight = batch.get("distill_sample_weight")
                distill_loss = weighted_masked_mse(
                    pred=y_hat,
                    true=batch["distill_y_soft"],
                    mask=traj_mask,
                    sample_weight=sample_weight,
                )
                distill_tail_loss = weighted_masked_mse(
                    pred=y_hat[:, TAIL_START:, :],
                    true=batch["distill_y_soft"][:, TAIL_START:, :],
                    mask=tail_mask,
                    sample_weight=sample_weight,
                )
                distill_weight_mean = float(sample_weight.mean().item()) if sample_weight is not None else 1.0
            loss = (
                float(getattr(args, "trajectory_loss_weight", 1.0)) * traj_loss
                + float(args.event_loss_weight) * event_breakdown.total
                + W_TAIL_AMP * tail_amp_loss
                + float(getattr(args, "response_type_loss_weight", 0.0)) * response_type_loss
                + float(getattr(args, "multi_candidate_loss_weight", 0.0)) * multi_candidate_losses["min_loss"]
                + float(getattr(args, "candidate_selector_loss_weight", 0.0)) * multi_candidate_losses["selector_loss"]
                + float(getattr(args, "distill_weight", 0.0)) * distill_loss
                + float(getattr(args, "distill_tail_weight", 0.0)) * distill_tail_loss
                + float(getattr(args, "steer_amp_loss_weight", 0.0)) * physical_losses["amp_loss"]
                + float(getattr(args, "steer_direction_loss_weight", 0.0)) * physical_losses["direction_loss"]
            )
            loss.backward()
            if float(args.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()

            loss_sum += float(loss.item())
            traj_sum += float(traj_loss.item())
            event_sum += float(event_breakdown.total.item())
            tail_amp_sum += float(tail_amp_loss.item())
            distill_sum += float(distill_loss.item())
            distill_tail_sum += float(distill_tail_loss.item())
            distill_weight_mean_sum += float(distill_weight_mean)
            steer_amp_physical_sum += float(physical_losses["amp_loss"].item())
            steer_direction_physical_sum += float(physical_losses["direction_loss"].item())
            response_type_sum += float(response_type_loss.item())
            multi_candidate_sum += float(multi_candidate_losses["min_loss"].item())
            candidate_selector_sum += float(multi_candidate_losses["selector_loss"].item())
            response_label_candidate_sum += float(multi_candidate_losses.get("response_label_loss", traj_loss.new_zeros(())).item())
            n_batch += 1

        train_metrics = {
            "loss": loss_sum / max(n_batch, 1),
            "traj_loss": traj_sum / max(n_batch, 1),
            "event_loss": event_sum / max(n_batch, 1),
            "tail_amp_loss": tail_amp_sum / max(n_batch, 1),
            "distill_loss": distill_sum / max(n_batch, 1),
            "distill_tail_loss": distill_tail_sum / max(n_batch, 1),
            "distill_weight_mean": distill_weight_mean_sum / max(n_batch, 1),
            "steer_amp_physical_loss": steer_amp_physical_sum / max(n_batch, 1),
            "steer_direction_physical_loss": steer_direction_physical_sum / max(n_batch, 1),
            "response_type_loss": response_type_sum / max(n_batch, 1),
            "multi_candidate_loss": multi_candidate_sum / max(n_batch, 1),
            "candidate_selector_loss": candidate_selector_sum / max(n_batch, 1),
            "response_label_candidate_loss": response_label_candidate_sum / max(n_batch, 1),
            "teacher_step_ratio": teacher_steps / max(n_batch, 1),
        }
        val_metrics = evaluate_epoch(
            model=model,
            loader=val_loader,
            meta_df=val_ds.meta_df,
            split_name="val",
            seed=int(args.seed),
            device=device,
            y_mean_t=y_mean_t,
            y_std_t=y_std_t,
            event_loss_weight=float(args.event_loss_weight),
            use_privileged_teacher=bool(args.use_privileged_teacher),
        )
        selection_summary = val_metrics["selection_summary"]
        structure_key = structure_aware_selection_key(selection_summary)
        legacy_key = (float(val_metrics["steer_rmse"]),)

        epoch_log = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_traj_loss": train_metrics["traj_loss"],
            "train_event_loss": train_metrics["event_loss"],
            "train_tail_amp_loss": train_metrics["tail_amp_loss"],
            "train_distill_loss": train_metrics["distill_loss"],
            "train_distill_tail_loss": train_metrics["distill_tail_loss"],
            "train_response_type_loss": train_metrics["response_type_loss"],
            "train_multi_candidate_loss": train_metrics["multi_candidate_loss"],
            "train_candidate_selector_loss": train_metrics["candidate_selector_loss"],
            "train_response_label_candidate_loss": train_metrics["response_label_candidate_loss"],
            "train_distill_weight_mean": train_metrics["distill_weight_mean"],
            "train_steer_amp_physical_loss": train_metrics["steer_amp_physical_loss"],
            "train_steer_direction_physical_loss": train_metrics["steer_direction_physical_loss"],
            "train_teacher_step_ratio": train_metrics["teacher_step_ratio"],
            "val_loss": val_metrics["loss"],
            "val_steer_rmse": val_metrics["steer_rmse"],
            "val_speed_rmse": val_metrics["speed_rmse"],
            "val_turn_has_acc": val_metrics["turn_has_acc"],
            "val_reversal_has_acc": val_metrics["reversal_has_acc"],
            "val_main_peak_idx_mae": val_metrics["main_peak_idx_mae"],
            "val_selection_mode": selection_mode,
            "val_selection_score": float(selection_summary["selection_score"]),
            "val_primary_rmse_score": float(selection_summary["primary_rmse_score"]),
            "val_trajectory_score": float(selection_summary["trajectory_score"]),
            "val_tail_score": float(selection_summary["tail_score"]),
            "val_trend_score": float(selection_summary["trend_score"]),
            "val_turning_score": float(selection_summary["turning_score"]),
            "val_continuity_score": float(selection_summary["continuity_score"]),
            "val_tail_rmse": float(selection_summary["rmse_tail_abs_steer"]),
            "val_tail_pre_ratio": float(selection_summary["tail_pre_ratio_abs_steer"]),
            "val_tail_trend_corr": float(selection_summary["tail_trend_corr"]),
            "val_turning_count_abs_err": float(selection_summary["turning_count_abs_err"]),
            "val_peak_time_abs_err_s": float(selection_summary["peak_time_abs_err_s"]),
            "val_boundary_shift_abs_err": float(selection_summary["boundary_shift_abs_err"]),
            "val_interaction_sample_count": int(val_metrics["interaction_sample_count"]),
        }
        history.append(epoch_log)

        checkpoint_payload = {
            "model_state": model.state_dict(),
            "args": vars(args),
            "norm_stats": norm_stats,
            "epoch": int(epoch),
            "selection_summary": selection_summary,
        }
        if best_structure_key is None or structure_key < best_structure_key:
            best_structure_key = structure_key
            best_structure_epoch = int(epoch)
            best_structure_summary = _compact_eval_summary(val_metrics)
            safe_torch_save(checkpoint_payload, structure_best_ckpt)
        if best_legacy_key is None or legacy_key < best_legacy_key:
            best_legacy_key = legacy_key
            best_legacy_epoch = int(epoch)
            best_legacy_summary = _compact_eval_summary(val_metrics)
            safe_torch_save(checkpoint_payload, legacy_best_ckpt)

        active_key = structure_key if selection_mode == "structure_aware_primary" else legacy_key
        if active_best_key is None or active_key < active_best_key:
            active_best_key = active_key
            best_val = float(val_metrics["steer_rmse"])
            best_epoch = int(epoch)
            bad_epochs = 0
            safe_torch_save(checkpoint_payload, best_ckpt)
        else:
            bad_epochs += 1

        print(
            f"Epoch {epoch:03d}/{epochs:03d} "
            f"train_loss={train_metrics['loss']:.5f} "
            f"val_rmse={val_metrics['steer_rmse']:.5f} "
            f"val_tail={float(selection_summary['rmse_tail_abs_steer']):.5f} "
            f"selection={float(selection_summary['selection_score']):.5f} "
            f"best_epoch={best_epoch}",
            flush=True,
        )

        if epoch >= min_epochs and bad_epochs >= patience:
            break

    active_ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(active_ckpt["model_state"])
    val_metrics = evaluate_epoch(
        model=model,
        loader=val_loader,
        meta_df=val_ds.meta_df,
        split_name="val",
        seed=int(args.seed),
        device=device,
        y_mean_t=y_mean_t,
        y_std_t=y_std_t,
        event_loss_weight=float(args.event_loss_weight),
        use_privileged_teacher=bool(args.use_privileged_teacher),
    )
    test_metrics = evaluate_epoch(
        model=model,
        loader=test_loader,
        meta_df=test_ds.meta_df,
        split_name="test",
        seed=int(args.seed),
        device=device,
        y_mean_t=y_mean_t,
        y_std_t=y_std_t,
        event_loss_weight=float(args.event_loss_weight),
        use_privileged_teacher=bool(args.use_privileged_teacher),
    )

    selection_compare_rows: list[dict[str, Any]] = []
    compare_payload: dict[str, Any] = {}
    for tag, ckpt_path in (("legacy", legacy_best_ckpt), ("structure", structure_best_ckpt), ("active", best_ckpt)):
        if not ckpt_path.exists():
            continue
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        val_eval = evaluate_epoch(
            model=model,
            loader=val_loader,
            meta_df=val_ds.meta_df,
            split_name="val",
            seed=int(args.seed),
            device=device,
            y_mean_t=y_mean_t,
            y_std_t=y_std_t,
            event_loss_weight=float(args.event_loss_weight),
            use_privileged_teacher=bool(args.use_privileged_teacher),
        )
        test_eval = evaluate_epoch(
            model=model,
            loader=test_loader,
            meta_df=test_ds.meta_df,
            split_name="test",
            seed=int(args.seed),
            device=device,
            y_mean_t=y_mean_t,
            y_std_t=y_std_t,
            event_loss_weight=float(args.event_loss_weight),
            use_privileged_teacher=bool(args.use_privileged_teacher),
        )
        selection_summary_eval = val_eval["selection_summary"]
        selection_compare_rows.append(
            {
                "selection_tag": tag,
                "epoch": int(ckpt.get("epoch", 0)),
                "val_steer_rmse": float(val_eval["steer_rmse"]),
                "val_selection_score": float(selection_summary_eval["selection_score"]),
                "val_trajectory_score": float(selection_summary_eval["trajectory_score"]),
                "val_turning_score": float(selection_summary_eval["turning_score"]),
                "val_tail_trend_corr": float(selection_summary_eval["tail_trend_corr"]),
                "val_tail_rmse": float(selection_summary_eval["rmse_tail_abs_steer"]),
                "val_boundary_shift_abs_err": float(selection_summary_eval["boundary_shift_abs_err"]),
                "test_steer_rmse": float(test_eval["steer_rmse"]),
                "test_selection_score": float(test_eval["selection_summary"]["selection_score"]),
                "test_tail_trend_corr": float(test_eval["selection_summary"]["tail_trend_corr"]),
                "test_tail_rmse": float(test_eval["selection_summary"]["rmse_tail_abs_steer"]),
            }
        )
        compare_payload[tag] = {
            "epoch": int(ckpt.get("epoch", 0)),
            "val": _compact_eval_summary(val_eval),
            "test": _compact_eval_summary(test_eval),
        }

    history_df = pd.DataFrame(history)
    history_df.to_csv(run_root / "loss_history.csv", index=False)
    pd.DataFrame(selection_compare_rows).to_csv(run_root / "selection_comparison.csv", index=False)
    meta_df.to_csv(run_root / "sample_manifest_used.csv", index=False)
    save_json(
        run_root / "run_summary.json",
        {
            "run_root": str(run_root),
            "smoke_test": bool(args.smoke_test),
            "manifest": str(args.manifest),
            "dropped_samples": int(dropped_count),
            "device": str(device),
            "parameter_count": count_parameters(model),
            "selection_mode": selection_mode,
            "best_epoch": int(best_epoch),
            "best_val_steer_rmse": float(best_val),
            "best_structure_epoch": int(best_structure_epoch),
            "best_legacy_epoch": int(best_legacy_epoch),
            "final_val_metrics": _compact_eval_summary(val_metrics),
            "final_test_metrics": _compact_eval_summary(test_metrics),
            "selection_compare": compare_payload,
            "context_augmentation": context_augmentation_meta,
            "distillation": distillation_meta,
            "candidate_prototypes": candidate_prototype_meta,
            "config": vars(args),
        },
    )
    save_json(
        run_root / "metrics.json",
        {
            "val": _compact_eval_summary(val_metrics),
            "test": _compact_eval_summary(test_metrics),
        },
    )
    return {
        "run_root": str(run_root),
        "best_epoch": int(best_epoch),
        "best_val_steer_rmse": float(best_val),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "selection_mode": selection_mode,
        "best_structure_epoch": int(best_structure_epoch),
        "best_legacy_epoch": int(best_legacy_epoch),
        "selection_compare_path": str(run_root / "selection_comparison.csv"),
        "context_augmentation": context_augmentation_meta,
        "distillation": distillation_meta,
        "candidate_prototypes": candidate_prototype_meta,
        "parameter_count": count_parameters(model),
        "dropped_samples": int(dropped_count),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--run-prefix", default="EXP_EVENT_CONDITIONED_TRAJECTORY_BASELINE")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--init-checkpoint", default=None)

    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--min-epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--event-loss-weight", type=float, default=0.50)
    parser.add_argument("--trajectory-loss-weight", type=float, default=1.0)
    parser.add_argument("--teacher-forcing-ratio", type=float, default=1.0)
    parser.add_argument(
        "--selection-mode",
        default="legacy_rmse",
        choices=["legacy_rmse", "structure_aware_primary"],
    )

    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=2)
    parser.add_argument("--enc-layers", type=int, default=2)
    parser.add_argument("--dec-layers", type=int, default=2)
    parser.add_argument("--ffn-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--event-embed-dim", type=int, default=96)
    parser.add_argument("--event-bin-size", type=int, default=20)
    parser.add_argument(
        "--conditioning-mode",
        default="baseline",
        choices=[
            "vehicle_direct",
            "vehicle_direct_coarse_fine",
            "baseline",
            "structured_v2",
            "structured_v2_coarse_fine",
        ],
    )
    parser.add_argument("--structure-width", type=float, default=0.065)
    parser.add_argument("--gate-temperature", type=float, default=0.040)
    parser.add_argument("--event-residual-scale", type=float, default=1.0)
    parser.add_argument("--use-privileged-teacher", action="store_true")

    parser.add_argument("--enable-response-type-head", action="store_true")
    parser.add_argument("--enable-response-type-condition", action="store_true")
    parser.add_argument("--response-type-use-context", action="store_true")
    parser.add_argument("--response-type-hidden-dim", type=int, default=96)
    parser.add_argument("--response-type-loss-weight", type=float, default=0.0)
    parser.add_argument("--response-type-amp-threshold", type=float, default=0.30)
    parser.add_argument("--response-type-late-peak-threshold-s", type=float, default=1.20)
    parser.add_argument("--num-trajectory-candidates", type=int, default=1)
    parser.add_argument("--candidate-delta-scale", type=float, default=1.0)
    parser.add_argument(
        "--candidate-base-mode",
        default="learned_delta",
        choices=["learned_delta", "response_prototype"],
    )
    parser.add_argument(
        "--multi-candidate-target-mode",
        default="oracle",
        choices=["oracle", "response_type", "hybrid"],
    )
    parser.add_argument("--multi-candidate-loss-weight", type=float, default=0.0)
    parser.add_argument("--candidate-selector-loss-weight", type=float, default=0.0)

    parser.add_argument("--distill-teacher-checkpoint", default="")
    parser.add_argument("--distill-weight", type=float, default=0.0)
    parser.add_argument("--distill-tail-weight", type=float, default=0.0)
    parser.add_argument("--distill-reliability-weighting", action="store_true")
    parser.add_argument("--distill-reliability-min-weight", type=float, default=0.25)
    parser.add_argument("--distill-reliability-amp-min-ratio", type=float, default=0.65)
    parser.add_argument("--distill-reliability-amp-max-ratio", type=float, default=1.60)
    parser.add_argument("--distill-reliability-direction-threshold", type=float, default=0.05)
    parser.add_argument("--distill-hardcase-weighting", action="store_true")
    parser.add_argument("--distill-hardcase-extra-weight", type=float, default=0.50)
    parser.add_argument("--distill-hardcase-amp-threshold", type=float, default=0.30)
    parser.add_argument("--distill-hardcase-late-peak-threshold-s", type=float, default=1.20)

    parser.add_argument("--steer-amp-loss-weight", type=float, default=0.0)
    parser.add_argument("--steer-direction-loss-weight", type=float, default=0.0)
    parser.add_argument("--steer-amp-major-threshold", type=float, default=0.20)
    parser.add_argument("--steer-amp-target-ratio", type=float, default=0.85)
    parser.add_argument("--steer-direction-threshold", type=float, default=0.10)
    parser.add_argument("--steer-direction-margin", type=float, default=0.03)
    parser.add_argument("--steer-physical-peak-window-steps", type=int, default=-1)
    parser.add_argument("--steer-amp-peak-window-only", action="store_true")
    parser.add_argument("--steer-direction-major-only", action="store_true")

    parser.add_argument("--enable-teacher-state-context", action="store_true")
    parser.add_argument(
        "--teacher-state-mode",
        default="pca_latent",
        choices=[
            "old_ac",
            "pca_latent",
            "raw_physio",
            "raw_physio_no_eeg",
            "raw_hr_only",
            "raw_eda_only",
            "raw_emg_only",
            "semantic_driver_state",
            "semantic_driver_state_no_eeg",
            "semantic_driver_state_eeg_only",
            "semantic_driver_state_hr_only",
            "semantic_driver_state_eda_only",
            "semantic_driver_state_emg_only",
            "raw_eeg_only",
            "semantic_driver_state_local_delta",
            "semantic_driver_state_local_delta_no_eeg",
            "signal_current_delta_hr_only",
            "signal_current_delta_eda_only",
            "signal_current_delta_emg_only",
            "signal_current_delta_eeg_only",
            "signal_current_delta_all",
            "signal_pca_hr_only",
            "signal_pca_eda_only",
            "signal_pca_emg_only",
            "signal_pca_eeg_only",
            "signal_pca_all",
        ],
    )
    parser.add_argument("--teacher-state-dim", type=int, default=4)
    parser.add_argument("--enable-driver-style-context", action="store_true")
    parser.add_argument("--driver-style-vector-csv", default=str(DEFAULT_DRIVER_STYLE_VECTOR_CSV))
    parser.add_argument("--driver-style-embed-dim", type=int, default=4)
    parser.add_argument("--driver-style-include-iqr", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)

    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-epochs", type=int, default=2)
    parser.add_argument("--smoke-batch-size", type=int, default=16)
    parser.add_argument("--smoke-train-samples", type=int, default=96)
    parser.add_argument("--smoke-val-samples", type=int, default=32)
    parser.add_argument("--smoke-test-samples", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_one_run(args)
    print(result["run_root"])
    print(result["best_val_steer_rmse"])
    print(result["test_metrics"]["steer_rmse"])


if __name__ == "__main__":
    main()
