from .shared import *

# =========================
# Helpers
# =========================
def find_col(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None


def _series_to_float32(series, fill_value=0.0):
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float32, copy=True)
    bad = ~np.isfinite(arr)
    if np.any(bad):
        arr[bad] = float(fill_value)
    return arr


def _build_speed_source_report(df_v, cols):
    col_vkph = find_col(cols, ["zx1|v_km/h", "v_km/h"])
    col_vx = find_col(cols, ["zx|vx", "Vx", "vx", "Speed", "speed"])
    report = {
        "input_pipeline_version": INPUT_PIPELINE_VERSION,
        "col_v_km_h": col_vkph,
        "col_vx": col_vx,
        "selected_speed_source": None,
        "selected_speed_feature_name": None,
        "selected_speed_unit": None,
        "speed_source_warning": False,
        "speed_source_warning_reason": None,
        "n_ratio_rows": 0,
        "ratio_median": None,
        "ratio_mean": None,
        "ratio_min": None,
        "ratio_max": None,
    }

    if INPUT_PIPELINE_VERSION == "legacy_v1":
        if col_vx is None:
            return None, None, report
        speed_arr = _series_to_float32(df_v[col_vx], fill_value=0.0) / 3.6
        report["selected_speed_source"] = col_vx
        report["selected_speed_feature_name"] = col_vx
        report["selected_speed_unit"] = "legacy_divide_by_3.6"
        return speed_arr, col_vx, report

    vx_arr = _series_to_float32(df_v[col_vx], fill_value=np.nan) if col_vx is not None else None
    vkph_arr = _series_to_float32(df_v[col_vkph], fill_value=np.nan) if col_vkph is not None else None
    if vx_arr is not None and vkph_arr is not None:
        valid = np.isfinite(vx_arr) & np.isfinite(vkph_arr) & (vx_arr > 1e-6) & (vkph_arr > 1e-6)
        if np.any(valid):
            ratio = vkph_arr[valid] / np.maximum(vx_arr[valid], 1e-6)
            report["n_ratio_rows"] = int(valid.sum())
            report["ratio_median"] = float(np.median(ratio))
            report["ratio_mean"] = float(np.mean(ratio))
            report["ratio_min"] = float(np.min(ratio))
            report["ratio_max"] = float(np.max(ratio))
            if abs(report["ratio_median"] - 3.6) > 0.36:
                report["speed_source_warning"] = True
                report["speed_source_warning_reason"] = "ratio_median_not_close_to_3p6"

    if col_vkph is not None:
        speed_arr = _series_to_float32(df_v[col_vkph], fill_value=0.0) / 3.6
        report["selected_speed_source"] = col_vkph
        report["selected_speed_feature_name"] = "speed_mps"
        report["selected_speed_unit"] = "km_h_divide_by_3.6"
    elif col_vx is not None:
        speed_arr = _series_to_float32(df_v[col_vx], fill_value=0.0)
        report["selected_speed_source"] = col_vx
        report["selected_speed_feature_name"] = "speed_mps"
        report["selected_speed_unit"] = "assume_m_s"
    else:
        speed_arr = None

    if report["n_ratio_rows"] > 0 and report["n_ratio_rows"] < 0.90 * max(1, len(df_v)):
        report["speed_source_warning"] = True
        if report["speed_source_warning_reason"] is None:
            report["speed_source_warning_reason"] = "dual_speed_ratio_coverage_below_90pct"

    return speed_arr, report["selected_speed_feature_name"], report


def resolve_reversal_weight_blend(epoch: int | None = None, total_epochs: int | None = None):
    if REV_SAMPLE_WEIGHT_MODE != "hybrid":
        return None, None
    if REV_BRIDGE_MODE == "static" or epoch is None or total_epochs is None:
        return float(REV_HYBRID_WEAK_COEF), float(REV_HYBRID_STRONG_COEF)

    start_epoch = min(20, int(total_epochs))
    if epoch <= start_epoch:
        return 0.60, 0.40

    denom = max(1, int(total_epochs) - start_epoch)
    progress = min(1.0, max(0.0, float(epoch - start_epoch) / float(denom)))
    weak_coef = 0.60 * (1.0 - progress)
    strong_coef = 1.0 - weak_coef
    return float(weak_coef), float(strong_coef)


def summarize_input_qc_records(input_qc_records, feature_names):
    selected_source_counts = {}
    speed_source_counts = {}
    warning_files = []
    for rec in input_qc_records:
        for feat_name, raw_col in rec.get("selected_source_columns", {}).items():
            key = f"{feat_name}::{raw_col if raw_col is not None else 'MISSING'}"
            selected_source_counts[key] = selected_source_counts.get(key, 0) + 1
        speed_source = rec.get("speed_source_report", {}).get("selected_speed_source")
        speed_key = speed_source if speed_source is not None else "MISSING"
        speed_source_counts[speed_key] = speed_source_counts.get(speed_key, 0) + 1
        if rec.get("speed_source_report", {}).get("speed_source_warning"):
            warning_files.append(rec.get("vehicle_file"))

    return {
        "input_pipeline_version": INPUT_PIPELINE_VERSION,
        "feature_names": list(feature_names) if feature_names is not None else [],
        "use_pedals": bool(USE_PEDALS),
        "use_vy": bool(USE_VY),
        "use_vroll": bool(USE_VROLL),
        "use_mu": bool(USE_MU),
        "use_z": bool(USE_Z),
        "use_is_curve_ctx": bool(USE_IS_CURVE_CTX),
        "vehicle_file_count": int(len(input_qc_records)),
        "selected_source_column_counts": selected_source_counts,
        "speed_source_counts": speed_source_counts,
        "speed_warning_vehicle_files": warning_files,
        "records": input_qc_records,
    }

def make_strictly_increasing(x, eps=1e-6):
    x = np.asarray(x, dtype=np.float64).copy()
    for i in range(1, len(x)):
        if x[i] <= x[i - 1]:
            x[i] = x[i - 1] + eps
    return x


def unwrap_lane_center_signal(x, lane_width=3.5, jump_thr=1.8):
    """Unwrap a lane-centered lateral error signal by compensating sudden ~lane_width jumps.

    Some simulators/loggers switch the reference lane centerline during lane changes, causing
    lateraldistance to jump by approximately ±lane_width (or multiples). This function converts
    such piecewise signals into a continuous lateral position (relative to the initial lane).

    Args:
        x: 1D array-like, lane-centered lateral error (m)
        lane_width: lane width in meters (default 3.5)
        jump_thr: jump detection threshold in meters (default 1.8 ~ half lane width)

    Returns:
        unwrapped: 1D np.float32 array, continuous lateral position (relative)
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x.astype(np.float32)
    y = x.copy()
    offset = 0.0
    for i in range(1, y.size):
        if not np.isfinite(y[i]) or not np.isfinite(y[i - 1]):
            continue
        d = y[i] - y[i - 1]
        if d > jump_thr:
            k = int(np.round(d / lane_width))
            if k == 0:
                k = 1
            k = min(k, int(LANE_JUMP_MAX_MULTIPLES))
            offset -= k * lane_width
        elif d < -jump_thr:
            k = int(np.round((-d) / lane_width))
            if k == 0:
                k = 1
            k = min(k, int(LANE_JUMP_MAX_MULTIPLES))
            offset += k * lane_width
        y[i] = y[i] + offset
    return y.astype(np.float32)

def get_subject_id_from_path(vehicle_file):
    norm = os.path.normpath(vehicle_file)
    parts = norm.split(os.sep)
    return parts[-3] if len(parts) >= 3 else "unknown"


def load_protocol_split(protocol_config_path=PROTOCOL_CONFIG_PATH, frozen_split_path=FROZEN_SPLIT_PATH):
    protocol_config = load_json(protocol_config_path)
    frozen_split = load_json(frozen_split_path)
    expected = protocol_config.get("splits", {})
    if expected != frozen_split:
        raise ValueError(
            f"Protocol split mismatch between {protocol_config_path} and {frozen_split_path}"
        )
    return protocol_config, frozen_split


def build_subject_split_indices(sample_meta_df, split_subjects):
    split_indices = {}
    sample_subjects = set(sample_meta_df["subject_id"].astype(str).tolist())
    expected_subjects = {str(x) for vs in split_subjects.values() for x in vs}
    for split_name in ("train", "val", "test"):
        allowed = {str(x) for x in split_subjects.get(split_name, [])}
        mask = sample_meta_df["subject_id"].astype(str).isin(allowed)
        split_indices[split_name] = sample_meta_df.index[mask].to_numpy(dtype=np.int64)
        sample_subjects -= allowed
    if sample_subjects:
        raise ValueError(
            "Found subjects outside frozen protocol split. "
            f"sample_subjects={sorted(sample_subjects)}, expected_subjects={sorted(expected_subjects)}"
        )
    return split_indices


def subset_list(items, indices):
    return [items[int(i)] for i in indices]


def subset_array(items, indices):
    idx = np.asarray(indices, dtype=np.int64)
    return np.asarray(items)[idx]


def choose_smoke_indices(split_indices, max_total, rng):
    split_names = [name for name in ("train", "val", "test") if len(split_indices[name]) > 0]
    if len(split_names) != 3:
        raise ValueError("Smoke mode requires non-empty train/val/test splits")
    max_total = max(int(max_total), len(split_names))
    capacities = {name: int(len(split_indices[name])) for name in split_names}
    chosen_counts = {name: 1 for name in split_names}
    remaining = min(max_total, sum(capacities.values())) - len(split_names)
    available = {name: capacities[name] - 1 for name in split_names}
    while remaining > 0 and any(v > 0 for v in available.values()):
        total_avail = float(sum(max(v, 0) for v in available.values()))
        progressed = False
        for name in split_names:
            if remaining <= 0:
                break
            if available[name] <= 0:
                continue
            quota = max(1, int(round(remaining * (available[name] / total_avail)))) if total_avail > 0 else 1
            take = min(available[name], quota, remaining)
            if take <= 0:
                take = 1
            chosen_counts[name] += take
            available[name] -= take
            remaining -= take
            progressed = True
        if not progressed:
            break

    chosen = {}
    for name in split_names:
        perm = rng.permutation(split_indices[name])
        chosen[name] = np.sort(perm[:chosen_counts[name]].astype(np.int64))
    return chosen, chosen_counts


def compute_split_overlap(split_subjects):
    overlap = {}
    for left in ("train", "val", "test"):
        for right in ("train", "val", "test"):
            if left >= right:
                continue
            overlap[f"{left}_{right}"] = sorted(
                set(split_subjects.get(left, [])) & set(split_subjects.get(right, []))
            )
    return overlap


def export_split_audit(
    run_dir,
    sample_meta_df,
    split_indices,
    expected_subjects,
    protocol_config,
    smoke_mode,
    smoke_sampling_policy,
):
    protocol_summary = None
    if PROTOCOL_SPLIT_SUMMARY_PATH.exists():
        protocol_summary = pd.read_csv(PROTOCOL_SPLIT_SUMMARY_PATH)
        protocol_summary["split"] = protocol_summary["split"].astype(str)

    subject_rows = []
    sample_rows = []
    applied_subjects = {}
    for split_name, indices in split_indices.items():
        split_df = sample_meta_df.loc[np.asarray(indices, dtype=np.int64)].copy()
        split_df["split"] = split_name
        subject_counts = (
            split_df.groupby("subject_id").size().rename("sample_count").reset_index()
            if len(split_df) > 0 else pd.DataFrame(columns=["subject_id", "sample_count"])
        )
        applied_subjects[split_name] = sorted(split_df["subject_id"].astype(str).unique().tolist())
        for _, row in subject_counts.iterrows():
            subject_rows.append({
                "split": split_name,
                "subject_id": str(row["subject_id"]),
                "sample_count": int(row["sample_count"]),
            })
        curve_count = int(split_df["is_curve_applied"].fillna(0).astype(int).sum()) if len(split_df) else 0
        sample_row = {
            "split": split_name,
            "sample_count": int(len(split_df)),
            "subject_count": int(split_df["subject_id"].nunique()) if len(split_df) else 0,
            "vehicle_file_count": int(split_df["vehicle_file"].nunique()) if len(split_df) else 0,
            "curve_count": curve_count,
            "straight_count": int(len(split_df) - curve_count),
        }
        if protocol_summary is not None and split_name in set(protocol_summary["split"]):
            proto_row = protocol_summary.loc[protocol_summary["split"] == split_name].iloc[0]
            sample_row["protocol_sample_count"] = int(proto_row["sample_count"])
            sample_row["sample_count_diff_vs_protocol"] = int(sample_row["sample_count"] - int(proto_row["sample_count"]))
        sample_rows.append(sample_row)

    subject_counts_df = pd.DataFrame(subject_rows)
    sample_counts_df = pd.DataFrame(sample_rows)
    subject_counts_df.to_csv(str(run_dir / "split_subject_counts.csv"), index=False, encoding="utf-8-sig")
    sample_counts_df.to_csv(str(run_dir / "split_sample_counts.csv"), index=False, encoding="utf-8-sig")

    audit = {
        "protocol_config_path": str(PROTOCOL_CONFIG_PATH),
        "protocol_version": protocol_config.get("protocol_version"),
        "split_policy_expected": "subject-level fixed split",
        "split_policy_applied": "subject-level fixed split",
        "split_source": str(FROZEN_SPLIT_PATH),
        "smoke_mode": bool(smoke_mode),
        "smoke_sampling_policy": smoke_sampling_policy,
        "expected_subjects": {k: list(v) for k, v in expected_subjects.items()},
        "applied_subjects": applied_subjects,
        "subject_overlap": compute_split_overlap(applied_subjects),
        "sample_counts": sample_counts_df.to_dict(orient="records"),
        "protocol_split_summary_path": str(PROTOCOL_SPLIT_SUMMARY_PATH) if PROTOCOL_SPLIT_SUMMARY_PATH.exists() else None,
    }
    save_json(run_dir / "split_audit.json", audit)
    return audit, subject_counts_df, sample_counts_df

def load_vehicle_and_events(vehicle_file):
    event_file = vehicle_file.replace("\\vehicle\\", "\\event\\") \
        .replace("_vehicle_aligned_cleaned.csv",
                 "_vehicle_aligned_cleaned_events_v312.csv")
    if not os.path.exists(event_file):
        print(f"⚠ 事件文件不存在: {event_file}")
        return None, None
    return pd.read_csv(vehicle_file), pd.read_csv(event_file)

def load_driver_style_map(style_csv):
    if not os.path.exists(style_csv):
        print(f"⚠ 未找到驾驶风格结果文件: {style_csv} → 所有 style_id=0")
        return {}

    df = pd.read_excel(style_csv)
    cols = df.columns.tolist()

    subj_col = None
    for c in ["subject", "Subject", "subject_id", "被试", "被试编号"]:
        if c in cols:
            subj_col = c
            break
    if subj_col is None:
        raise ValueError(f"在 {style_csv} 中找不到 subject 列，请检查列名。")

    style_col = None
    for c in ["cluster_main_k2", "style_main", "style_3style",
              "cluster", "style_id", "cluster_id"]:
        if c in cols:
            style_col = c
            break
    if style_col is None:
        raise ValueError(f"在 {style_csv} 中找不到风格列。")

    style_vals = df[style_col].values
    if not np.issubdtype(style_vals.dtype, np.number):
        cats, idx = np.unique(style_vals, return_inverse=True)
        style_ids = idx
        print("🔧 风格列为字符串，已 factorize：")
        for i, cat in enumerate(cats):
            print(f"  style_id={i} ⇔ '{cat}'")
    else:
        style_ids = style_vals.astype(int)

    subj_vals = df[subj_col].astype(str).values
    style_map = {s: int(k) for s, k in zip(subj_vals, style_ids)}
    return style_map

def infer_physio_file(vehicle_file):
    """
    尽量稳妥地推断 physio 文件：
    ROOT/<subj>/physio/ 与 vehicle 同前缀的 physio CSV
    """
    subj_dir = os.path.dirname(os.path.dirname(vehicle_file))
    physio_dir = os.path.join(subj_dir, "physio")
    if not os.path.isdir(physio_dir):
        return None

    prefix = os.path.basename(vehicle_file).replace("_vehicle_aligned_cleaned.csv", "")
    cand = glob(os.path.join(physio_dir, prefix + "*physio*.csv"))
    if len(cand) > 0:
        return cand[0]
    # 兜底：physio 目录任意一个
    cand2 = glob(os.path.join(physio_dir, "*.csv"))
    return cand2[0] if len(cand2) else None

def infer_eeg_event_feature_file(vehicle_file):
    """
    EEG 事件特征 CSV 位于 ROOT/<subj>/eeg_clean/
    文件名包含同 recording 前缀，并以 _eeg_event_features_rollpeak_hist{EEG_HIST_SEC}s.csv 结尾
    """
    subj_dir = os.path.dirname(os.path.dirname(vehicle_file))
    eeg_dir = os.path.join(subj_dir, "eeg_clean")
    if not os.path.isdir(eeg_dir):
        return None

    prefix = os.path.basename(vehicle_file).replace("_vehicle_aligned_cleaned.csv", "")
    suffix = f"_eeg_event_features_rollpeak_hist{int(EEG_HIST_SEC)}s.csv"
    cand = glob(os.path.join(eeg_dir, prefix + "*" + suffix))
    return cand[0] if len(cand) else None


# =========================
# Teacher feature extraction (event-level)
# =========================
EEG_FEAT_KEYS = [
    "Occipital_ta_beta",
    "Frontal_ta_beta",
    "Temporal_ta_beta",
    "Occipital_alpha_abs",
    "Temporal_gamma_rel",
    "Occipital_gamma_rel",
    "Frontal_gamma_rel",
]

def build_eeg_feat_map(eeg_event_csv):
    """
    return dict: event_row_index(int) -> eeg_feat_vector(8,)
    8 dims: [alpha_asym, 7 others]
    """
    if eeg_event_csv is None or (not os.path.exists(eeg_event_csv)):
        return {}

    df = pd.read_csv(eeg_event_csv)
    if "event_row_index" not in df.columns:
        return {}

    # alpha asym column could be Frontal_alpha_asym_AF4AF3 / F8F7 / ...
    asym_cols = [c for c in df.columns if c.startswith("Frontal_alpha_asym_")]
    asym_col = asym_cols[0] if len(asym_cols) else None

    df = df.set_index("event_row_index")

    m = {}
    for k, row in df.iterrows():
        feats = []
        feats.append(float(row[asym_col]) if (asym_col is not None and asym_col in row) else np.nan)
        for name in EEG_FEAT_KEYS:
            feats.append(float(row[name]) if name in row else np.nan)
        m[int(k)] = np.array(feats, dtype=np.float32)  # (8,)
    return m

def safe_nanmean(values, default=np.nan):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float(default)
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return float(default)
    return float(valid.mean())

def extract_physio_window_means(df_p, peak_idx):
    """
    从 physio CSV 中取 [peak-WIN_LEN, peak) 的均值特征：
    HR, EDA_Tonic, EDA_Phasic, EMG_RMS
    """
    if df_p is None or len(df_p) < peak_idx or peak_idx - WIN_LEN < 0:
        return None

    cols = df_p.columns.tolist()

    col_hr = find_col(cols, ["HR", "HR_bpm", "hr", "hr_bpm"])
    col_t  = find_col(cols, ["EDA_Tonic", "eda_tonic", "Tonic"])
    col_p  = find_col(cols, ["EDA_Phasic", "eda_phasic", "Phasic"])
    col_emg= find_col(cols, ["EMG_RMS", "emg_rms", "EMG"])

    if col_hr is None or col_t is None or col_p is None or col_emg is None:
        return None

    seg = df_p.iloc[peak_idx - WIN_LEN: peak_idx]
    if seg.empty:
        return None

    hr  = safe_nanmean(seg[col_hr].to_numpy(dtype=np.float64))
    ton = safe_nanmean(seg[col_t].to_numpy(dtype=np.float64))
    pha = safe_nanmean(seg[col_p].to_numpy(dtype=np.float64))
    emg = safe_nanmean(seg[col_emg].to_numpy(dtype=np.float64))

    return np.array([hr, ton, pha, emg], dtype=np.float32)  # (4,)

def compute_teacher_state_old_ac(base_feat_z):
    """
    base_feat_z: (B, 12)  已经按 train-set 统计做过 z-score 的基础特征
    dims:
      0..3  : HR, tonic, phasic, emg
      4..11 : eeg [alpha_asym, occ_ta_beta, frontal_ta_beta, temporal_ta_beta,
                   occ_alpha_abs, temporal_gamma_rel, occ_gamma_rel, frontal_gamma_rel]
    Output:
      z_phys_raw (B,2): [A,C] legacy proxy state
    """
    hr     = base_feat_z[:, 0]
    tonic  = base_feat_z[:, 1]
    phasic = base_feat_z[:, 2]
    emg    = base_feat_z[:, 3]

    alpha_asym = base_feat_z[:, 4]
    occ_ta  = base_feat_z[:, 5]
    fr_ta   = base_feat_z[:, 6]
    te_ta   = base_feat_z[:, 7]
    occ_aabs= base_feat_z[:, 8]
    te_g    = base_feat_z[:, 9]
    oc_g    = base_feat_z[:,10]
    fr_g    = base_feat_z[:,11]

    gamma_mean = (te_g + oc_g + fr_g) / 3.0
    ta_mean = (occ_ta + fr_ta + te_ta) / 3.0

    A = (
        0.70 * hr +
        0.40 * tonic +
        0.80 * phasic +
        0.30 * gamma_mean +
        (-0.30) * occ_aabs +
        0.10 * alpha_asym
    )

    C = (
        0.70 * emg +
        0.50 * ta_mean
    )

    z = np.stack([A, C], axis=1).astype(np.float32)
    return z


def fit_pca_projection(train_x: np.ndarray, out_dim: int):
    """Fit PCA on train split only using numpy SVD; keep explicit valid-dim mapping."""
    x_full = np.asarray(train_x, dtype=np.float64)
    if x_full.ndim != 2:
        raise ValueError(f"Expected 2D train_x, got shape={x_full.shape}")

    valid_mask = np.isfinite(x_full).all(axis=0)
    if not np.any(valid_mask):
        raise ValueError("No valid feature dims available for PCA")

    x = x_full[:, valid_mask]
    mean = np.mean(x, axis=0, keepdims=True)
    xc = x - mean
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    basis = vt[:out_dim].T.astype(np.float32)
    return {
        "valid_mask": valid_mask.astype(bool),
        "mean": mean.reshape(-1).astype(np.float32),
        "basis": basis,
    }


def apply_pca_projection(x: np.ndarray, pca_params: dict):
    x_full = np.asarray(x, dtype=np.float32)
    valid_mask = np.asarray(pca_params["valid_mask"], dtype=bool)
    x = x_full[:, valid_mask]
    mean = pca_params["mean"].reshape(1, -1)
    basis = pca_params["basis"]
    return ((x - mean) @ basis).astype(np.float32)


def build_teacher_state(base_feat_z, mode: str, state_dim: int, fit_indices=None):
    if mode == "old_ac":
        z_raw = compute_teacher_state_old_ac(base_feat_z)
        meta = {
            "mode": mode,
            "raw_dim": int(z_raw.shape[1]),
            "component_names": ["A", "C"],
        }
        return z_raw.astype(np.float32), meta

    if mode == "pca_latent":
        fit_idx = np.asarray(fit_indices if fit_indices is not None else np.arange(len(base_feat_z)), dtype=np.int64)
        if fit_idx.size == 0:
            raise ValueError("fit_indices for teacher state PCA cannot be empty")
        fit_x = np.asarray(base_feat_z, dtype=np.float32)[fit_idx]
        fit_dim = int(min(state_dim, base_feat_z.shape[1], len(fit_idx)))
        fit_dim = max(fit_dim, 1)
        pca_params = fit_pca_projection(fit_x, fit_dim)
        z_raw = apply_pca_projection(base_feat_z, pca_params)
        meta = {
            "mode": mode,
            "raw_dim": int(z_raw.shape[1]),
            "component_names": [f"latent_{i}" for i in range(fit_dim)],
            "pca_valid_mask": pca_params["valid_mask"].astype(int).tolist(),
            "pca_mean": pca_params["mean"].tolist(),
            "pca_basis": pca_params["basis"].tolist(),
        }
        return z_raw.astype(np.float32), meta

    raise ValueError(f"Unsupported TEACHER_STATE_MODE: {mode}")


def make_state_column_names(prefix: str, dim: int, component_names=None):
    if component_names is not None and len(component_names) == dim:
        return [f"{prefix}_{name}" for name in component_names]
    return [f"{prefix}_d{i}" for i in range(dim)]


def summarize_state_vector(z_row, component_names=None):
    z = np.asarray(z_row, dtype=np.float32).reshape(-1)
    if component_names is None or len(component_names) != len(z):
        component_names = [f"d{i}" for i in range(len(z))]
    return " ".join([f"{name}={float(val):.2f}" for name, val in zip(component_names, z)])


def compute_teacher_state(base_feat_z):
    z_raw, _ = build_teacher_state(
        base_feat_z,
        mode=TEACHER_STATE_MODE,
        state_dim=TEACHER_STATE_DIM,
        fit_indices=np.arange(len(base_feat_z), dtype=np.int64),
    )
    return z_raw


# =========================
# Build samples (vehicle + teacher base feats)
# =========================
def build_samples_for_vehicle(vehicle_file, style_map):
    df_v, df_e = load_vehicle_and_events(vehicle_file)
    if df_v is None:
        return [], [], [], [], [], [], None, None

    cols = df_v.columns.tolist()
    n_rows = int(len(df_v))

    col_roll = find_col(cols, ["zx|roll", "roll", "Roll"])
    col_steer = find_col(cols, ["zx|SteeringWheel", "SteeringWheel", "steer"])
    col_yawrate = find_col(cols, ["vyaw", "zx|vyaw", "YawRate", "zx|YawRate", "yaw_rate"])
    col_z = find_col(cols, ["zx|z", "z", "Z"])
    col_ay = find_col(cols, ["zx|ay", "ay", "Ay", "lat_acc"])
    col_ax = find_col(cols, ["zx|ax", "ax", "Ax", "Long_acc"])
    lane_candidates = ["lateraldistance", "lateralDistance", "lateraldistance_start"]
    if INPUT_PIPELINE_VERSION != "legacy_v1":
        lane_candidates = ["zx1|lateraldistance", *lane_candidates]
    col_lane = find_col(cols, lane_candidates)
    col_curve = find_col(cols, ["zx1|lanecurvatureXY", "laneCurvature", "lanecurvature_start"])
    col_roadtype = find_col(cols, ["road_type_fixed", "road_type", "roadType_fixed"])
    col_refok = find_col(cols, ["ref_nn_ok", "ref_ok", "refnn_ok"])
    col_yaw = find_col(cols, ["zx|yaw", "yaw", "Yaw"])

    col_accel = find_col(cols, ["zx|AcceleratorPedal", "AcceleratorPedal", "accelerator_pedal"])
    col_brake = find_col(cols, ["zx|BrakePedal", "BrakePedal", "brake_pedal"])
    col_vy = find_col(cols, ["zx|vy", "vy", "Vy"])
    col_vroll = find_col(cols, ["zx|vroll", "vroll", "Vroll", "roll_rate"])
    col_mu = find_col(cols, ["zx1|mu", "mu", "Mu"])

    speed_arr, speed_feature_name, speed_source_report = _build_speed_source_report(df_v, cols)
    input_qc_record = {
        "vehicle_file": str(vehicle_file),
        "input_pipeline_version": INPUT_PIPELINE_VERSION,
        "row_count": n_rows,
        "event_row_count_raw": int(len(df_e)) if df_e is not None else 0,
        "selected_source_columns": {},
        "derived_columns_generated": {},
        "requested_optional_inputs": {
            "use_pedals": bool(USE_PEDALS),
            "use_vy": bool(USE_VY),
            "use_vroll": bool(USE_VROLL),
            "use_mu": bool(USE_MU),
            "use_z": bool(USE_Z),
            "use_is_curve_ctx": bool(USE_IS_CURVE_CTX),
        },
        "effective_optional_inputs": {},
        "optional_inputs_disabled_due_to_missing": [],
        "missing_required_columns_filled_zero": [],
        "speed_source_report": speed_source_report,
        "actual_feature_columns": [],
    }

    required_missing = []
    if col_roll is None:
        required_missing.append("roll")
    if col_steer is None:
        required_missing.append("steer")
    if col_ay is None:
        required_missing.append("ay")
    if col_yawrate is None:
        required_missing.append("yawrate")
    if col_curve is None:
        required_missing.append("lane_curvature")
    if speed_arr is None:
        required_missing.append("speed")
    if required_missing:
        input_qc_record["skip_reason"] = f"missing_required_columns:{','.join(required_missing)}"
        return [], [], [], [], [], [], None, input_qc_record

    dt = 1.0 / FS
    steer = steer_array_from_rad(_series_to_float32(df_v[col_steer], fill_value=0.0))
    steer_rate = np.gradient(steer, dt).astype(np.float32)

    feature_arrays = {}

    def add_feature(name, values, raw_col=None, derived=False):
        feature_arrays[name] = np.asarray(values, dtype=np.float32)
        input_qc_record["selected_source_columns"][name] = raw_col
        if derived:
            input_qc_record["derived_columns_generated"][name] = True

    def add_optional_feature(flag_key, feature_name, raw_col, missing_note):
        requested = bool(input_qc_record["requested_optional_inputs"][flag_key])
        if not requested:
            input_qc_record["effective_optional_inputs"][flag_key] = False
            return
        if raw_col is None:
            add_feature(feature_name, np.zeros((n_rows,), dtype=np.float32), raw_col=None)
            input_qc_record["effective_optional_inputs"][flag_key] = False
            input_qc_record["optional_inputs_disabled_due_to_missing"].append(missing_note)
            return
        add_feature(feature_name, _series_to_float32(df_v[raw_col], fill_value=0.0), raw_col=raw_col)
        input_qc_record["effective_optional_inputs"][flag_key] = True

    if INPUT_PIPELINE_VERSION == "legacy_v1":
        legacy_speed_source = speed_source_report.get("selected_speed_source")
        base_cols = [c for c in [
            col_roll,
            col_yawrate,
            col_ay,
            col_ax,
            legacy_speed_source,
            col_z,
            col_lane,
            col_curve,
            col_yaw,
            col_steer,
        ] if c is not None]
        df_feat = df_v[base_cols].copy()
        if legacy_speed_source is not None and speed_arr is not None:
            df_feat[legacy_speed_source] = speed_arr
        if col_ay is not None:
            df_feat["LTR_est"] = _series_to_float32(df_v[col_ay], fill_value=0.0) * float(LTR_COEFF)
        df_feat[col_steer] = steer
        df_feat["steer_rate"] = steer_rate
        input_qc_record["selected_source_columns"] = {
            feat_name: feat_name for feat_name in df_feat.columns.tolist() if feat_name not in {"LTR_est", "steer_rate"}
        }
        input_qc_record["selected_source_columns"]["LTR_est"] = col_ay
        input_qc_record["selected_source_columns"]["steer_rate"] = col_steer
        input_qc_record["derived_columns_generated"] = {
            "LTR_est": True,
            "steer_rate": True,
        }

        if col_lane is not None:
            lane_err = np.clip(
                _series_to_float32(df_v[col_lane], fill_value=0.0),
                -LANE_SIGNAL_ABS_CLIP_M,
                LANE_SIGNAL_ABS_CLIP_M,
            )
            lane_rate = np.clip(
                np.gradient(lane_err, dt).astype(np.float32),
                -LANE_RATE_ABS_CLIP_MPS,
                LANE_RATE_ABS_CLIP_MPS,
            )
            lane_acc = np.clip(
                np.gradient(lane_rate, dt).astype(np.float32),
                -LANE_ACC_ABS_CLIP_MPS2,
                LANE_ACC_ABS_CLIP_MPS2,
            )
            lane_unwrap = np.clip(
                unwrap_lane_center_signal(lane_err, lane_width=LANE_WIDTH_M, jump_thr=LANE_JUMP_THR_M).astype(np.float32),
                -LANE_SIGNAL_ABS_CLIP_M,
                LANE_SIGNAL_ABS_CLIP_M,
            )
            lane_unwrap_rate = np.clip(
                np.gradient(lane_unwrap, dt).astype(np.float32),
                -LANE_RATE_ABS_CLIP_MPS,
                LANE_RATE_ABS_CLIP_MPS,
            )
            lane_unwrap_acc = np.clip(
                np.gradient(lane_unwrap_rate, dt).astype(np.float32),
                -LANE_ACC_ABS_CLIP_MPS2,
                LANE_ACC_ABS_CLIP_MPS2,
            )
            df_feat["lane_rate"] = lane_rate
            df_feat["lane_acc"] = lane_acc
            df_feat["lane_unwrap"] = lane_unwrap.astype(np.float32)
            df_feat["lane_unwrap_rate"] = lane_unwrap_rate
            df_feat["lane_unwrap_acc"] = lane_unwrap_acc
            input_qc_record["derived_columns_generated"].update({
                "lane_rate": True,
                "lane_acc": True,
                "lane_unwrap": True,
                "lane_unwrap_rate": True,
                "lane_unwrap_acc": True,
            })
            input_qc_record["selected_source_columns"].update({
                "lane_rate": col_lane,
                "lane_acc": col_lane,
                "lane_unwrap": col_lane,
                "lane_unwrap_rate": col_lane,
                "lane_unwrap_acc": col_lane,
            })

        feature_cols = df_feat.columns.tolist()
        steer_feature_name = col_steer
        roll_feature_name = col_roll
        ay_feature_name = col_ay
        yawrate_feature_name = col_yawrate
        curve_feature_name = col_curve
        speed_feature_name = legacy_speed_source if legacy_speed_source is not None else speed_feature_name
    else:
        add_feature("roll", _series_to_float32(df_v[col_roll], fill_value=0.0), raw_col=col_roll)
        add_feature("yawrate", _series_to_float32(df_v[col_yawrate], fill_value=0.0), raw_col=col_yawrate)
        add_feature("ay", _series_to_float32(df_v[col_ay], fill_value=0.0), raw_col=col_ay)
        if col_ax is None:
            add_feature("ax", np.zeros((n_rows,), dtype=np.float32), raw_col=None)
            input_qc_record["missing_required_columns_filled_zero"].append("ax")
        else:
            add_feature("ax", _series_to_float32(df_v[col_ax], fill_value=0.0), raw_col=col_ax)
        add_feature("speed_mps", speed_arr, raw_col=speed_source_report.get("selected_speed_source"))

        if USE_Z:
            if col_z is None:
                add_feature("z", np.zeros((n_rows,), dtype=np.float32), raw_col=None)
                input_qc_record["effective_optional_inputs"]["use_z"] = False
                input_qc_record["optional_inputs_disabled_due_to_missing"].append("z")
            else:
                add_feature("z", _series_to_float32(df_v[col_z], fill_value=0.0), raw_col=col_z)
                input_qc_record["effective_optional_inputs"]["use_z"] = True
        else:
            input_qc_record["effective_optional_inputs"]["use_z"] = False

        if col_lane is None:
            lane_err = np.zeros((n_rows,), dtype=np.float32)
            input_qc_record["missing_required_columns_filled_zero"].append("lane_distance_m")
        else:
            lane_err = np.clip(
                _series_to_float32(df_v[col_lane], fill_value=0.0),
                -LANE_SIGNAL_ABS_CLIP_M,
                LANE_SIGNAL_ABS_CLIP_M,
            )
        add_feature("lane_distance_m", lane_err, raw_col=col_lane)
        add_feature("lane_curvature", _series_to_float32(df_v[col_curve], fill_value=0.0), raw_col=col_curve)

        if col_yaw is None:
            add_feature("yaw", np.zeros((n_rows,), dtype=np.float32), raw_col=None)
            input_qc_record["missing_required_columns_filled_zero"].append("yaw")
        else:
            add_feature("yaw", _series_to_float32(df_v[col_yaw], fill_value=0.0), raw_col=col_yaw)
        add_feature("steer", steer, raw_col=col_steer)
        add_feature("LTR_est", _series_to_float32(df_v[col_ay], fill_value=0.0) * float(LTR_COEFF), raw_col=col_ay, derived=True)
        add_feature("steer_rate", steer_rate, raw_col=col_steer, derived=True)

        lane_rate = np.clip(
            np.gradient(lane_err, dt).astype(np.float32),
            -LANE_RATE_ABS_CLIP_MPS,
            LANE_RATE_ABS_CLIP_MPS,
        )
        lane_acc = np.clip(
            np.gradient(lane_rate, dt).astype(np.float32),
            -LANE_ACC_ABS_CLIP_MPS2,
            LANE_ACC_ABS_CLIP_MPS2,
        )
        lane_unwrap = np.clip(
            unwrap_lane_center_signal(lane_err, lane_width=LANE_WIDTH_M, jump_thr=LANE_JUMP_THR_M).astype(np.float32),
            -LANE_SIGNAL_ABS_CLIP_M,
            LANE_SIGNAL_ABS_CLIP_M,
        )
        lane_unwrap_rate = np.clip(
            np.gradient(lane_unwrap, dt).astype(np.float32),
            -LANE_RATE_ABS_CLIP_MPS,
            LANE_RATE_ABS_CLIP_MPS,
        )
        lane_unwrap_acc = np.clip(
            np.gradient(lane_unwrap_rate, dt).astype(np.float32),
            -LANE_ACC_ABS_CLIP_MPS2,
            LANE_ACC_ABS_CLIP_MPS2,
        )
        add_feature("lane_rate", lane_rate, raw_col=col_lane, derived=True)
        add_feature("lane_acc", lane_acc, raw_col=col_lane, derived=True)
        add_feature("lane_unwrap", lane_unwrap, raw_col=col_lane, derived=True)
        add_feature("lane_unwrap_rate", lane_unwrap_rate, raw_col=col_lane, derived=True)
        add_feature("lane_unwrap_acc", lane_unwrap_acc, raw_col=col_lane, derived=True)

        add_optional_feature("use_pedals", "accelerator_pedal", col_accel, "accelerator_pedal")
        add_optional_feature("use_pedals", "brake_pedal", col_brake, "brake_pedal")
        add_optional_feature("use_vy", "vy", col_vy, "vy")
        add_optional_feature("use_vroll", "vroll", col_vroll, "vroll")
        add_optional_feature("use_mu", "mu", col_mu, "mu")

        feature_cols = list(feature_arrays.keys())
        df_feat = pd.DataFrame(feature_arrays, copy=False)
        steer_feature_name = "steer"
        roll_feature_name = "roll"
        ay_feature_name = "ay"
        yawrate_feature_name = "yawrate"
        curve_feature_name = "lane_curvature"
        speed_feature_name = "speed_mps"

    input_qc_record["actual_feature_columns"] = list(feature_cols)

    X_all = df_feat.to_numpy(dtype=np.float32)
    N = X_all.shape[0]

    steer_idx = feature_cols.index(steer_feature_name)
    roll_idx = feature_cols.index(roll_feature_name)
    ay_idx = feature_cols.index(ay_feature_name)
    yawrate_idx = feature_cols.index(yawrate_feature_name)
    steer_rate_idx = feature_cols.index("steer_rate")
    curve_idx = feature_cols.index(curve_feature_name)

    v_idx = feature_cols.index(speed_feature_name) if speed_feature_name in feature_cols else None
    if v_idx is not None:
        v_arr = np.nan_to_num(X_all[:, v_idx].astype(np.float32), nan=0.0)
        v_arr = np.clip(v_arr, 0.0, None)
        s_axis = np.zeros(N, dtype=np.float64)
        s_axis[1:] = np.cumsum(v_arr[:-1].astype(np.float64) * dt)
        s_axis = make_strictly_increasing(s_axis)
        curve_arr = np.nan_to_num(X_all[:, curve_idx].astype(np.float32), nan=0.0)
    else:
        v_arr = None
        s_axis = None
        curve_arr = None

    physio_file = infer_physio_file(vehicle_file)
    df_p = pd.read_csv(physio_file) if (physio_file is not None and os.path.exists(physio_file)) else None

    eeg_event_csv = infer_eeg_event_feature_file(vehicle_file)
    eeg_map = build_eeg_feat_map(eeg_event_csv)

    subject_id = get_subject_id_from_path(vehicle_file)
    style_id = style_map.get(subject_id, 0)

    X_list, y_list, curve_list, ctx_list, base_feat_list, meta_list = [], [], [], [], [], []

    df_e = df_e[df_e["event_level"].isin(STRONG_LABELS)]
    input_qc_record["event_row_count_filtered"] = int(len(df_e))
    if len(df_e) == 0:
        input_qc_record["skip_reason"] = "no_strong_events"
        return [], [], [], [], [], [], None, input_qc_record

    for ev_idx, ev in df_e.iterrows():
        t0 = float(ev["start_s"])
        t1 = float(ev["end_s"])
        i0 = int(t0 * FS)
        i1 = int(t1 * FS)
        if i0 < 0 or i1 > N or (i1 - i0) < 10:
            continue

        curve_seg = X_all[i0:i1, curve_idx]
        curve_seg_mean = float(np.nanmean(np.abs(curve_seg))) if curve_seg.size else 0.0

        is_curve = None
        if (col_roadtype is not None) and (col_refok is not None):
            ok_seg = df_v[col_refok].to_numpy(dtype=np.float32, copy=False)[i0:i1]
            ok_ratio = float(np.nanmean(ok_seg > 0.5)) if ok_seg.size else 0.0
            if ok_ratio >= ROAD_OK_RATIO_THR:
                rt_seg = df_v[col_roadtype].to_numpy(copy=False)[i0:i1]
                if rt_seg.dtype.kind in ("i", "u", "f"):
                    is_curve = (float(np.nanmean(rt_seg)) >= 0.5)
                else:
                    rt_low = np.char.lower(rt_seg.astype(str))
                    is_curve = (float(np.mean(rt_low == "curve")) >= 0.5)

        if is_curve is None:
            is_curve = (curve_seg_mean > CURVE_THR_FOR_ANCHOR)

        if is_curve:
            roll_seg = X_all[i0:i1, roll_idx]
            if roll_seg.size == 0:
                continue
            peak_rel = int(np.argmax(np.abs(roll_seg)))
            peak_idx = i0 + peak_rel
        else:
            sr_seg = X_all[i0:i1, steer_rate_idx]
            if sr_seg.size == 0:
                continue
            abs_sr = np.abs(sr_seg)
            max_abs = float(np.nanmax(abs_sr))
            if (not np.isfinite(max_abs)) or max_abs < 1e-6:
                roll_seg = X_all[i0:i1, roll_idx]
                if roll_seg.size == 0:
                    continue
                peak_rel = int(np.argmax(np.abs(roll_seg)))
                peak_idx = i0 + peak_rel
            else:
                thr = STEER_RATE_PEAK_FRAC * max_abs
                cand = np.where(abs_sr >= thr)[0]
                peak_rel = int(cand[0]) if cand.size else int(np.argmax(abs_sr))
                peak_idx = i0 + peak_rel

        if peak_idx - WIN_LEN < 0 or peak_idx + FUTURE_LEN >= N:
            continue

        phys4 = extract_physio_window_means(df_p, peak_idx)
        eeg8 = eeg_map.get(int(ev_idx), None)
        if phys4 is None:
            phys4 = np.full((4,), np.nan, dtype=np.float32)
        if eeg8 is None:
            eeg8 = np.full((8,), np.nan, dtype=np.float32)

        base12 = np.concatenate([phys4, eeg8], axis=0).astype(np.float32)
        x_win = X_all[peak_idx - WIN_LEN: peak_idx]

        steer_anchor = float(X_all[peak_idx, steer_idx])
        y_steer = X_all[peak_idx + 1: peak_idx + 1 + FUTURE_LEN, steer_idx] - steer_anchor
        y_yaw = X_all[peak_idx + 1: peak_idx + 1 + FUTURE_LEN, yawrate_idx]
        y_ay = X_all[peak_idx + 1: peak_idx + 1 + FUTURE_LEN, ay_idx]
        y_seq = np.stack([y_steer, y_yaw, y_ay], axis=-1)

        if v_arr is not None and s_axis is not None and curve_arr is not None:
            v0 = float(v_arr[peak_idx])
            s0 = float(s_axis[peak_idx])
            t_grid = (np.arange(1, FUTURE_LEN + 1, dtype=np.float64) * dt)
            s_query = np.clip(s0 + v0 * t_grid, s_axis[0], s_axis[-1])
            curve_future = np.interp(s_query, s_axis, curve_arr).astype(np.float32)
        else:
            curve_future = X_all[peak_idx + 1: peak_idx + 1 + FUTURE_LEN, curve_idx].astype(np.float32)

        ctx_values = [
            float(X_all[peak_idx, steer_idx]),
            float(X_all[peak_idx, steer_rate_idx]),
            float(X_all[peak_idx, ay_idx]),
            float(X_all[peak_idx, yawrate_idx]),
            float(style_id),
        ]
        if USE_IS_CURVE_CTX:
            ctx_values.append(float(bool(is_curve)))
        ctx = np.asarray(ctx_values, dtype=np.float32)

        X_list.append(x_win.astype(np.float32))
        y_list.append(y_seq.astype(np.float32))
        curve_list.append(curve_future.astype(np.float32))
        ctx_list.append(ctx)
        base_feat_list.append(base12)
        meta_list.append({
            "sample_key": f"{subject_id}::{os.path.basename(vehicle_file)}::{int(ev_idx)}::maintained_anchor",
            "subject_id": str(subject_id),
            "vehicle_file": str(vehicle_file),
            "event_idx": int(ev_idx),
            "event_level": str(ev.get("event_level", "")),
            "event_start_s": float(t0),
            "event_end_s": float(t1),
            "anchor_idx": int(peak_idx),
            "anchor_source_applied": "roll_peak" if is_curve else "steer_rate_peak80_first",
            "maintained_anchor_policy": "curve->roll_peak; straight->steer_rate_peak80_first",
            "is_curve_applied": int(bool(is_curve)),
            "curve_score_event_mean_abs": float(curve_seg_mean),
        })

    input_qc_record["sample_count"] = int(len(X_list))
    if len(X_list) == 0:
        input_qc_record["skip_reason"] = "no_valid_windows_after_anchor_filter"
        return [], [], [], [], [], [], None, input_qc_record
    return X_list, y_list, curve_list, ctx_list, base_feat_list, meta_list, feature_cols, input_qc_record


def build_all_samples(style_map):
    pattern = os.path.join(ROOT, "*", "vehicle", "*_vehicle_aligned_cleaned.csv")
    vehicle_files = sorted(glob(pattern))

    X_pool, y_pool, curve_pool, ctx_pool, base_pool, meta_pool = [], [], [], [], [], []
    feature_names = None
    input_qc_records = []

    print("🔍 遍历车辆文件构造事件样本 + teacher base feats ...")
    total = 0
    for vf in vehicle_files:
        X_list, y_list, curve_list, ctx_list, base_list, meta_list, feat_cols, input_qc_record = build_samples_for_vehicle(vf, style_map)
        if input_qc_record is not None:
            input_qc_records.append(input_qc_record)
        if feat_cols is None or len(X_list) == 0:
            continue

        if feature_names is None:
            feature_names = feat_cols
        elif feat_cols != feature_names:
            print("⚠ 特征列顺序不一致，跳过:", vf)
            continue

        X_pool.extend(X_list)
        y_pool.extend(y_list)
        curve_pool.extend(curve_list)
        ctx_pool.extend(ctx_list)
        base_pool.extend(base_list)
        meta_pool.extend(meta_list)
        total += len(X_list)

    print(f"✅ 共收集到 {total} 个事件样本\n")
    input_qc_summary = summarize_input_qc_records(input_qc_records, feature_names)
    sample_meta_df = pd.DataFrame(meta_pool)
    return X_pool, y_pool, curve_pool, ctx_pool, base_pool, sample_meta_df, feature_names, input_qc_summary
