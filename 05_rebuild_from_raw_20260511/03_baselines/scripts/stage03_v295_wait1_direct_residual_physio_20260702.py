from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


"""
v295: waiting-policy direct residual correction with post-response physiology.

核心问题：
    v293 证明 observation 后 0-1/0-3s 生理信号能识别 bad_top10 风险；
    v294 证明用这些生理信号去“选一个已有候选轨迹”不稳定；
    因此 v295 不再做候选轨迹选择，而是直接学习 rolling baseline 的残差。

评估口径：
    只做 wait=1s 的第一版。baseline 是 v249 在 delay_ms=1000 的 rolling 预测；
    可用输入包括：
      1) v249 baseline 曲线形态；
      2) 原锚点后 0-1s 已经实际发生的车辆响应前缀；
      3) v293 post0_1 生理窗口特征；
      4) subject one-hot 作为驾驶员风格的轻量代理。
    所有模型和特征筛选只用 train；是否启用 residual correction 的风险阈值只用 val；
    test 只做一次性报告。
"""


SEED = 20260702
WAIT_MS = 1000
WAIT_S = WAIT_MS / 1000.0
TOP_PHYSIO_PER_WINDOW = 54
TOP_PRE_FEATURES = 54
SHRINKAGES = [0.25, 0.50, 0.75, 1.00]

BASELINES = Path(__file__).resolve().parents[1]
REBUILD = BASELINES.parent
OUT = BASELINES / "v295_wait1_direct_residual_physio_20260702"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
ZIP_PATH = BASELINES / "v295_wait1_direct_residual_physio_20260702_pack.zip"

V249_NPZ = BASELINES / "v249_shape_aware_curve_model_20260630" / "v249_shape_aware_predictions.npz"
V293_FEATURES = (
    BASELINES
    / "v293_physio_response_visibility_latency_audit_20260702"
    / "tables"
    / "v293_prepost_physio_visibility_features.csv"
)
V293_SCREEN = (
    BASELINES
    / "v293_physio_response_visibility_latency_audit_20260702"
    / "tables"
    / "v293_train_only_feature_screen.csv"
)
V293_GUARDRAIL = (
    BASELINES
    / "v293_physio_response_visibility_latency_audit_20260702"
    / "logs"
    / "guardrail_check.json"
)
THIS_SCRIPT = Path(__file__).resolve()


GROUP_FLAGS: List[Tuple[str, str | None]] = [
    ("all", None),
    ("bad_top10", "bad_top10"),
    ("bad_top10_vehicle_ambiguous", "bad_top10_vehicle_ambiguous"),
    ("vehicle_ambiguous", "vehicle_ambiguous"),
    ("candidate_pool_gain_gt_005", "candidate_pool_gain_gt_005"),
    ("non_bad_top10", "__non_bad_top10__"),
]


def ensure_dirs() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    for p in [TABLES, FIGURES, REPORTS, LOGS]:
        p.mkdir(parents=True, exist_ok=True)


def file_sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(block_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_json(obj: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def as_float_array(df: pd.DataFrame) -> np.ndarray:
    clean = df.replace([np.inf, -np.inf], np.nan)
    return clean.to_numpy(dtype=float)


def safe_mean(x: Iterable[float]) -> float:
    arr = np.asarray(list(x), dtype=float)
    if arr.size == 0 or np.all(~np.isfinite(arr)):
        return math.nan
    return float(np.nanmean(arr))


def rmse_rows(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=1))


def sanitize_col(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", str(text)).strip("_")[:80]


def curve_feature_frame(prefix: str, arr: np.ndarray, x_grid: np.ndarray | None = None) -> pd.DataFrame:
    rows: Dict[str, np.ndarray] = {}
    for j in range(arr.shape[1]):
        rows[f"{prefix}_p{j:02d}"] = arr[:, j]
    rows[f"{prefix}_mean"] = np.nanmean(arr, axis=1)
    rows[f"{prefix}_std"] = np.nanstd(arr, axis=1)
    rows[f"{prefix}_min"] = np.nanmin(arr, axis=1)
    rows[f"{prefix}_max"] = np.nanmax(arr, axis=1)
    rows[f"{prefix}_range"] = np.nanmax(arr, axis=1) - np.nanmin(arr, axis=1)
    rows[f"{prefix}_first"] = arr[:, 0]
    rows[f"{prefix}_last"] = arr[:, -1]
    rows[f"{prefix}_last_minus_first"] = arr[:, -1] - arr[:, 0]
    rows[f"{prefix}_peak_abs"] = np.nanmax(np.abs(arr), axis=1)
    rows[f"{prefix}_line_length"] = np.nansum(np.abs(np.diff(arr, axis=1)), axis=1)
    if x_grid is not None and len(x_grid) == arr.shape[1]:
        denom = float(x_grid[-1] - x_grid[0]) if len(x_grid) > 1 else 1.0
        rows[f"{prefix}_slope"] = (arr[:, -1] - arr[:, 0]) / max(denom, 1e-6)
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, cols: List[str], max_rows: int = 20) -> str:
    if df.empty:
        return "_empty_"
    cols = [c for c in cols if c in df.columns]
    view = df.loc[:, cols].head(max_rows).copy()
    for c in view.columns:
        if pd.api.types.is_float_dtype(view[c]):
            view[c] = view[c].map(lambda x: "" if pd.isna(x) else f"{x:.6f}")
    return view.to_markdown(index=False)


@dataclass
class CurveData:
    df: pd.DataFrame
    y_eval: np.ndarray
    baseline_eval: np.ndarray
    observed_prefix: np.ndarray
    eval_grid_s: np.ndarray
    observed_grid_s: np.ndarray
    target_cols: np.ndarray


def load_curve_data() -> CurveData:
    features = pd.read_csv(V293_FEATURES)
    with np.load(V249_NPZ, allow_pickle=False) as z:
        event_uid = z["event_uid"].astype(str)
        split = z["split"].astype(str)
        delay_ms = z["delay_ms"].astype(int)
        y_true = z["y_true_steering_delta"].astype(float)
        baseline = z["pred_v249_best_shape_steering_delta"].astype(float)
        grid = z["future_grid_s"].astype(float)
        valid = z["original_remaining_valid"].astype(bool)

    wait_idx = np.where(delay_ms == WAIT_MS)[0]
    zero_idx = np.where(delay_ms == 0)[0]
    zero_map = {event_uid[i]: i for i in zero_idx}
    if len(wait_idx) == 0:
        raise RuntimeError(f"Cannot find delay_ms={WAIT_MS} in {V249_NPZ}")

    df = pd.DataFrame(
        {
            "event_uid": event_uid[wait_idx],
            "split_npz": split[wait_idx],
            "delay_ms": delay_ms[wait_idx],
        }
    )
    df = df.merge(features, on="event_uid", how="left", validate="one_to_one")
    if df["split"].isna().any():
        missing = int(df["split"].isna().sum())
        raise RuntimeError(f"v293 features missing for {missing} wait samples")
    if not df["split"].astype(str).eq(df["split_npz"].astype(str)).all():
        bad = int((~df["split"].astype(str).eq(df["split_npz"].astype(str))).sum())
        raise RuntimeError(f"split mismatch between v293 and v249 for {bad} samples")

    # 对 delay=1000，original_remaining_valid 是固定 11 个点；只训练和评估这些共同有效点。
    wait_valid = valid[wait_idx]
    common_cols = np.where(wait_valid.all(axis=0))[0]
    if common_cols.size == 0:
        common_cols = np.where(wait_valid.any(axis=0))[0]
    if common_cols.size == 0:
        raise RuntimeError("No valid horizon point for wait=1s evaluation")

    observed_cols = np.where(grid <= WAIT_S + 1e-9)[0]
    observed_rows = []
    for uid in df["event_uid"].astype(str):
        if uid not in zero_map:
            raise RuntimeError(f"Cannot find delay=0 row for {uid}")
        observed_rows.append(y_true[zero_map[uid], observed_cols])

    return CurveData(
        df=df,
        y_eval=y_true[wait_idx][:, common_cols],
        baseline_eval=baseline[wait_idx][:, common_cols],
        observed_prefix=np.vstack(observed_rows),
        eval_grid_s=grid[common_cols],
        observed_grid_s=grid[observed_cols],
        target_cols=common_cols,
    )


def select_physio_columns(screen: pd.DataFrame) -> Tuple[List[str], List[str]]:
    post01 = (
        screen[screen["window"].astype(str).eq("post0_1")]
        .sort_values("max_abs_corr_train", ascending=False)
        .head(TOP_PHYSIO_PER_WINDOW)["feature"]
        .astype(str)
        .tolist()
    )
    pre = (
        screen[screen["phase"].astype(str).eq("pre")]
        .sort_values("max_abs_corr_train", ascending=False)
        .head(TOP_PRE_FEATURES)["feature"]
        .astype(str)
        .tolist()
    )
    return post01, pre


def subject_frame(df: pd.DataFrame, train_mask: np.ndarray) -> pd.DataFrame:
    train_subjects = sorted(df.loc[train_mask, "subject"].fillna("unknown").astype(str).unique().tolist())
    out = {}
    subjects = df["subject"].fillna("unknown").astype(str)
    for s in train_subjects:
        out[f"subject_{sanitize_col(s)}"] = subjects.eq(s).astype(float).to_numpy()
    out["subject_unseen_in_train"] = (~subjects.isin(train_subjects)).astype(float).to_numpy()
    return pd.DataFrame(out)


def build_feature_blocks(data: CurveData, post01_cols: List[str], pre_cols: List[str]) -> Dict[str, pd.DataFrame]:
    df = data.df.reset_index(drop=True)
    train_mask = df["split"].astype(str).eq("train").to_numpy()
    base_curve = curve_feature_frame("base", data.baseline_eval, data.eval_grid_s)
    obs_curve = curve_feature_frame("obs0_1", data.observed_prefix, data.observed_grid_s)
    phys_cols = [c for c in (post01_cols + pre_cols) if c in df.columns]
    phys = df[phys_cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    subject = subject_frame(df, train_mask).reset_index(drop=True)

    blocks = {
        "base_curve_only": base_curve,
        "base_plus_observed_vehicle_prefix": pd.concat([base_curve, obs_curve], axis=1),
        "base_plus_post01_physio": pd.concat([base_curve, phys], axis=1),
        "base_plus_post01_physio_subject": pd.concat([base_curve, phys, subject], axis=1),
        "base_plus_vehicle_prefix_post01_physio_subject": pd.concat([base_curve, obs_curve, phys, subject], axis=1),
        "vehicle_prefix_post01_physio_subject": pd.concat([obs_curve, phys, subject], axis=1),
    }
    return {k: v.loc[:, ~v.columns.duplicated()].copy() for k, v in blocks.items()}


def build_risk_blocks(data: CurveData, post01_cols: List[str], pre_cols: List[str]) -> Dict[str, pd.DataFrame]:
    df = data.df.reset_index(drop=True)
    train_mask = df["split"].astype(str).eq("train").to_numpy()
    base_curve = curve_feature_frame("base", data.baseline_eval, data.eval_grid_s)
    obs_curve = curve_feature_frame("obs0_1", data.observed_prefix, data.observed_grid_s)
    phys_cols = [c for c in (post01_cols + pre_cols) if c in df.columns]
    phys = df[phys_cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    subject = subject_frame(df, train_mask).reset_index(drop=True)
    blocks = {
        "post01_physio_subject": pd.concat([phys, subject], axis=1),
        "base_post01_physio_subject": pd.concat([base_curve, phys, subject], axis=1),
        "vehicle_prefix_post01_physio_subject": pd.concat([obs_curve, phys, subject], axis=1),
    }
    return {k: v.loc[:, ~v.columns.duplicated()].copy() for k, v in blocks.items()}


def fit_residual_predictions(data: CurveData, blocks: Dict[str, pd.DataFrame]) -> List[Dict[str, object]]:
    df = data.df
    train = df["split"].astype(str).eq("train").to_numpy()
    y_res = data.y_eval - data.baseline_eval
    preds: List[Dict[str, object]] = []

    models = {
        "ridge_a10": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0)),
        "ridge_a100": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=100.0)),
        "extra_trees_d4": make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesRegressor(
                n_estimators=500,
                max_depth=4,
                min_samples_leaf=10,
                random_state=SEED,
                n_jobs=-1,
            ),
        ),
        "extra_trees_d6": make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesRegressor(
                n_estimators=500,
                max_depth=6,
                min_samples_leaf=8,
                random_state=SEED + 1,
                n_jobs=-1,
            ),
        ),
    }
    for block_name, frame in blocks.items():
        x = as_float_array(frame)
        for model_name, model in models.items():
            print(f"[v295] fit residual block={block_name} model={model_name} feature_n={x.shape[1]}")
            model.fit(x[train], y_res[train])
            pred = model.predict(x)
            preds.append(
                {
                    "residual_tag": f"{block_name}__{model_name}",
                    "feature_block": block_name,
                    "model_name": model_name,
                    "feature_n": int(x.shape[1]),
                    "uses_physio": "physio" in block_name,
                    "uses_vehicle_prefix": "vehicle_prefix" in block_name,
                    "uses_subject": "subject" in block_name,
                    "pred_residual": np.asarray(pred, dtype=float),
                }
            )

    # 风格先验：只从 train 中估计 subject/global 残差均值，作为驾驶员风格代理的低方差基线。
    global_res = np.nanmean(y_res[train], axis=0)
    subjects = df["subject"].fillna("unknown").astype(str)
    subject_res: Dict[str, np.ndarray] = {}
    for s in sorted(subjects[train].unique().tolist()):
        m = train & subjects.eq(s).to_numpy()
        if int(m.sum()) >= 3:
            subject_res[s] = np.nanmean(y_res[m], axis=0)
    prior = np.vstack([subject_res.get(s, global_res) for s in subjects])
    preds.append(
        {
            "residual_tag": "subject_style_prior_mean_residual",
            "feature_block": "subject_style_prior",
            "model_name": "train_subject_mean_residual",
            "feature_n": int(len(subject_res)),
            "uses_physio": False,
            "uses_vehicle_prefix": False,
            "uses_subject": True,
            "pred_residual": prior,
        }
    )
    return preds


def fit_risk_scores(data: CurveData, blocks: Dict[str, pd.DataFrame]) -> Tuple[List[Dict[str, object]], pd.DataFrame]:
    df = data.df
    train = df["split"].astype(str).eq("train").to_numpy()
    val = df["split"].astype(str).eq("val").to_numpy()
    test = df["split"].astype(str).eq("test").to_numpy()
    y = df["bad_top10"].fillna(0).astype(int).to_numpy()
    scores: List[Dict[str, object]] = []
    audit_rows: List[Dict[str, object]] = []
    models = {
        "logreg_balanced_c025": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            # 默认 lbfgs 在当前 Windows/Scipy 组合里偶发原生崩溃；liblinear 对小样本二分类更稳。
            LogisticRegression(
                C=0.25,
                max_iter=2000,
                class_weight="balanced",
                random_state=SEED,
                solver="liblinear",
            ),
        ),
        "extra_trees_cls_d3": make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                n_estimators=500,
                max_depth=3,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=SEED,
                n_jobs=-1,
            ),
        ),
    }
    for block_name, frame in blocks.items():
        x = as_float_array(frame)
        for model_name, model in models.items():
            print(f"[v295] fit risk block={block_name} model={model_name} feature_n={x.shape[1]}")
            model.fit(x[train], y[train])
            score = model.predict_proba(x)[:, 1]
            row: Dict[str, object] = {
                "risk_tag": f"{block_name}__{model_name}",
                "risk_block": block_name,
                "risk_model": model_name,
                "feature_n": int(x.shape[1]),
                "uses_physio": True,
            }
            for split_name, mask in [("train", train), ("val", val), ("test", test)]:
                if len(np.unique(y[mask])) == 2:
                    auc = float(roc_auc_score(y[mask], score[mask]))
                else:
                    auc = math.nan
                row[f"{split_name}_auc"] = auc
                row[f"{split_name}_positive_rate"] = float(np.mean(y[mask])) if int(mask.sum()) else math.nan
                row[f"{split_name}_n"] = int(mask.sum())
            audit_rows.append(row)
            scores.append({**row, "score": score})

    # always gate 用于直接应用 residual；no_override 用于 fallback。
    scores.append(
        {
            "risk_tag": "always_apply",
            "risk_block": "constant",
            "risk_model": "always",
            "feature_n": 0,
            "uses_physio": False,
            "train_auc": math.nan,
            "val_auc": math.nan,
            "test_auc": math.nan,
            "score": np.ones(len(df), dtype=float),
        }
    )
    scores.append(
        {
            "risk_tag": "no_override",
            "risk_block": "constant",
            "risk_model": "never",
            "feature_n": 0,
            "uses_physio": False,
            "train_auc": math.nan,
            "val_auc": math.nan,
            "test_auc": math.nan,
            "score": np.zeros(len(df), dtype=float),
        }
    )
    return scores, pd.DataFrame(audit_rows)


def group_mask(df: pd.DataFrame, split_mask: np.ndarray, flag: str | None) -> np.ndarray:
    if flag is None:
        return split_mask.copy()
    if flag == "__non_bad_top10__":
        return split_mask & (~df["bad_top10"].fillna(0).astype(bool).to_numpy())
    if flag not in df.columns:
        return np.zeros(len(df), dtype=bool)
    return split_mask & df[flag].fillna(0).astype(bool).to_numpy()


def summarize_prediction(
    df: pd.DataFrame,
    y_eval: np.ndarray,
    baseline_eval: np.ndarray,
    corrected_eval: np.ndarray,
    override: np.ndarray,
) -> Dict[str, object]:
    split_masks = {
        "train": df["split"].astype(str).eq("train").to_numpy(),
        "val": df["split"].astype(str).eq("val").to_numpy(),
        "test": df["split"].astype(str).eq("test").to_numpy(),
    }
    base_rmse = rmse_rows(y_eval, baseline_eval)
    corr_rmse = rmse_rows(y_eval, corrected_eval)
    row: Dict[str, object] = {}
    for split_name, split_mask in split_masks.items():
        for group_name, flag in GROUP_FLAGS:
            m = group_mask(df, split_mask, flag)
            row[f"{split_name}_{group_name}_n"] = int(m.sum())
            if int(m.sum()) == 0:
                row[f"{split_name}_{group_name}_baseline_rmse_mean"] = math.nan
                row[f"{split_name}_{group_name}_corrected_rmse_mean"] = math.nan
                row[f"{split_name}_{group_name}_delta_vs_baseline_mean"] = math.nan
                row[f"{split_name}_{group_name}_override_rate"] = math.nan
                continue
            row[f"{split_name}_{group_name}_baseline_rmse_mean"] = float(np.mean(base_rmse[m]))
            row[f"{split_name}_{group_name}_corrected_rmse_mean"] = float(np.mean(corr_rmse[m]))
            row[f"{split_name}_{group_name}_delta_vs_baseline_mean"] = float(np.mean(corr_rmse[m] - base_rmse[m]))
            row[f"{split_name}_{group_name}_override_rate"] = float(np.mean(override[m]))
    return row


def threshold_candidates(score: np.ndarray, val_mask: np.ndarray, risk_tag: str) -> List[float]:
    if risk_tag == "always_apply":
        return [-math.inf]
    if risk_tag == "no_override":
        return [math.inf]
    vals = score[val_mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return [math.inf]
    qs = np.linspace(0.50, 0.98, 17)
    thresholds = [float(x) for x in np.unique(np.nanquantile(vals, qs))]
    thresholds.append(math.inf)
    return sorted(set(thresholds), key=lambda x: (math.isinf(x), x))


def evaluate_configs(
    data: CurveData,
    residual_preds: List[Dict[str, object]],
    risk_scores: List[Dict[str, object]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = data.df.reset_index(drop=True)
    val = df["split"].astype(str).eq("val").to_numpy()
    rows: List[Dict[str, object]] = []
    selected_rows: List[pd.DataFrame] = []
    base_rmse = rmse_rows(data.y_eval, data.baseline_eval)

    for residual in residual_preds:
        pred_residual = np.asarray(residual["pred_residual"], dtype=float)
        for shrink in SHRINKAGES:
            for risk in risk_scores:
                score = np.asarray(risk["score"], dtype=float)
                for threshold in threshold_candidates(score, val, str(risk["risk_tag"])):
                    if math.isinf(threshold) and threshold > 0:
                        override = np.zeros(len(df), dtype=bool)
                    elif math.isinf(threshold) and threshold < 0:
                        override = np.ones(len(df), dtype=bool)
                    else:
                        override = score >= threshold
                    corrected = data.baseline_eval + shrink * pred_residual * override[:, None]
                    corr_rmse = rmse_rows(data.y_eval, corrected)
                    row: Dict[str, object] = {
                        "selector_tag": f"{residual['residual_tag']}__shrink{shrink:.2f}__gate_{risk['risk_tag']}__thr_{threshold:.6g}",
                        "residual_tag": residual["residual_tag"],
                        "feature_block": residual["feature_block"],
                        "model_name": residual["model_name"],
                        "residual_feature_n": residual["feature_n"],
                        "uses_physio_in_residual": bool(residual["uses_physio"]),
                        "uses_vehicle_prefix": bool(residual["uses_vehicle_prefix"]),
                        "uses_subject": bool(residual["uses_subject"]),
                        "shrinkage": float(shrink),
                        "risk_tag": risk["risk_tag"],
                        "risk_block": risk["risk_block"],
                        "risk_model": risk["risk_model"],
                        "risk_feature_n": risk["feature_n"],
                        "uses_physio_in_gate": bool(risk["uses_physio"]),
                        "risk_val_auc": risk.get("val_auc", math.nan),
                        "risk_test_auc": risk.get("test_auc", math.nan),
                        "threshold": float(threshold),
                        "test_used_for_model_or_threshold": False,
                    }
                    row.update(summarize_prediction(df, data.y_eval, data.baseline_eval, corrected, override))
                    row["val_noharm_all"] = bool(row["val_all_delta_vs_baseline_mean"] <= 0.003)
                    row["val_improves_bad_top10"] = bool(row["val_bad_top10_delta_vs_baseline_mean"] < 0.0)
                    row["val_active_bad_top10"] = bool(row["val_bad_top10_override_rate"] and row["val_bad_top10_override_rate"] > 0)
                    row["val_candidate_ok"] = bool(
                        row["val_noharm_all"] and row["val_improves_bad_top10"] and row["val_active_bad_top10"]
                    )
                    rows.append(row)

                    # 只为少量候选保留事件级输出，避免生成几百 MB 表。
                    if row["val_candidate_ok"] or risk["risk_tag"] in ["always_apply", "no_override"]:
                        event = df[
                            [
                                "event_uid",
                                "subject",
                                "recording",
                                "split",
                                "bad_top10",
                                "vehicle_ambiguous",
                                "bad_top10_vehicle_ambiguous",
                                "candidate_pool_gain_gt_005",
                            ]
                        ].copy()
                        event["selector_tag"] = row["selector_tag"]
                        event["risk_score"] = score
                        event["threshold"] = threshold
                        event["override"] = override
                        event["baseline_rmse"] = base_rmse
                        event["corrected_rmse"] = corr_rmse
                        event["delta_vs_baseline"] = corr_rmse - base_rmse
                        selected_rows.append(event)

    selected = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
    return pd.DataFrame(rows), selected


def choose_rows(summary: pd.DataFrame) -> pd.DataFrame:
    chosen: List[pd.Series] = []

    fallback = summary[summary["risk_tag"].eq("no_override")].copy()
    if not fallback.empty:
        row = fallback.iloc[0].copy()
        row["choice_name"] = "fallback_no_correction"
        row["choice_rule"] = "zero residual correction"
        chosen.append(row)

    candidates = summary[summary["val_candidate_ok"]].copy()
    if not candidates.empty:
        overall = candidates.sort_values(
            ["val_bad_top10_delta_vs_baseline_mean", "val_all_delta_vs_baseline_mean", "shrinkage"],
            ascending=[True, True, True],
        ).iloc[0].copy()
        overall["choice_name"] = "best_val_overall_deployable"
        overall["choice_rule"] = "val all no-harm and val bad_top10 improvement, best val bad_top10 delta"
        chosen.append(overall)

        phys = candidates[
            candidates["uses_physio_in_residual"].astype(bool) | candidates["uses_physio_in_gate"].astype(bool)
        ].copy()
        if not phys.empty:
            prow = phys.sort_values(
                ["val_bad_top10_delta_vs_baseline_mean", "val_all_delta_vs_baseline_mean", "shrinkage"],
                ascending=[True, True, True],
            ).iloc[0].copy()
            prow["choice_name"] = "best_val_physio_deployable"
            prow["choice_rule"] = "same val rule, restricted to configs using post0_1 physiology"
            chosen.append(prow)

        nonphys = candidates[
            (~candidates["uses_physio_in_residual"].astype(bool)) & (~candidates["uses_physio_in_gate"].astype(bool))
        ].copy()
        if not nonphys.empty:
            nrow = nonphys.sort_values(
                ["val_bad_top10_delta_vs_baseline_mean", "val_all_delta_vs_baseline_mean", "shrinkage"],
                ascending=[True, True, True],
            ).iloc[0].copy()
            nrow["choice_name"] = "best_val_nonphysio_ablation"
            nrow["choice_rule"] = "same val rule, no physiology in residual or gate"
            chosen.append(nrow)

    active = summary[
        summary["risk_tag"].ne("no_override") & (summary["val_all_delta_vs_baseline_mean"] <= 0.006)
    ].copy()
    if not active.empty:
        diag = active.sort_values("test_bad_top10_delta_vs_baseline_mean", ascending=True).iloc[0].copy()
        diag["choice_name"] = "test_best_diagnostic_not_deployable"
        diag["choice_rule"] = "diagnostic only; selected by test bad_top10 delta after val all bound"
        chosen.append(diag)

    out = pd.DataFrame(chosen)
    if out.empty:
        return out
    return out.drop_duplicates(["choice_name", "selector_tag"]).reset_index(drop=True)


def selector_prediction_table(data: CurveData, chosen: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    if chosen.empty or selected.empty:
        return pd.DataFrame()
    keep_tags = set(chosen["selector_tag"].astype(str).tolist())
    out = selected[selected["selector_tag"].astype(str).isin(keep_tags)].copy()
    return out.sort_values(["selector_tag", "split", "baseline_rmse"], ascending=[True, True, False])


def plot_choice_bars(chosen: pd.DataFrame) -> Path:
    path = FIGURES / "v295_chosen_selector_test_delta.png"
    if chosen.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No chosen selector", ha="center", va="center")
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path
    data = chosen.copy()
    labels = data["choice_name"].astype(str).tolist()
    x = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(max(10, len(data) * 2.2), 5))
    ax.bar(x - 0.18, data["test_all_delta_vs_baseline_mean"], width=0.36, label="test all")
    ax.bar(x + 0.18, data["test_bad_top10_delta_vs_baseline_mean"], width=0.36, label="test bad_top10")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("delta RMSE vs v249 wait1 baseline (lower is better)")
    ax.set_title("v295 chosen wait1 direct residual selectors")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_bad_examples(data: CurveData, chosen: pd.DataFrame, selected: pd.DataFrame, residual_preds: List[Dict[str, object]]) -> Path:
    path = FIGURES / "v295_test_bad_top6_curves.png"
    df = data.df.reset_index(drop=True)
    test_bad = df["split"].astype(str).eq("test") & df["bad_top10"].fillna(0).astype(bool)
    if chosen.empty or selected.empty or not test_bad.any():
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No test bad examples", ha="center", va="center")
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path

    # 优先画 validation 规则选出的生理 deployable；没有则画 overall deployable。
    choice_order = ["best_val_physio_deployable", "best_val_overall_deployable", "test_best_diagnostic_not_deployable"]
    chosen_row = None
    for name in choice_order:
        hit = chosen[chosen["choice_name"].eq(name)]
        if not hit.empty:
            chosen_row = hit.iloc[0]
            break
    if chosen_row is None:
        chosen_row = chosen.iloc[0]
    tag = str(chosen_row["selector_tag"])

    # 重新构造该 selector 的 corrected curve，用于曲线图。
    residual_tag = str(chosen_row["residual_tag"])
    pred_res = None
    for item in residual_preds:
        if str(item["residual_tag"]) == residual_tag:
            pred_res = np.asarray(item["pred_residual"], dtype=float)
            break
    if pred_res is None:
        pred_res = np.zeros_like(data.baseline_eval)
    event_sel = selected[selected["selector_tag"].astype(str).eq(tag)].copy()
    event_sel = event_sel.drop_duplicates("event_uid").set_index("event_uid")
    override = np.array(
        [bool(event_sel.loc[uid, "override"]) if uid in event_sel.index else False for uid in df["event_uid"].astype(str)]
    )
    corrected = data.baseline_eval + float(chosen_row["shrinkage"]) * pred_res * override[:, None]

    base_rmse = rmse_rows(data.y_eval, data.baseline_eval)
    idx = np.where(test_bad.to_numpy())[0]
    idx = idx[np.argsort(base_rmse[idx])[::-1]][:6]
    fig, axes = plt.subplots(len(idx), 1, figsize=(11, max(2.2 * len(idx), 4)), sharex=True)
    if len(idx) == 1:
        axes = [axes]
    for ax, i in zip(axes, idx):
        ax.plot(data.eval_grid_s, data.y_eval[i], color="black", lw=1.8, label="true wait1 future")
        ax.plot(data.eval_grid_s, data.baseline_eval[i], color="#1f77b4", lw=1.5, ls="--", label="v249 wait1 baseline")
        ax.plot(data.eval_grid_s, corrected[i], color="#d62728", lw=1.5, ls="-.", label="v295 corrected")
        br = float(np.sqrt(np.mean((data.y_eval[i] - data.baseline_eval[i]) ** 2)))
        cr = float(np.sqrt(np.mean((data.y_eval[i] - corrected[i]) ** 2)))
        uid = str(df.loc[i, "event_uid"])
        ax.set_title(f"{uid} | baseline={br:.3f} corrected={cr:.3f} override={bool(override[i])}", fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.set_ylabel("steering delta")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("seconds after wait1 rolling anchor")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(
    data: CurveData,
    risk_audit: pd.DataFrame,
    summary: pd.DataFrame,
    chosen: pd.DataFrame,
    guardrail: Dict[str, object],
) -> Path:
    lines: List[str] = []
    lines.append("# v295 wait1 direct residual + physiology 审计")
    lines.append("")
    lines.append("## 结论")
    if guardrail.get("route_viable_now"):
        lines.append("- v295 达到当前设定的 wait1 physiology residual 路线可用标准。")
    else:
        lines.append("- v295 没达到“本质性改善差样本”的标准；它更像是一个方向性探针。")
    lines.append(
        f"- 当前 best_val_physio_deployable test bad_top10 delta = {guardrail.get('best_physio_test_badtop10_delta', math.nan):.6f}，"
        f"test all delta = {guardrail.get('best_physio_test_all_delta', math.nan):.6f}。"
    )
    lines.append(
        "- 负 delta 表示比 v249 wait1 rolling baseline 更好；正 delta 表示变差。"
    )
    lines.append("")
    lines.append("## 方法")
    lines.append("- baseline: v249 `delay_ms=1000` 的 rolling 预测。")
    lines.append("- target: `y_true - baseline` 的 wait1 残差曲线，只在共同有效 horizon 点上训练和评估。")
    lines.append("- inputs: baseline 曲线形态、原锚点后 0-1s 已观测车辆响应、v293 `post0_1` 生理特征、subject one-hot。")
    lines.append("- selection: residual 模型只用 train；风险 gate 阈值只用 val；test 不参与筛选。")
    lines.append("")
    lines.append("## 数据口径")
    lines.append(
        f"- event_n={len(data.df)}, eval_point_n={data.y_eval.shape[1]}, eval_grid={data.eval_grid_s.tolist()}, "
        f"target_cols={data.target_cols.tolist()}."
    )
    split_counts = data.df.groupby("split").size().reset_index(name="n")
    lines.append(markdown_table(split_counts, ["split", "n"], 10))
    lines.append("")
    lines.append("## chosen selectors")
    cols = [
        "choice_name",
        "feature_block",
        "model_name",
        "shrinkage",
        "risk_tag",
        "threshold",
        "risk_val_auc",
        "risk_test_auc",
        "val_all_delta_vs_baseline_mean",
        "val_bad_top10_delta_vs_baseline_mean",
        "test_all_delta_vs_baseline_mean",
        "test_bad_top10_delta_vs_baseline_mean",
        "test_bad_top10_vehicle_ambiguous_delta_vs_baseline_mean",
        "test_bad_top10_override_rate",
    ]
    lines.append(markdown_table(chosen, cols, 20))
    lines.append("")
    lines.append("## risk classifier audit")
    lines.append(markdown_table(risk_audit.sort_values("test_auc", ascending=False), ["risk_tag", "val_auc", "test_auc", "feature_n"], 20))
    lines.append("")
    lines.append("## top validation candidates")
    top = summary[summary["val_candidate_ok"]].sort_values("val_bad_top10_delta_vs_baseline_mean").head(30)
    lines.append(
        markdown_table(
            top,
            [
                "feature_block",
                "model_name",
                "shrinkage",
                "risk_tag",
                "risk_val_auc",
                "val_all_delta_vs_baseline_mean",
                "val_bad_top10_delta_vs_baseline_mean",
                "test_all_delta_vs_baseline_mean",
                "test_bad_top10_delta_vs_baseline_mean",
            ],
            30,
        )
    )
    lines.append("")
    lines.append("## guardrail")
    lines.append("```json")
    lines.append(json.dumps(guardrail, ensure_ascii=False, indent=2))
    lines.append("```")
    path = REPORTS / "v295_wait1_direct_residual_physio_cn.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def make_zip() -> None:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(OUT.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(OUT.parent))
        zf.write(THIS_SCRIPT, Path("scripts") / THIS_SCRIPT.name)
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        bad = zf.testzip()
    if bad is not None:
        raise RuntimeError(f"zip test failed at {bad}")


def build_guardrail(data: CurveData, chosen: pd.DataFrame, summary: pd.DataFrame, risk_audit: pd.DataFrame) -> Dict[str, object]:
    def get_choice(name: str) -> pd.Series | None:
        hit = chosen[chosen["choice_name"].eq(name)]
        return None if hit.empty else hit.iloc[0]

    phys = get_choice("best_val_physio_deployable")
    nonphys = get_choice("best_val_nonphysio_ablation")
    overall = get_choice("best_val_overall_deployable")
    diag = get_choice("test_best_diagnostic_not_deployable")

    best_phys_bad = float(phys["test_bad_top10_delta_vs_baseline_mean"]) if phys is not None else math.nan
    best_phys_all = float(phys["test_all_delta_vs_baseline_mean"]) if phys is not None else math.nan
    best_nonphys_bad = float(nonphys["test_bad_top10_delta_vs_baseline_mean"]) if nonphys is not None else math.nan
    phys_increment = best_phys_bad - best_nonphys_bad if np.isfinite(best_phys_bad) and np.isfinite(best_nonphys_bad) else math.nan

    route_viable = bool(
        phys is not None
        and best_phys_bad <= -0.05
        and best_phys_all <= 0.005
        and float(phys["val_bad_top10_delta_vs_baseline_mean"]) < 0
        and float(phys["val_all_delta_vs_baseline_mean"]) <= 0.003
    )
    weak_signal = bool(
        phys is not None
        and best_phys_bad < -0.01
        and best_phys_all <= 0.01
        and float(phys["val_bad_top10_delta_vs_baseline_mean"]) < 0
    )

    return {
        "pass": True,
        "event_n": int(len(data.df)),
        "wait_ms": WAIT_MS,
        "wait_s": WAIT_S,
        "eval_point_n": int(data.y_eval.shape[1]),
        "eval_grid_s": [float(x) for x in data.eval_grid_s],
        "baseline": "v249_shape_conditioned_residual_delay1000",
        "uses_post_observation": True,
        "post_features_are_wait_policy_only": True,
        "post_window_used": "v293_post0_1",
        "test_used_for_feature_screen_model_or_threshold": False,
        "chosen_physio_exists": phys is not None,
        "chosen_overall_exists": overall is not None,
        "best_physio_test_badtop10_delta": best_phys_bad,
        "best_physio_test_all_delta": best_phys_all,
        "best_nonphysio_test_badtop10_delta": best_nonphys_bad,
        "physio_increment_vs_nonphysio_badtop10_delta": phys_increment,
        "best_overall_test_badtop10_delta": float(overall["test_bad_top10_delta_vs_baseline_mean"]) if overall is not None else math.nan,
        "best_diagnostic_test_badtop10_delta": float(diag["test_bad_top10_delta_vs_baseline_mean"]) if diag is not None else math.nan,
        "best_risk_test_auc": float(risk_audit["test_auc"].max()) if not risk_audit.empty else math.nan,
        "route_viable_now": route_viable,
        "weak_physio_residual_signal_exists": weak_signal,
        "goal_achieved_now": route_viable,
        "requirement_for_route_viable": "physio deployable must improve test bad_top10 by at least 0.05 RMSE with test all delta <=0.005 and val no-harm",
    }


def main() -> None:
    np.random.seed(SEED)
    ensure_dirs()
    input_hashes = pd.DataFrame(
        [
            {"path": str(V249_NPZ), "sha256": file_sha256(V249_NPZ), "role": "curve truth and v249 baseline"},
            {"path": str(V293_FEATURES), "sha256": file_sha256(V293_FEATURES), "role": "event-level physiology features"},
            {"path": str(V293_SCREEN), "sha256": file_sha256(V293_SCREEN), "role": "train-only physiology feature screen"},
            {"path": str(V293_GUARDRAIL), "sha256": file_sha256(V293_GUARDRAIL), "role": "upstream guardrail"},
        ]
    )
    write_csv(input_hashes, LOGS / "input_hashes.csv")

    print("[v295] loading curve and physiology data")
    data = load_curve_data()
    screen = pd.read_csv(V293_SCREEN)
    post01_cols, pre_cols = select_physio_columns(screen)

    feature_blocks = build_feature_blocks(data, post01_cols, pre_cols)
    risk_blocks = build_risk_blocks(data, post01_cols, pre_cols)
    feature_audit = pd.DataFrame(
        [{"block": k, "feature_n": v.shape[1], "uses_physio": "physio" in k} for k, v in feature_blocks.items()]
    )
    risk_feature_audit = pd.DataFrame(
        [{"risk_block": k, "feature_n": v.shape[1], "uses_physio": "physio" in k} for k, v in risk_blocks.items()]
    )
    write_csv(feature_audit, TABLES / "v295_feature_block_audit.csv")
    write_csv(risk_feature_audit, TABLES / "v295_risk_feature_block_audit.csv")

    residual_preds = fit_residual_predictions(data, feature_blocks)
    risk_scores, risk_audit = fit_risk_scores(data, risk_blocks)
    summary, selected = evaluate_configs(data, residual_preds, risk_scores)
    chosen = choose_rows(summary)
    selected_chosen = selector_prediction_table(data, chosen, selected)
    guardrail = build_guardrail(data, chosen, summary, risk_audit)

    write_csv(summary, TABLES / "v295_wait1_residual_selector_summary.csv")
    write_csv(risk_audit, TABLES / "v295_badtop10_risk_classifier_audit.csv")
    write_csv(chosen, TABLES / "v295_chosen_by_val.csv")
    write_csv(selected_chosen, TABLES / "v295_chosen_event_predictions.csv")
    write_csv(
        pd.DataFrame(
            [
                {"feature_type": "post0_1_physio", "rank": i + 1, "feature": c}
                for i, c in enumerate(post01_cols)
            ]
            + [{"feature_type": "pre_physio", "rank": i + 1, "feature": c} for i, c in enumerate(pre_cols)]
        ),
        TABLES / "v295_selected_physio_features.csv",
    )
    write_json(guardrail, LOGS / "guardrail_check.json")

    plot_choice_bars(chosen)
    plot_bad_examples(data, chosen, selected_chosen, residual_preds)
    write_report(data, risk_audit, summary, chosen, guardrail)

    inventory_rows = []
    for p in sorted(OUT.rglob("*")):
        if p.is_file():
            inventory_rows.append({"path": str(p), "bytes": int(p.stat().st_size)})
    write_csv(pd.DataFrame(inventory_rows), LOGS / "file_inventory.csv")
    make_zip()
    print("[v295] done")
    print(json.dumps(guardrail, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
