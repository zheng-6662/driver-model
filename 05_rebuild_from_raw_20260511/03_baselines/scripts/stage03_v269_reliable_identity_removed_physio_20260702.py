"""
v269 reliable / identity-removed physiology representation.

背景：
v268 已经说明，200Hz 连续生理源层本身稳定，但当前 bio260 派生表征
更容易编码 subject/recording 身份，而不是可迁移的行为差异。

本轮目标：
1. 不再盲目增加融合模型深度。
2. 从 v260 事件级 biomarker 中重新构造一组更可靠、更偏动态变化、
   更弱身份混淆的生理特征。
3. 在同一 subject-disjoint 协议下，重新验证两个最直接任务：
   - wait gate：只判断 0ms vs latest 是否等待；
   - query-prototype pair reranker：在 v266/v267 已有候选库中选择轨迹。

边界：
- 特征选择只使用 train split 的统计与标签，test 不参与选择。
- 阈值/策略选择只用 val bad_top10，test 只报告。
- prototype 仍只来自 train split，不使用 val/test 驾驶员历史。
- 生理特征只来自 observation 前窗口；复用 v260/v266 的 post-observation guardrail。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


ROOT = Path(r"F:\data_set_process\data_process")
REBUILD = ROOT / "05_rebuild_from_raw_20260511"
BASELINES = REBUILD / "03_baselines"

OUT = BASELINES / "v269_reliable_identity_removed_physio_20260702"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
ZIP_PATH = BASELINES / "v269_reliable_identity_removed_physio_20260702_pack.zip"

V260_FEATURES = BASELINES / "v260_event_biomarker_physio_rebuild_20260702" / "tables" / "v260_event_biomarker_features.csv"
V266_EVENTS = BASELINES / "v266_vehicle_matched_bio_residual_prototype_20260702" / "tables" / "v266_event_context_table.csv"
V267_SUMMARY = BASELINES / "v267_supervised_bio_prototype_reranker_20260702" / "tables" / "v267_pair_reranker_summary.csv"
V267_CHOSEN = BASELINES / "v267_supervised_bio_prototype_reranker_20260702" / "tables" / "v267_val_chosen_pair_strategy_summary.csv"

V266_SCRIPT = BASELINES / "scripts" / "stage03_v266_vehicle_matched_bio_residual_prototype_20260702.py"
V267_SCRIPT = BASELINES / "scripts" / "stage03_v267_supervised_bio_prototype_reranker_20260702.py"

SEED = 26902
K_VALUES = [3, 5, 10, 20, 40]
FIXED_WAIT_LATEST_BADTOP10 = 0.695048


def import_module_from_path(module_name: str, path: Path):
    """按路径导入前序脚本，复用已经通过 guardrail 的候选与汇总函数。"""
    if not path.exists():
        raise FileNotFoundError(f"缺少前序脚本：{path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法导入模块：{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


V266 = import_module_from_path("stage03_v266_for_v269", V266_SCRIPT)
V267 = import_module_from_path("stage03_v267_for_v269", V267_SCRIPT)


def ensure_dirs() -> None:
    """创建 v269 输出目录。"""
    for folder in (TABLES, FIGURES, REPORTS, LOGS):
        folder.mkdir(parents=True, exist_ok=True)


def clean_out_dir() -> None:
    """只清理 v269 自己的输出，避免误删其他实验。"""
    if OUT.exists():
        shutil.rmtree(OUT)
    ensure_dirs()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """统一用 utf-8-sig，方便 Excel 和中文报告读取。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def eta_squared(feature: np.ndarray, labels: np.ndarray) -> float:
    """单变量 eta^2，用来衡量某个特征对离散标签的可分性。"""
    x = np.asarray(feature, dtype=float)
    lab = np.asarray(labels)
    mask = np.isfinite(x) & pd.notna(lab)
    if int(mask.sum()) < 8:
        return 0.0
    x = x[mask]
    lab = lab[mask]
    grand = float(np.mean(x))
    total = float(np.sum((x - grand) ** 2))
    if total <= 1e-12:
        return 0.0
    between = 0.0
    for value in pd.unique(lab):
        sub = x[lab == value]
        if len(sub) == 0:
            continue
        between += float(len(sub)) * (float(np.mean(sub)) - grand) ** 2
    return max(0.0, min(1.0, between / total))


def finite_columns(df: pd.DataFrame, cols: Iterable[str]) -> List[str]:
    """保留数值列，避免字符串状态列进入模型。"""
    out: List[str] = []
    for col in cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            out.append(col)
    return out


def bio_base_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    从 v260 delay=0 事件表中抽取可建模生理列。

    这里主动剔除 rows/duration/status/sample_hz 等元数据，以及 v268 已证明
    不可靠的 existing HRV 派生列；ECG 峰间期重算出来的 ibi_rmssd_s 仍保留，
    因为它来自 200Hz ECG 峰列而不是全空的 HRV_RMSSD 记录列。
    """
    banned_tokens = [
        "_rows",
        "_duration_s",
        "sample_hz",
        "recording_duration",
        "uses_post_observation",
        "baseline_rows",
        "baseline_duration",
        "hrv_existing",
    ]
    cols = []
    for col in df.columns:
        if not col.startswith("bio260_"):
            continue
        if any(tok in col for tok in banned_tokens):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def metric_suffix(col: str) -> Tuple[str, str] | None:
    """把 bio260_pre2_0_xxx 拆成窗口名和同一指标后缀，用于构造窗口差分。"""
    prefix = "bio260_"
    if not col.startswith(prefix):
        return None
    rest = col[len(prefix) :]
    for win in ["pre20_pre10", "pre10_pre5", "pre5_pre2", "pre2_0"]:
        head = win + "_"
        if rest.startswith(head):
            return win, rest[len(head) :]
    return None


def add_dynamic_delta_features(events: pd.DataFrame, base_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    构造同一生理指标的跨窗口变化量。

    这些差分特征比绝对水平更接近“锚点前状态变化”，理论上可以降低
    subject/recording 的静态身份差异影响。
    """
    grouped: Dict[str, Dict[str, str]] = {}
    for col in base_cols:
        parsed = metric_suffix(col)
        if parsed is None:
            continue
        win, suffix = parsed
        grouped.setdefault(suffix, {})[win] = col

    out = events.copy()
    new_cols: List[str] = []
    pairs = [
        ("pre2_0", "pre10_pre5", "last2_minus_pre10_5"),
        ("pre2_0", "pre5_pre2", "last2_minus_pre5_2"),
        ("pre5_pre2", "pre20_pre10", "pre5_2_minus_pre20_10"),
        ("pre2_0", "pre20_pre10", "last2_minus_pre20_10"),
    ]
    for suffix, win_map in grouped.items():
        for late, early, label in pairs:
            if late not in win_map or early not in win_map:
                continue
            col = f"irbio_delta_{label}_{suffix}"
            out[col] = pd.to_numeric(out[win_map[late]], errors="coerce") - pd.to_numeric(out[win_map[early]], errors="coerce")
            new_cols.append(col)
    return out, new_cols


def load_event_sources() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], List[str], pd.DataFrame]:
    """
    读取 v266 事件上下文和 v260 delay=0 生理特征。

    v266 提供已经验证过的 vehicle/prototype 事件框架；
    v260 提供更全的事件级 biomarker，v269 在其上重新筛选与差分。
    """
    cand, base_events, merge_audit, veh_cols, old_bio_cols = V266.load_candidate_and_events()
    v260 = pd.read_csv(V260_FEATURES, encoding="utf-8-sig", low_memory=False)
    v260["delay_ms"] = pd.to_numeric(v260["delay_ms"], errors="coerce")
    v260_0 = v260[v260["delay_ms"].eq(0)].copy()
    if v260_0["event_uid"].duplicated().any():
        v260_0 = v260_0.sort_values(["event_uid", "observation_s"]).drop_duplicates("event_uid", keep="first")

    bio_cols = bio_base_feature_columns(v260_0)
    keep = ["event_uid", "bio260_status", "bio260_uses_post_observation"] + bio_cols
    bio0 = v260_0[keep].copy()
    events = base_events.drop(columns=[c for c in old_bio_cols if c in base_events.columns], errors="ignore").merge(
        bio0, on="event_uid", how="left", validate="one_to_one"
    )
    events["bio260_status_ok"] = events["bio260_status"].astype(str).eq("ok").astype(float)
    if "bio260_uses_post_observation" in events.columns:
        events["bio260_uses_post_observation"] = events["bio260_uses_post_observation"].astype(str).str.lower().eq("true")
    else:
        events["bio260_uses_post_observation"] = False
    events["wait_better_latest_vs_keep0"] = (
        pd.to_numeric(events["latest_tail_rmse_v241"], errors="coerce")
        < pd.to_numeric(events["keep0_tail_rmse_v241"], errors="coerce")
    )
    events, delta_cols = add_dynamic_delta_features(events, bio_cols)
    all_bio_cols = finite_columns(events, bio_cols + delta_cols + ["bio260_status_ok"])
    return cand, events, merge_audit, veh_cols, all_bio_cols, v260


def screen_features(events: pd.DataFrame, bio_cols: List[str]) -> pd.DataFrame:
    """
    只用 train split 做特征可靠性与身份/行为可识别性打分。

    注意：这里不能用 test 标签做筛选，否则会把评估集信息泄漏进模型。
    """
    train = events["split"].astype(str).eq("train").to_numpy()
    subject = events.loc[train, "subject"].astype(str).to_numpy()
    recording = events.loc[train, "recording"].astype(str).to_numpy()
    bad = events.loc[train, "bad_top10"].astype(str).to_numpy()
    early = events.loc[train, "early_best_after_400"].astype(str).to_numpy()
    wait_better = events.loc[train, "wait_better_latest_vs_keep0"].astype(str).to_numpy()

    rows: List[Dict[str, object]] = []
    for col in bio_cols:
        x_all = pd.to_numeric(events[col], errors="coerce").to_numpy(dtype=float)
        x_train = x_all[train]
        finite_train = np.isfinite(x_train)
        finite_all = np.isfinite(x_all)
        std_train = float(np.nanstd(x_train)) if finite_train.any() else 0.0
        eta_subject = eta_squared(x_train, subject)
        eta_recording = eta_squared(x_train, recording)
        eta_bad = eta_squared(x_train, bad)
        eta_early = eta_squared(x_train, early)
        eta_wait = eta_squared(x_train, wait_better)
        identity_eta = max(eta_subject, eta_recording)
        behavior_eta = max(eta_bad, eta_early, eta_wait)
        missing_train = 1.0 - float(finite_train.mean())
        family = signal_family(col)
        is_delta = col.startswith("irbio_delta_")
        reliable = bool(finite_train.mean() >= 0.75 and std_train > 1e-9)
        # 分数偏向行为可分性，同时惩罚身份可分性和缺失；保留正负排序而不做硬阈值，避免全被筛空。
        score = (behavior_eta + 0.002) / (identity_eta + 0.02) - 0.25 * missing_train + (0.015 if is_delta else 0.0)
        rows.append(
            {
                "feature": col,
                "family": family,
                "is_delta": bool(is_delta),
                "finite_rate_train": float(finite_train.mean()),
                "finite_rate_all": float(finite_all.mean()),
                "std_train": std_train,
                "eta_subject_train": eta_subject,
                "eta_recording_train": eta_recording,
                "eta_bad_top10_train": eta_bad,
                "eta_early_best_after_400_train": eta_early,
                "eta_wait_better_train": eta_wait,
                "identity_eta_max_train": identity_eta,
                "behavior_eta_max_train": behavior_eta,
                "identity_to_behavior_ratio_train": identity_eta / max(behavior_eta, 1e-6),
                "selection_score": score,
                "reliable": reliable,
            }
        )
    return pd.DataFrame(rows).sort_values("selection_score", ascending=False)


def signal_family(col: str) -> str:
    s = col.lower()
    if "ecg" in s or "ibi" in s:
        return "ecg"
    if "scr" in s or "eda" in s:
        return "scr"
    if "resp" in s:
        return "resp"
    if "emg" in s:
        return "emg"
    if "_hr_" in s or s.endswith("_hr"):
        return "hr"
    return "other"


def choose_feature_sets(screen: pd.DataFrame, old_bio_cols: List[str] | None = None) -> Tuple[Dict[str, List[str]], pd.DataFrame]:
    """
    构造多个可证伪特征集。

    - reliable_top64：只做可靠性过滤后按身份惩罚分数选取；
    - dynamic_top48：只保留跨窗口变化量；
    - low_identity_top32：优先低身份 eta；
    - combo_identity_removed64：动态变化 + 少量低身份原始特征。
    """
    reliable = screen[screen["reliable"].astype(bool)].copy()
    raw = reliable[~reliable["is_delta"].astype(bool)].copy()
    delta = reliable[reliable["is_delta"].astype(bool)].copy()
    lowid = reliable.sort_values(["identity_eta_max_train", "selection_score"], ascending=[True, False]).copy()

    sets: Dict[str, List[str]] = {}
    sets["reliable_top64"] = reliable.sort_values("selection_score", ascending=False)["feature"].head(64).tolist()
    sets["dynamic_top48"] = delta.sort_values("selection_score", ascending=False)["feature"].head(48).tolist()
    sets["low_identity_top32"] = lowid["feature"].head(32).tolist()
    combo = (
        delta.sort_values("selection_score", ascending=False)["feature"].head(48).tolist()
        + raw.sort_values(["identity_eta_max_train", "selection_score"], ascending=[True, False])["feature"].head(16).tolist()
    )
    sets["combo_identity_removed64"] = list(dict.fromkeys(combo))[:64]
    if old_bio_cols:
        sets["old_sp64_reference"] = list(old_bio_cols)

    rows: List[Dict[str, object]] = []
    for name, cols in sets.items():
        sub = screen[screen["feature"].isin(cols)]
        rows.append(
            {
                "bio_set": name,
                "feature_n": int(len(cols)),
                "delta_feature_n": int(sub["is_delta"].astype(bool).sum()) if len(sub) else 0,
                "behavior_eta_max_mean": float(sub["behavior_eta_max_train"].mean()) if len(sub) else math.nan,
                "identity_eta_max_mean": float(sub["identity_eta_max_train"].mean()) if len(sub) else math.nan,
                "identity_to_behavior_ratio_median": float(sub["identity_to_behavior_ratio_train"].median()) if len(sub) else math.nan,
                "finite_rate_train_mean": float(sub["finite_rate_train"].mean()) if len(sub) else math.nan,
                "features": ";".join(cols),
            }
        )
    return sets, pd.DataFrame(rows)


def fit_predict_hgb(events: pd.DataFrame, cols: List[str], target: str, bad_weight: bool = False) -> Tuple[np.ndarray, pd.DataFrame]:
    """训练一个 train-only HGB 回归器，并返回全 split 预测。"""
    train_mask = events["split"].astype(str).eq("train").to_numpy()
    x = events[cols].to_numpy(dtype=float)
    y = pd.to_numeric(events[target], errors="coerce").to_numpy(dtype=float)
    xz, med, mean, std = V266.fit_fill_scale(x, train_mask)
    good = train_mask & np.isfinite(y)
    weights = None
    if bad_weight:
        weights = 1.0 + 4.0 * events.loc[good, "bad_top10"].astype(bool).to_numpy(dtype=float)
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=420,
        learning_rate=0.045,
        max_leaf_nodes=31,
        l2_regularization=0.03,
        random_state=SEED + (17 if bad_weight else 0),
    )
    model.fit(xz[good], y[good], sample_weight=weights)
    pred = model.predict(xz)
    audit = pd.DataFrame(
        {
            "feature": cols,
            "fill_median": med,
            "scale_mean": mean,
            "scale_std": std,
            "bad_weight": bad_weight,
        }
    )
    return pred, audit


def tune_threshold(events: pd.DataFrame, pred_col: str, bad_weight: bool) -> Tuple[float, pd.DataFrame]:
    """
    在 val split 上选择 wait gate 阈值。

    只优化 val bad_top10 加权 RMSE，不看 test。
    """
    val = events[events["split"].astype(str).eq("val")].copy()
    pred = pd.to_numeric(val[pred_col], errors="coerce").to_numpy(dtype=float)
    finite = pred[np.isfinite(pred)]
    if len(finite) == 0:
        return 0.0, pd.DataFrame()
    grid = np.unique(np.quantile(finite, np.linspace(0.02, 0.98, 97)))
    grid = np.unique(np.concatenate([grid, np.array([0.0])]))
    keep0 = pd.to_numeric(val["keep0_tail_rmse_v241"], errors="coerce").to_numpy(dtype=float)
    latest = pd.to_numeric(val["latest_tail_rmse_v241"], errors="coerce").to_numpy(dtype=float)
    weights = np.ones(len(val), dtype=float)
    if bad_weight:
        weights += 4.0 * val["bad_top10"].astype(bool).to_numpy(dtype=float)
    rows: List[Dict[str, object]] = []
    for threshold in grid:
        choose_latest = pred > float(threshold)
        selected = np.where(choose_latest, latest, keep0)
        rows.append(
            {
                "pred_col": pred_col,
                "bad_weight": bad_weight,
                "threshold": float(threshold),
                "val_tail_rmse_weighted": float(np.average(selected, weights=weights)),
                "val_tail_rmse": float(np.mean(selected)),
                "val_latest_rate": float(choose_latest.mean()),
            }
        )
    audit = pd.DataFrame(rows).sort_values(["val_tail_rmse_weighted", "val_tail_rmse", "val_latest_rate"], ascending=[True, True, True])
    return float(audit.iloc[0]["threshold"]), audit


def build_wait_selected(events: pd.DataFrame, strategy: str, family: str, pred_col: str | None = None, threshold: float = 0.0) -> pd.DataFrame:
    """生成 wait gate 逐事件选择结果，结构对齐 v266/v267 的汇总函数。"""
    rows: List[Dict[str, object]] = []
    for _, event in events.iterrows():
        uid = str(event["event_uid"])
        if strategy == "policy_keep_0ms_anchor":
            delay = 0
            rmse = float(event["keep0_tail_rmse_v241"])
        elif strategy == "policy_wait_to_latest_anchor":
            delay = int(event["latest_delay_ms"])
            rmse = float(event["latest_tail_rmse_v241"])
        elif strategy == "oracle_best_anchor_upper_bound":
            delay = int(event["oracle_delay_ms"])
            rmse = float(event["oracle_tail_rmse_v241"])
        else:
            if pred_col is None:
                raise ValueError("learned gate 需要 pred_col")
            choose_latest = float(event[pred_col]) > float(threshold)
            delay = int(event["latest_delay_ms"]) if choose_latest else int(event["keep0_delay_ms"])
            rmse = float(event["latest_tail_rmse_v241"]) if choose_latest else float(event["keep0_tail_rmse_v241"])
        deployable = strategy != "oracle_best_anchor_upper_bound"
        rows.append(V267.selected_row(event, strategy, family, deployable, np.nan, delay, rmse, 1))
        rows[-1]["event_uid"] = uid
    return pd.DataFrame(rows)


def run_wait_gate(events: pd.DataFrame, veh_cols: List[str], feature_sets: Dict[str, List[str]]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """训练并评估 vehicle、bio、vehicle+bio wait gate。"""
    out = events.copy()
    out["target_gain_latest_vs_keep0"] = (
        pd.to_numeric(out["keep0_tail_rmse_v241"], errors="coerce") - pd.to_numeric(out["latest_tail_rmse_v241"], errors="coerce")
    )
    selected_parts = [
        build_wait_selected(out, "policy_keep_0ms_anchor", "baseline"),
        build_wait_selected(out, "policy_wait_to_latest_anchor", "baseline"),
        build_wait_selected(out, "oracle_best_anchor_upper_bound", "oracle"),
    ]
    threshold_parts: List[pd.DataFrame] = []
    fill_parts: List[pd.DataFrame] = []
    feature_rows: List[Dict[str, object]] = []

    model_specs: List[Tuple[str, str, List[str], bool]] = [
        ("wait_vehicle_gain", "vehicle_only", veh_cols, False),
        ("wait_vehicle_gain_badweighted", "vehicle_only", veh_cols, True),
    ]
    for set_name, bio_cols in feature_sets.items():
        if not bio_cols:
            continue
        model_specs.extend(
            [
                (f"wait_bio_{set_name}_gain", "bio_only", bio_cols, False),
                (f"wait_vehicle_bio_{set_name}_gain", "vehicle_bio", veh_cols + bio_cols, False),
                (f"wait_vehicle_bio_{set_name}_gain_badweighted", "vehicle_bio", veh_cols + bio_cols, True),
            ]
        )

    for model_name, family, cols, bad_weight in model_specs:
        pred, fill = fit_predict_hgb(out, cols, "target_gain_latest_vs_keep0", bad_weight=bad_weight)
        pred_col = f"pred_{model_name}"
        out[pred_col] = pred
        threshold, audit = tune_threshold(out, pred_col, bad_weight=bad_weight)
        if len(audit):
            threshold_parts.append(audit.assign(model_name=model_name, family=family))
        fill_parts.append(fill.assign(model_name=model_name, family=family))
        feature_rows.append(
            {
                "model_name": model_name,
                "family": family,
                "feature_n": int(len(cols)),
                "bio_feature_n": int(sum(1 for c in cols if c not in veh_cols)),
                "bad_weight": bool(bad_weight),
                "threshold": float(threshold),
            }
        )
        selected_parts.append(build_wait_selected(out, model_name, family, pred_col=pred_col, threshold=threshold))

    selected = pd.concat(selected_parts, ignore_index=True)
    summary = V266.summarize_selected(selected)
    threshold_audit = pd.concat(threshold_parts, ignore_index=True) if threshold_parts else pd.DataFrame()
    feature_audit = pd.DataFrame(feature_rows)
    fill_audit = pd.concat(fill_parts, ignore_index=True) if fill_parts else pd.DataFrame()
    write_csv(fill_audit, TABLES / "v269_wait_gate_fill_audit.csv")
    return selected, summary, pd.concat([feature_audit, threshold_audit], ignore_index=True, sort=False)


def run_pair_rerank(
    events: pd.DataFrame,
    cand: pd.DataFrame,
    veh_cols: List[str],
    feature_sets: Dict[str, List[str]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """对每个 v269 生理特征集运行 v267 式监督 pair reranker。"""
    lookup = V266.candidate_rmse_lookup(cand)
    train_mask = events["split"].astype(str).eq("train").to_numpy()
    veh_z, _, _, _ = V266.fit_fill_scale(events[veh_cols].to_numpy(dtype=float), train_mask)
    all_selected: List[pd.DataFrame] = []
    all_summary: List[pd.DataFrame] = []
    all_chosen: List[pd.DataFrame] = []
    all_feature_audit: List[pd.DataFrame] = []
    compact_pairs: List[pd.DataFrame] = []

    for set_name, bio_cols in feature_sets.items():
        if not bio_cols:
            continue
        print(f"[v269] pair rerank bio_set={set_name} feature_n={len(bio_cols)}", flush=True)
        bio_z, _, _, _ = V266.fit_fill_scale(events[bio_cols].to_numpy(dtype=float), train_mask)
        neighbors = V266.build_neighbor_table(events, veh_z, bio_z, train_mask, max_k=max(K_VALUES))
        pair_meta, matrices, names = V267.build_pair_dataset(events, neighbors, lookup, veh_z, bio_z, max_k=max(K_VALUES))
        pair_pred, fill_audit, feature_block = V267.add_pair_predictions(pair_meta, matrices, names)
        selected = V267.build_selected(events, pair_pred, lookup)
        summary = V267.summarize_selected(selected)
        chosen = V267.choose_val_strategies(summary)

        for df in (selected, summary, chosen, feature_block, fill_audit):
            df["bio_set"] = set_name
            df["bio_feature_n"] = int(len(bio_cols))
        all_selected.append(selected)
        all_summary.append(summary)
        all_chosen.append(chosen)
        all_feature_audit.append(feature_block)

        pred_cols = [c for c in pair_pred.columns if c.startswith("pred_pair_")]
        keep_cols = [
            "event_uid",
            "split",
            "subject",
            "prototype_event_uid",
            "prototype_subject",
            "neighbor_rank_vehicle",
            "prototype_oracle_delay_ms",
            "mapped_delay_ms",
            "target_tail_rmse_v241",
            "vehicle_distance",
            "bio_distance",
            "bad_top10",
        ] + pred_cols
        compact = pair_pred[[c for c in keep_cols if c in pair_pred.columns]].copy()
        compact.insert(0, "bio_set", set_name)
        compact_pairs.append(compact)

        fill_path = TABLES / f"v269_pair_fill_audit_{set_name}.csv"
        write_csv(fill_audit, fill_path)

    selected_all = pd.concat(all_selected, ignore_index=True) if all_selected else pd.DataFrame()
    summary_all = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
    chosen_all = pd.concat(all_chosen, ignore_index=True) if all_chosen else pd.DataFrame()
    feature_audit_all = pd.concat(all_feature_audit, ignore_index=True) if all_feature_audit else pd.DataFrame()
    pair_compact_all = pd.concat(compact_pairs, ignore_index=True) if compact_pairs else pd.DataFrame()
    return selected_all, summary_all, chosen_all, feature_audit_all, pair_compact_all


def choose_v269_pair_strategies(chosen: pd.DataFrame) -> pd.DataFrame:
    """跨 bio_set 选择 val bad_top10 最好的 vehicle_bio 策略，再映射到 test。"""
    if chosen.empty:
        return chosen
    val = chosen[
        chosen["chosen_label"].eq("val_best_pair_vehicle_bio")
        & chosen["split"].eq("val")
        & chosen["event_group"].eq("bad_top10")
    ].copy()
    rows: List[pd.Series] = []
    if not val.empty:
        best = val.sort_values(["selected_tail_rmse_mean", "selected_delay_ms_mean", "bio_set"], ascending=[True, True, True]).iloc[0]
        key_set = str(best["bio_set"])
        key_strategy = str(best["chosen_strategy"])
        rows.append(best)
        mapped = chosen[
            chosen["bio_set"].astype(str).eq(key_set)
            & chosen["chosen_strategy"].astype(str).eq(key_strategy)
            & chosen["chosen_label"].eq("val_best_pair_vehicle_bio")
            & chosen["split"].eq("test")
            & chosen["event_group"].eq("bad_top10")
        ]
        if len(mapped):
            rows.append(mapped.iloc[0])
    return pd.DataFrame(rows)


def plot_wait_summary(summary: pd.DataFrame) -> Path:
    path = FIGURES / "v269_wait_gate_test_badtop10.png"
    sub = summary[summary["split"].eq("test") & summary["event_group"].eq("bad_top10")].copy()
    if sub.empty:
        return path
    keep = sub[sub["strategy"].isin(["policy_keep_0ms_anchor", "policy_wait_to_latest_anchor", "oracle_best_anchor_upper_bound"])]
    learned = sub[sub["deployable"].astype(bool) & ~sub["strategy"].isin(keep["strategy"])].copy()
    learned = learned.sort_values("selected_tail_rmse_mean").head(8)
    focus = pd.concat([keep, learned], ignore_index=True).drop_duplicates("strategy")
    fig, ax = plt.subplots(figsize=(13, 5.2))
    x = np.arange(len(focus))
    ax.bar(x, focus["selected_tail_rmse_mean"], color="#4C78A8")
    ax.axhline(FIXED_WAIT_LATEST_BADTOP10, color="#E15759", linestyle="--", linewidth=1.2, label="fixed latest")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s).replace("_", "\n") for s in focus["strategy"]], fontsize=7)
    ax.set_ylabel("test bad_top10 selected tail RMSE")
    ax.set_title("v269: reliable / identity-removed bio wait gate")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_pair_summary(summary: pd.DataFrame, chosen: pd.DataFrame) -> Path:
    path = FIGURES / "v269_pair_reranker_test_badtop10.png"
    sub = summary[summary["split"].eq("test") & summary["event_group"].eq("bad_top10")].copy()
    if sub.empty:
        return path
    rows: List[Dict[str, object]] = []
    base = sub[sub["strategy"].isin(["policy_keep_0ms_anchor", "policy_wait_to_latest_anchor", "oracle_best_anchor_upper_bound"])]
    for _, row in base.drop_duplicates("strategy").iterrows():
        rows.append({"label": str(row["strategy"]), "rmse": float(row["selected_tail_rmse_mean"])})
    for fam in ["candidate_oracle", "vehicle_only", "vehicle_bio"]:
        one = sub[sub["strategy_family"].eq(fam)].sort_values("selected_tail_rmse_mean").head(1)
        if len(one):
            row = one.iloc[0]
            rows.append({"label": f"test-best {fam}\n{row['bio_set']} {row['strategy']}", "rmse": float(row["selected_tail_rmse_mean"])})
    cross = choose_v269_pair_strategies(chosen)
    for _, row in cross[cross["split"].eq("test")].iterrows():
        rows.append({"label": f"val-best v269 bio\n{row['bio_set']} {row['chosen_strategy']}", "rmse": float(row["selected_tail_rmse_mean"])})
    if not rows:
        return path
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(13.5, 5.2))
    x = np.arange(len(df))
    ax.bar(x, df["rmse"], color="#59A14F")
    ax.axhline(FIXED_WAIT_LATEST_BADTOP10, color="#E15759", linestyle="--", linewidth=1.2, label="fixed latest")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s).replace("_", "\n") for s in df["label"]], fontsize=7)
    ax.set_ylabel("test bad_top10 selected tail RMSE")
    ax.set_title("v269: reliable / identity-removed bio pair reranker")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def build_decision_summary(wait_summary: pd.DataFrame, pair_summary: pd.DataFrame, pair_chosen: pd.DataFrame) -> pd.DataFrame:
    """把最重要的 test bad_top10 对照收口到一张表。"""
    rows: List[Dict[str, object]] = []
    wait_bad = wait_summary[wait_summary["split"].eq("test") & wait_summary["event_group"].eq("bad_top10")].copy()
    pair_bad = pair_summary[pair_summary["split"].eq("test") & pair_summary["event_group"].eq("bad_top10")].copy()
    for strategy in ["policy_keep_0ms_anchor", "policy_wait_to_latest_anchor", "oracle_best_anchor_upper_bound"]:
        src = wait_bad[wait_bad["strategy"].eq(strategy)]
        if len(src):
            rows.append({"source": "baseline", "label": strategy, "rmse": float(src["selected_tail_rmse_mean"].iloc[0])})
    learned_wait = wait_bad[
        wait_bad["deployable"].astype(bool)
        & ~wait_bad["strategy"].isin(["policy_keep_0ms_anchor", "policy_wait_to_latest_anchor", "oracle_best_anchor_upper_bound"])
    ].sort_values("selected_tail_rmse_mean")
    if len(learned_wait):
        row = learned_wait.iloc[0]
        rows.append({"source": "wait_gate_test_best", "label": str(row["strategy"]), "rmse": float(row["selected_tail_rmse_mean"])})
    pair_oracle = pair_bad[pair_bad["strategy_family"].eq("candidate_oracle")].sort_values("selected_tail_rmse_mean")
    if len(pair_oracle):
        row = pair_oracle.iloc[0]
        rows.append({"source": "pair_candidate_oracle", "label": f"{row['bio_set']}:{row['strategy']}", "rmse": float(row["selected_tail_rmse_mean"])})
    pair_deploy = pair_bad[
        pair_bad["deployable"].astype(bool)
        & ~pair_bad["strategy_family"].isin(["baseline", "oracle", "candidate_oracle"])
    ].sort_values("selected_tail_rmse_mean")
    if len(pair_deploy):
        row = pair_deploy.iloc[0]
        rows.append({"source": "pair_test_best_deployable", "label": f"{row['bio_set']}:{row['strategy']}", "rmse": float(row["selected_tail_rmse_mean"])})
    cross = choose_v269_pair_strategies(pair_chosen)
    cross_test = cross[cross["split"].eq("test") & cross["event_group"].eq("bad_top10")]
    if len(cross_test):
        row = cross_test.iloc[0]
        rows.append({"source": "pair_val_best_vehicle_bio", "label": f"{row['bio_set']}:{row['chosen_strategy']}", "rmse": float(row["selected_tail_rmse_mean"])})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["delta_vs_fixed_latest"] = out["rmse"] - FIXED_WAIT_LATEST_BADTOP10
        out["passes_fixed_latest"] = out["rmse"] < FIXED_WAIT_LATEST_BADTOP10
    return out


def write_input_hashes() -> None:
    rows = []
    for label, path in [
        ("v260_event_biomarker_features", V260_FEATURES),
        ("v266_event_context", V266_EVENTS),
        ("v267_pair_summary", V267_SUMMARY),
        ("v267_chosen_summary", V267_CHOSEN),
        ("v266_script", V266_SCRIPT),
        ("v267_script", V267_SCRIPT),
    ]:
        rows.append({"label": label, "path": str(path), "exists": path.exists(), "sha256": file_sha256(path) if path.exists() else ""})
    write_csv(pd.DataFrame(rows), LOGS / "input_file_hashes.csv")


def write_file_inventory() -> None:
    rows = []
    if OUT.exists():
        for path in sorted(OUT.rglob("*")):
            if path.is_file():
                rows.append({"relative_path": str(path.relative_to(OUT)), "size_bytes": path.stat().st_size})
    write_csv(pd.DataFrame(rows), LOGS / "file_inventory.csv")


def make_zip() -> bool:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(OUT.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(OUT))
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        return zf.testzip() is None


def write_report(
    feature_set_audit: pd.DataFrame,
    wait_summary: pd.DataFrame,
    pair_summary: pd.DataFrame,
    pair_chosen: pd.DataFrame,
    decision: pd.DataFrame,
    figures: List[Path],
) -> None:
    lines: List[str] = []
    lines.append("# v269 reliable / identity-removed physiology")
    lines.append("")
    lines.append("## 本轮目的")
    lines.append("")
    lines.append("- v268 显示当前派生生理表征存在不可用列和强身份混淆。")
    lines.append("- v269 不继续堆模型，而是先把生理特征改成可靠、动态变化、低身份混淆的候选集合。")
    lines.append("- 然后在 wait gate 与 pair reranker 两个可部署任务上验证是否真正改善 test bad_top10。")
    lines.append("")
    lines.append("## 特征集审计")
    lines.append("")
    lines.append(
        feature_set_audit[
            [
                "bio_set",
                "feature_n",
                "delta_feature_n",
                "behavior_eta_max_mean",
                "identity_eta_max_mean",
                "identity_to_behavior_ratio_median",
                "finite_rate_train_mean",
            ]
        ].to_markdown(index=False)
    )
    lines.append("")
    lines.append("## test bad_top10 决策收口")
    lines.append("")
    if len(decision):
        lines.append(decision.to_markdown(index=False))
    else:
        lines.append("- 无可用决策结果。")
    lines.append("")
    lines.append("## wait gate 关键结果")
    lines.append("")
    wait_focus = wait_summary[wait_summary["split"].eq("test") & wait_summary["event_group"].eq("bad_top10")].copy()
    if len(wait_focus):
        wait_focus = wait_focus.sort_values("selected_tail_rmse_mean").head(12)
        cols = [
            "strategy",
            "strategy_family",
            "n",
            "selected_tail_rmse_mean",
            "delta_selected_minus_latest_mean",
            "selected_delay_ms_mean",
            "selected_latest_rate",
        ]
        lines.append(wait_focus[[c for c in cols if c in wait_focus.columns]].to_markdown(index=False))
    lines.append("")
    lines.append("## pair reranker 关键结果")
    lines.append("")
    pair_focus = pair_summary[pair_summary["split"].eq("test") & pair_summary["event_group"].eq("bad_top10")].copy()
    if len(pair_focus):
        pair_focus = pair_focus.sort_values("selected_tail_rmse_mean").head(16)
        cols = [
            "bio_set",
            "strategy",
            "strategy_family",
            "n",
            "selected_tail_rmse_mean",
            "delta_selected_minus_latest_mean",
            "selected_delay_ms_mean",
            "selected_latest_rate",
        ]
        lines.append(pair_focus[[c for c in cols if c in pair_focus.columns]].to_markdown(index=False))
    lines.append("")
    lines.append("## val 选择的 pair vehicle+bio 策略")
    lines.append("")
    cross = choose_v269_pair_strategies(pair_chosen)
    if len(cross):
        lines.append(cross.to_markdown(index=False))
    else:
        lines.append("- 没有可用的 val-best vehicle+bio pair 策略。")
    lines.append("")
    lines.append("## 判读")
    lines.append("")
    deployable_sources = ["wait_gate_test_best", "pair_val_best_vehicle_bio", "pair_test_best_deployable"]
    deploy_decision = decision[decision["source"].isin(deployable_sources)].copy() if len(decision) else pd.DataFrame()
    deploy_pass = (
        deploy_decision["passes_fixed_latest"].astype(str).str.lower().eq("true")
        if len(deploy_decision) and "passes_fixed_latest" in deploy_decision.columns
        else pd.Series(dtype=bool)
    )
    if len(deploy_decision) and bool(deploy_pass.any()):
        lines.append("- 至少一个可部署 v269 策略低于 fixed wait-latest，说明可靠/去身份化生理表征开始触及 goal。")
    else:
        lines.append("- 当前可部署 v269 策略仍未低于 fixed wait-latest，因此还不能称为差样本本质改善。")
    if len(deploy_decision):
        best = deploy_decision.sort_values("rmse").iloc[0]
        lines.append(f"- 最好可部署策略 `{best['label']}` 的 test bad_top10 RMSE 为 `{float(best['rmse']):.4f}`。")
        if abs(float(best["rmse"]) - FIXED_WAIT_LATEST_BADTOP10) < 1e-5:
            lines.append("- 这个最好策略实际退化为接近全 wait-latest，不是生理判断带来的新增收益。")
    lines.append("- 若 v269 仍失败，说明问题不是简单特征筛选，而可能需要回到原始波形事件表示、更多驾驶员内校准，或承认当前 subject-disjoint 生理增量不足。")
    lines.append("")
    lines.append("## 关键图")
    lines.append("")
    for fig in figures:
        lines.append(f"- `{fig.relative_to(OUT)}`")
    (REPORTS / "v269_reliable_identity_removed_physio_cn.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print("[v269] reliable / identity-removed physiology", flush=True)
    clean_out_dir()
    np.random.seed(SEED)

    cand, events, merge_audit, veh_cols, all_bio_cols, _v260 = load_event_sources()
    old_reference = [c for c in pd.read_csv(V266_EVENTS, encoding="utf-8-sig", nrows=0).columns if c.startswith("floor_bio260_")]
    screen = screen_features(events, all_bio_cols)
    feature_sets, feature_set_audit = choose_feature_sets(screen, old_bio_cols=None)

    # 只评估 v269 新特征集。旧 sp64 已由 v267 完整验证，报告中通过 v267 表做外部参考。
    write_csv(events[["event_uid", "split", "subject", "recording"] + veh_cols + all_bio_cols], TABLES / "v269_event_context_table.csv")
    write_csv(screen, TABLES / "v269_feature_screening_train_only.csv")
    write_csv(feature_set_audit, TABLES / "v269_feature_set_audit.csv")

    wait_selected, wait_summary, wait_audit = run_wait_gate(events, veh_cols, feature_sets)
    write_csv(wait_selected, TABLES / "v269_wait_gate_selected_by_strategy.csv")
    write_csv(wait_summary, TABLES / "v269_wait_gate_summary.csv")
    write_csv(wait_audit, TABLES / "v269_wait_gate_audit.csv")

    pair_selected, pair_summary, pair_chosen, pair_feature_audit, pair_compact = run_pair_rerank(events, cand, veh_cols, feature_sets)
    write_csv(pair_selected, TABLES / "v269_pair_selected_by_strategy.csv")
    write_csv(pair_summary, TABLES / "v269_pair_reranker_summary.csv")
    write_csv(pair_chosen, TABLES / "v269_pair_val_chosen_summary.csv")
    write_csv(pair_feature_audit, TABLES / "v269_pair_feature_block_audit.csv")
    write_csv(pair_compact, TABLES / "v269_pair_predictions_compact.csv")

    decision = build_decision_summary(wait_summary, pair_summary, pair_chosen)
    write_csv(decision, TABLES / "v269_decision_summary.csv")
    figures = [plot_wait_summary(wait_summary), plot_pair_summary(pair_summary, pair_chosen)]

    write_input_hashes()
    write_file_inventory()
    write_report(feature_set_audit, wait_summary, pair_summary, pair_chosen, decision, figures)
    write_file_inventory()
    zip_ok = make_zip()

    post_obs = bool(events["bio260_uses_post_observation"].astype(bool).any()) if "bio260_uses_post_observation" in events.columns else False
    best_deploy = decision[decision["source"].isin(["wait_gate_test_best", "pair_val_best_vehicle_bio", "pair_test_best_deployable"])].copy()
    best_rmse = float(best_deploy["rmse"].min()) if len(best_deploy) else math.nan
    guardrail = {
        "pass": bool(zip_ok and not post_obs),
        "zip_testzip": bool(zip_ok),
        "no_post_observation_physio": bool(not post_obs),
        "event_n": int(events["event_uid"].nunique()),
        "vehicle_feature_n": int(len(veh_cols)),
        "candidate_bio_feature_n": int(len(all_bio_cols)),
        "feature_set_n": int(len(feature_sets)),
        "pair_row_n": int(len(pair_compact)),
        "best_deployable_test_badtop10": best_rmse,
        "fixed_wait_latest_badtop10": float(FIXED_WAIT_LATEST_BADTOP10),
        "best_deployable_passes_fixed_latest": bool(np.isfinite(best_rmse) and best_rmse < FIXED_WAIT_LATEST_BADTOP10),
        "v267_old_floor_bio_reference_feature_n": int(len(old_reference)),
        "v260_bio260_uses_post_observation_from_merge": float(merge_audit["bio260_uses_post_observation_max"].iloc[0])
        if "bio260_uses_post_observation_max" in merge_audit.columns
        else math.nan,
    }
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    if not bool(guardrail["pass"]):
        raise AssertionError("v269 guardrail 未通过：\n" + json.dumps(guardrail, ensure_ascii=False, indent=2))
    write_file_inventory()

    print(f"[v269] report={REPORTS / 'v269_reliable_identity_removed_physio_cn.md'}", flush=True)
    print(f"[v269] zip={ZIP_PATH}", flush=True)
    if len(decision):
        print(decision.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
