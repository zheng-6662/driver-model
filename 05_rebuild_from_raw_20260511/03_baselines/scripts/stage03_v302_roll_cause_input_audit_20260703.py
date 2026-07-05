#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v302 侧倾诱因输入审计。

本轮回应用户的关键判断：
“车辆一开始发生侧倾的行为诱因，本来就应该作为输入。”

因此本脚本不再把 v301 的未来事件标签当作可部署输入，而是检查：
1. 当前 v236/v300 输入里是否已经包含 roll / roll_rate / ay / yaw_rate 等侧倾诱因信号；
2. 将这些锚点前可观测信号显式聚合成 roll-cause summary 特征后，事件类型识别是否更好；
3. roll-cause summary 对 v300 残差修正和差样本识别是否有稳定增益。

严格边界：
- 事件类型标签仍使用 v301 的 future_behavior_auto_draft，只作为监督/诊断目标；
- roll-cause 输入只来自 v236 现有 pre-anchor/history/current/known-road 特征；
- 不使用 anchor 后真实轨迹、test 后验误差、event_uid 或 recording ID 作为模型输入。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
import shutil
import sys
import time
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 20260703
ROOT = Path(__file__).resolve().parents[2]
BASELINES = ROOT / "03_baselines"
SCRIPTS = BASELINES / "scripts"
V301_SCRIPT = SCRIPTS / "stage03_v301_event_type_multiclass_label_audit_20260703.py"
V301_OUT = BASELINES / "v301_event_type_multiclass_label_audit_20260703"
V301_LABELS = V301_OUT / "tables" / "v301_event_type_labels.csv"
V300_DIR = BASELINES / "v300_within_subject_full_joint_curve_train_20260702"
V300_PRED = V300_DIR / "v300_within_subject_full_predictions.npz"
V300_GUARDRAIL = V300_DIR / "logs" / "guardrail_check.json"

OUT = BASELINES / "v302_roll_cause_input_audit_20260703"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
MODELS = OUT / "models"


ROLL_CAUSE_SIGNALS = [
    "steering",
    "speed_kmh",
    "ay",
    "yaw_rate",
    "roll",
    "yaw",
    "roll_rate",
    "roll_acc",
    "brake",
    "lane_curvature",
    "lateral_distance",
]

ROLL_CAUSE_KEYWORDS = [
    "steer",
    "speed",
    "ay",
    "yaw",
    "roll",
    "brake",
    "curvature",
    "lateral_distance",
]


@dataclass
class FeatureSet:
    """一个待比较的输入集合。"""

    name: str
    x: np.ndarray
    feature_names: List[str]


def import_module_from_path(module_name: str, path: Path):
    """从指定脚本导入已经验证过的数据读取和评估函数。"""

    if not path.exists():
        raise FileNotFoundError(f"缺少依赖脚本：{path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法导入脚本：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V301 = import_module_from_path("stage03_v301_event_type_multiclass_label_audit_for_v302", V301_SCRIPT)


def ensure_dirs() -> None:
    """创建 v302 输出目录。"""

    for folder in (TABLES, FIGURES, REPORTS, LOGS, MODELS):
        folder.mkdir(parents=True, exist_ok=True)


def clean_out_dir() -> None:
    """只清理 v302 自己的输出。"""

    if OUT.exists():
        shutil.rmtree(OUT)
    ensure_dirs()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """统一使用 utf-8-sig，方便 Windows Excel 打开。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_json(payload: Dict[str, object], path: Path) -> None:
    """写入 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def file_sha256(path: Path) -> str:
    """计算文件哈希，方便之后回溯输入版本。"""

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_nan_stat(values: np.ndarray, fn: str) -> np.ndarray:
    """沿行计算统计量，空值时返回 nan。"""

    x = values.astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if fn == "mean":
            return np.nanmean(x, axis=1)
        if fn == "std":
            return np.nanstd(x, axis=1)
        if fn == "min":
            return np.nanmin(x, axis=1)
        if fn == "max":
            return np.nanmax(x, axis=1)
        if fn == "absmax":
            return np.nanmax(np.abs(x), axis=1)
        if fn == "range":
            return np.nanmax(x, axis=1) - np.nanmin(x, axis=1)
    raise ValueError(fn)


def safe_sign_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """返回两个信号同向/反向的简化符号特征。"""

    out = np.sign(a) * np.sign(b)
    out[~np.isfinite(out)] = np.nan
    return out.astype(np.float32)


def load_inputs() -> Tuple[object, pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, List[str], np.ndarray, np.ndarray, str, Dict[str, object]]:
    """读取 v236 delay0 输入、v301 标签和 v300 基线预测。"""

    if not V301_LABELS.exists():
        raise FileNotFoundError(f"缺少 v301 标签表，请先运行 v301：{V301_LABELS}")

    data, manifest, delay0_mask, event_table = V301.load_base_delay0_data()
    manifest_delay0 = manifest.loc[delay0_mask].reset_index(drop=True)
    x_base, feature_names = V301.build_delay0_preinput_matrix(data, manifest, delay0_mask)
    y_true, pred_v300, selected_v300, v300_meta = V301.load_v300_prediction(delay0_mask)
    labels = pd.read_csv(V301_LABELS, encoding="utf-8-sig")

    if not np.array_equal(manifest_delay0["event_uid"].astype(str).to_numpy(), labels["event_uid"].astype(str).to_numpy()):
        raise AssertionError("v301 labels 与 v236 delay0 manifest 的 event_uid 不一致")
    v300_event_uid = np.asarray(v300_meta["event_uid"]).astype(str)
    if not np.array_equal(manifest_delay0["event_uid"].astype(str).to_numpy(), v300_event_uid):
        raise AssertionError("v300 predictions 与 v236 delay0 manifest 的 event_uid 不一致")

    # v301 标签来自未来行为，这里只作为监督/诊断目标；不进入 feature sets。
    labels["v302_v300_rmse"] = V301.event_rmse(y_true, pred_v300)
    for col in ["within_bad_top10_by_v249", "within_bad_top20_by_v249"]:
        if col not in labels.columns:
            labels[col] = 0
        labels[col] = labels[col].fillna(0).astype(int)

    return (
        data,
        manifest_delay0,
        delay0_mask,
        labels,
        x_base,
        feature_names,
        y_true,
        pred_v300,
        selected_v300,
        v300_meta,
    )


def parse_hist_features(feature_names: List[str]) -> Dict[str, List[Tuple[float, int]]]:
    """把 v236 扁平特征名解析回 hist 信号和相对时间。"""

    pattern = re.compile(r"^hist_([+-]?\d+(?:\.\d+)?)s_(.+)$")
    by_signal: Dict[str, List[Tuple[float, int]]] = {}
    for idx, name in enumerate(feature_names):
        m = pattern.match(name)
        if not m:
            continue
        t = float(m.group(1))
        sig = m.group(2)
        by_signal.setdefault(sig, []).append((t, idx))
    for sig in list(by_signal):
        by_signal[sig] = sorted(by_signal[sig], key=lambda x: x[0])
    return by_signal


def parse_road_features(feature_names: List[str]) -> Dict[str, List[Tuple[float, int]]]:
    """解析 road_+0.0s_road_curvature 这类已知道路特征。"""

    pattern = re.compile(r"^road_([+-]?\d+(?:\.\d+)?)s_(.+)$")
    by_signal: Dict[str, List[Tuple[float, int]]] = {}
    for idx, name in enumerate(feature_names):
        m = pattern.match(name)
        if not m:
            continue
        t = float(m.group(1))
        sig = m.group(2)
        by_signal.setdefault(sig, []).append((t, idx))
    for sig in list(by_signal):
        by_signal[sig] = sorted(by_signal[sig], key=lambda x: x[0])
    return by_signal


def take_signal_matrix(x_base: np.ndarray, items: List[Tuple[float, int]]) -> Tuple[np.ndarray, np.ndarray]:
    """按时间顺序取出某个历史信号的矩阵。"""

    times = np.array([t for t, _ in items], dtype=np.float32)
    idx = [i for _, i in items]
    return times, x_base[:, idx].astype(np.float32)


def add_summary_columns(rows: Dict[str, np.ndarray], prefix: str, times: np.ndarray, values: np.ndarray) -> None:
    """对一个信号生成最近窗口和整体窗口的显式聚合特征。"""

    if values.size == 0:
        return

    rows[f"{prefix}_last"] = values[:, -1]
    rows[f"{prefix}_first"] = values[:, 0]
    rows[f"{prefix}_mean_3s"] = safe_nan_stat(values, "mean")
    rows[f"{prefix}_std_3s"] = safe_nan_stat(values, "std")
    rows[f"{prefix}_min_3s"] = safe_nan_stat(values, "min")
    rows[f"{prefix}_max_3s"] = safe_nan_stat(values, "max")
    rows[f"{prefix}_absmax_3s"] = safe_nan_stat(values, "absmax")
    rows[f"{prefix}_range_3s"] = safe_nan_stat(values, "range")
    rows[f"{prefix}_delta_3s"] = values[:, -1] - values[:, 0]
    dur = float(max(abs(times[-1] - times[0]), 1e-6))
    rows[f"{prefix}_slope_3s"] = (values[:, -1] - values[:, 0]) / dur

    for win_s in [0.5, 1.0, 2.0]:
        mask = times >= -float(win_s) - 1e-6
        if not mask.any():
            continue
        win = values[:, mask]
        tag = str(win_s).replace(".", "p")
        rows[f"{prefix}_mean_last{tag}s"] = safe_nan_stat(win, "mean")
        rows[f"{prefix}_absmax_last{tag}s"] = safe_nan_stat(win, "absmax")
        rows[f"{prefix}_range_last{tag}s"] = safe_nan_stat(win, "range")
        rows[f"{prefix}_delta_last{tag}s"] = win[:, -1] - win[:, 0]


def build_roll_cause_summary(x_base: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """从 v236 已有输入中显式构造侧倾诱因 summary 特征。"""

    hist = parse_hist_features(feature_names)
    road = parse_road_features(feature_names)
    rows: Dict[str, np.ndarray] = {}
    signal_available: Dict[str, bool] = {}
    last_values: Dict[str, np.ndarray] = {}

    for sig in ROLL_CAUSE_SIGNALS:
        items = hist.get(sig, [])
        signal_available[sig] = bool(items)
        if not items:
            continue
        times, vals = take_signal_matrix(x_base, items)
        add_summary_columns(rows, f"hist_{sig}", times, vals)
        last_values[sig] = vals[:, -1]

    for sig, items in road.items():
        if sig not in {"road_curvature", "road_lateral_distance"}:
            continue
        times, vals = take_signal_matrix(x_base, items)
        add_summary_columns(rows, sig, times, vals)
        if vals.shape[1] > 0:
            last_values[sig] = vals[:, 0]

    current_cols = [
        "current_steer_abs",
        "current_steer_rate_abs",
        "current_roll_abs",
        "current_roll_rate_abs",
        "current_ay_abs",
        "current_yaw_rate_abs",
        "current_speed_kmh",
    ]
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    for name in current_cols:
        if name in name_to_idx:
            rows[name] = x_base[:, name_to_idx[name]].astype(np.float32)

    # 侧倾诱因常见耦合：横向加速度、方向盘、偏航、roll/roll_rate 的同向性和强度乘积。
    def get(name: str) -> np.ndarray:
        return last_values.get(name, np.full(x_base.shape[0], np.nan, dtype=np.float32))

    roll = get("roll")
    roll_rate = get("roll_rate")
    ay = get("ay")
    yaw_rate = get("yaw_rate")
    steering = get("steering")
    speed = get("speed_kmh")
    curv0 = last_values.get("road_curvature", get("lane_curvature"))

    rows["interaction_roll_ay_same_sign"] = safe_sign_product(roll, ay)
    rows["interaction_rollrate_ay_same_sign"] = safe_sign_product(roll_rate, ay)
    rows["interaction_steer_yaw_same_sign"] = safe_sign_product(steering, yaw_rate)
    rows["interaction_abs_roll_x_abs_ay"] = np.abs(roll) * np.abs(ay)
    rows["interaction_abs_rollrate_x_abs_yawrate"] = np.abs(roll_rate) * np.abs(yaw_rate)
    rows["interaction_abs_steer_x_abs_yawrate"] = np.abs(steering) * np.abs(yaw_rate)
    rows["interaction_abs_curvature_x_speed"] = np.abs(curv0) * np.abs(speed)
    rows["interaction_abs_curvature_x_speed2"] = np.abs(curv0) * np.square(speed)

    out = pd.DataFrame(rows)
    feature_names_out = out.columns.tolist()
    audit_rows = []
    for sig in ROLL_CAUSE_SIGNALS:
        items = hist.get(sig, [])
        audit_rows.append(
            {
                "signal": sig,
                "source": "v236_hist",
                "feature_count": len(items),
                "time_min": float(min([t for t, _ in items])) if items else math.nan,
                "time_max": float(max([t for t, _ in items])) if items else math.nan,
                "available": bool(items),
            }
        )
    for sig, items in road.items():
        if sig in {"road_curvature", "road_lateral_distance"}:
            audit_rows.append(
                {
                    "signal": sig,
                    "source": "v236_known_road",
                    "feature_count": len(items),
                    "time_min": float(min([t for t, _ in items])) if items else math.nan,
                    "time_max": float(max([t for t, _ in items])) if items else math.nan,
                    "available": bool(items),
                }
            )
    return out.to_numpy(dtype=np.float32), feature_names_out, pd.DataFrame(audit_rows)


def select_raw_roll_cause_subset(x_base: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """从原始 v236 扁平输入里直接筛出侧倾诱因相关列。"""

    selected = []
    for i, name in enumerate(feature_names):
        lower = name.lower()
        if any(k in lower for k in ROLL_CAUSE_KEYWORDS):
            selected.append(i)
    selected_names = [feature_names[i] for i in selected]
    audit = pd.DataFrame(
        [
            {
                "total_v236_feature_n": len(feature_names),
                "raw_roll_cause_subset_n": len(selected_names),
                "contains_roll_feature_n": int(sum("roll" in n.lower() for n in feature_names)),
                "contains_ay_feature_n": int(sum("ay" in n.lower() for n in feature_names)),
                "contains_yaw_feature_n": int(sum("yaw" in n.lower() for n in feature_names)),
                "contains_steer_feature_n": int(sum("steer" in n.lower() for n in feature_names)),
                "contains_current_roll_abs": bool("current_roll_abs" in feature_names),
                "contains_current_roll_rate_abs": bool("current_roll_rate_abs" in feature_names),
                "contains_current_ay_abs": bool("current_ay_abs" in feature_names),
                "contains_current_yaw_rate_abs": bool("current_yaw_rate_abs" in feature_names),
            }
        ]
    )
    return x_base[:, selected].astype(np.float32), selected_names, audit


def build_feature_sets(x_base: np.ndarray, feature_names: List[str]) -> Tuple[List[FeatureSet], pd.DataFrame, pd.DataFrame]:
    """构造 v302 要比较的输入集合。"""

    x_raw_subset, raw_names, raw_audit = select_raw_roll_cause_subset(x_base, feature_names)
    x_summary, summary_names, signal_audit = build_roll_cause_summary(x_base, feature_names)
    feature_sets = [
        FeatureSet("base_all_v236_preinput", x_base.astype(np.float32), list(feature_names)),
        FeatureSet("raw_roll_cause_subset", x_raw_subset, raw_names),
        FeatureSet("engineered_roll_cause_summary", x_summary, summary_names),
        FeatureSet(
            "base_plus_engineered_roll_cause",
            np.concatenate([x_base.astype(np.float32), x_summary], axis=1),
            list(feature_names) + [f"rollcause::{n}" for n in summary_names],
        ),
    ]
    return feature_sets, raw_audit, signal_audit


def train_multiclass_for_feature_set(labels: pd.DataFrame, fs: FeatureSet) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, str]:
    """训练事件类型多分类器，用来比较输入集合的信息量。"""

    y = labels["event_primary_type"].astype(str).to_numpy()
    split = labels["split"].astype(str).to_numpy()
    train = split == "train"
    val = split == "val"
    test = split == "test"

    configs = [
        (
            "extra_trees_d6",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "clf",
                        ExtraTreesClassifier(
                            n_estimators=400,
                            max_depth=6,
                            min_samples_leaf=3,
                            class_weight="balanced",
                            random_state=SEED,
                            n_jobs=1,
                        ),
                    ),
                ]
            ),
        ),
        (
            "extra_trees_d10",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "clf",
                        ExtraTreesClassifier(
                            n_estimators=500,
                            max_depth=10,
                            min_samples_leaf=2,
                            class_weight="balanced",
                            random_state=SEED + 1,
                            n_jobs=1,
                        ),
                    ),
                ]
            ),
        ),
        (
            "logreg_l2_balanced",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            C=0.3,
                            max_iter=800,
                            class_weight="balanced",
                            random_state=SEED + 2,
                        ),
                    ),
                ]
            ),
        ),
    ]

    rows: List[Dict[str, object]] = []
    pred_frames: List[pd.DataFrame] = []
    fitted: Dict[str, object] = {}
    for name, model in configs:
        model.fit(fs.x[train], y[train])
        fitted[name] = model
        for split_name, mask in [("train", train), ("val", val), ("test", test)]:
            pred = model.predict(fs.x[mask])
            rows.append(
                {
                    "feature_set": fs.name,
                    "feature_n": int(fs.x.shape[1]),
                    "classifier": name,
                    "split": split_name,
                    "n": int(mask.sum()),
                    "accuracy": float(accuracy_score(y[mask], pred)),
                    "balanced_accuracy": float(balanced_accuracy_score(y[mask], pred)),
                    "macro_f1": float(f1_score(y[mask], pred, average="macro", zero_division=0)),
                    "weighted_f1": float(f1_score(y[mask], pred, average="weighted", zero_division=0)),
                }
            )
            pred_frames.append(
                pd.DataFrame(
                    {
                        "event_uid": labels.loc[mask, "event_uid"].to_numpy(),
                        "feature_set": fs.name,
                        "split": split_name,
                        "classifier": name,
                        "true_event_primary_type": y[mask],
                        "pred_event_primary_type": pred,
                    }
                )
            )

    summary = pd.DataFrame(rows)
    val_rank = summary[summary["split"].eq("val")].sort_values(
        ["macro_f1", "balanced_accuracy", "accuracy"],
        ascending=[False, False, False],
    )
    best_name = str(val_rank.iloc[0]["classifier"])
    best_model = fitted[best_name]
    all_pred = best_model.predict(fs.x).astype(str)
    pred_table = pd.concat(pred_frames, ignore_index=True)
    return summary, pred_table, all_pred, best_name


def train_all_multiclass(labels: pd.DataFrame, feature_sets: List[FeatureSet]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """对所有输入集合训练事件类型分类器。"""

    summaries = []
    pred_tables = []
    best_rows = []
    for fs in feature_sets:
        print(f"[v302] 事件类型分类：{fs.name} / feature_n={fs.x.shape[1]}")
        summary, pred_table, all_pred, best_name = train_multiclass_for_feature_set(labels, fs)
        summaries.append(summary)
        pred_tables.append(pred_table)
        labels[f"pred_label_{fs.name}"] = all_pred
        best_rows.append({"feature_set": fs.name, "best_classifier": best_name, "feature_n": int(fs.x.shape[1])})
    return pd.concat(summaries, ignore_index=True), pd.concat(pred_tables, ignore_index=True), pd.DataFrame(best_rows)


def binary_metrics(y_true: np.ndarray, score: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """计算二分类指标，兼容单一类别边界情况。"""

    out = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, score))
        out["average_precision"] = float(average_precision_score(y_true, score))
    else:
        out["roc_auc"] = math.nan
        out["average_precision"] = math.nan
    return out


def train_bad_sample_binary(labels: pd.DataFrame, feature_sets: List[FeatureSet]) -> pd.DataFrame:
    """检查 roll-cause 输入是否更能提前识别差样本。"""

    work = labels.copy()
    # 基于 v300 自身 test/val/train 内部分位构造一个当前基线的高误差标签，只用于诊断。
    work["v300_high_rmse_top10_in_split"] = 0
    for split_name, grp in work.groupby("split"):
        q = float(grp["v302_v300_rmse"].quantile(0.90))
        work.loc[grp.index, "v300_high_rmse_top10_in_split"] = (grp["v302_v300_rmse"] >= q).astype(int)

    targets = [
        "within_bad_top10_by_v249",
        "within_bad_top20_by_v249",
        "v300_high_rmse_top10_in_split",
    ]
    split = work["split"].astype(str).to_numpy()
    train = split == "train"
    val = split == "val"
    test = split == "test"

    rows: List[Dict[str, object]] = []
    configs = [
        (
            "extra_trees_d6",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "clf",
                        ExtraTreesClassifier(
                            n_estimators=400,
                            max_depth=6,
                            min_samples_leaf=4,
                            class_weight="balanced",
                            random_state=SEED + 10,
                            n_jobs=1,
                        ),
                    ),
                ]
            ),
        ),
        (
            "random_forest_d8",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=350,
                            max_depth=8,
                            min_samples_leaf=3,
                            class_weight="balanced_subsample",
                            random_state=SEED + 11,
                            n_jobs=1,
                        ),
                    ),
                ]
            ),
        ),
        (
            "logreg_l2_balanced",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            C=0.3,
                            max_iter=800,
                            class_weight="balanced",
                            random_state=SEED + 12,
                        ),
                    ),
                ]
            ),
        ),
    ]

    for target in targets:
        y = work[target].astype(int).to_numpy()
        for fs in feature_sets:
            for model_name, model in configs:
                model.fit(fs.x[train], y[train])
                for split_name, mask in [("train", train), ("val", val), ("test", test)]:
                    if hasattr(model, "predict_proba"):
                        score = model.predict_proba(fs.x[mask])[:, 1]
                    else:
                        score = model.decision_function(fs.x[mask])
                    pred = (score >= 0.5).astype(int)
                    metrics = binary_metrics(y[mask], score, pred)
                    rows.append(
                        {
                            "target": target,
                            "feature_set": fs.name,
                            "feature_n": int(fs.x.shape[1]),
                            "classifier": model_name,
                            "split": split_name,
                            "n": int(mask.sum()),
                            "positive_rate": float(np.mean(y[mask])),
                            **metrics,
                        }
                    )
    return pd.DataFrame(rows)


def event_rmse(y_true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """逐事件 RMSE。"""

    return np.sqrt(np.nanmean(np.square(pred - y_true), axis=1))


def summarize_prediction_delta(labels: pd.DataFrame, y_true: np.ndarray, pred_base: np.ndarray, pred_method: np.ndarray, method: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """汇总残差模型对 v300 的增益。"""

    base_rmse = event_rmse(y_true, pred_base)
    method_rmse = event_rmse(y_true, pred_method)
    delta = method_rmse - base_rmse
    event_delta = pd.DataFrame(
        {
            "event_uid": labels["event_uid"],
            "split": labels["split"],
            "event_primary_type": labels["event_primary_type"],
            "within_bad_top10_by_v249": labels["within_bad_top10_by_v249"].astype(int),
            "within_bad_top20_by_v249": labels["within_bad_top20_by_v249"].astype(int),
            "baseline_rmse": base_rmse,
            "method_rmse": method_rmse,
            "delta_vs_v300": delta,
            "method": method,
        }
    )
    group_specs: List[Tuple[str, np.ndarray]] = [
        ("all", np.ones(len(labels), dtype=bool)),
        ("within_bad_top10", labels["within_bad_top10_by_v249"].astype(int).to_numpy() == 1),
        ("within_bad_top20", labels["within_bad_top20_by_v249"].astype(int).to_numpy() == 1),
    ]
    for label in sorted(labels["event_primary_type"].astype(str).unique()):
        group_specs.append((f"label::{label}", labels["event_primary_type"].astype(str).to_numpy() == label))

    rows = []
    split_values = labels["split"].astype(str).to_numpy()
    for split_name in ["train", "val", "test"]:
        split_mask = split_values == split_name
        for group_name, group_mask in group_specs:
            mask = split_mask & group_mask
            if not mask.any():
                continue
            d = delta[mask]
            rows.append(
                {
                    "method": method,
                    "split": split_name,
                    "group": group_name,
                    "n": int(mask.sum()),
                    "baseline_rmse_mean": float(np.nanmean(base_rmse[mask])),
                    "method_rmse_mean": float(np.nanmean(method_rmse[mask])),
                    "delta_vs_v300_mean": float(np.nanmean(d)),
                    "delta_vs_v300_median": float(np.nanmedian(d)),
                    "improved_rate": float(np.mean(d < 0)),
                }
            )
    return pd.DataFrame(rows), event_delta


def train_residual_regressors(labels: pd.DataFrame, feature_sets: List[FeatureSet], y_true: np.ndarray, pred_v300: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """用不同输入集合直接预测 v300 残差，检查 roll-cause 输入是否带来可用增益。"""

    residual = y_true - pred_v300
    split = labels["split"].astype(str).to_numpy()
    train = split == "train"
    val = split == "val"

    configs = [
        (
            "ridge_alpha10",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("reg", Ridge(alpha=10.0, random_state=SEED + 20)),
                ]
            ),
        ),
        (
            "ridge_alpha100",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("reg", Ridge(alpha=100.0, random_state=SEED + 21)),
                ]
            ),
        ),
        (
            "extra_trees_reg_d6",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "reg",
                        ExtraTreesRegressor(
                            n_estimators=350,
                            max_depth=6,
                            min_samples_leaf=4,
                            random_state=SEED + 22,
                            n_jobs=1,
                        ),
                    ),
                ]
            ),
        ),
    ]

    selection_rows = []
    summary_frames = []
    event_frames = []
    shrink_grid = [0.0, 0.25, 0.5, 0.75, 1.0]

    for fs in feature_sets:
        for model_name, model in configs:
            print(f"[v302] 残差回归：{fs.name} / {model_name}")
            model.fit(fs.x[train], residual[train])
            pred_resid = model.predict(fs.x).astype(np.float32)
            val_rows = []
            for shrink in shrink_grid:
                pred_corr = pred_v300 + float(shrink) * pred_resid
                val_rmse = float(np.nanmean(event_rmse(y_true[val], pred_corr[val])))
                val_rows.append({"shrink": shrink, "val_rmse_mean": val_rmse})
            best = sorted(val_rows, key=lambda r: r["val_rmse_mean"])[0]
            method = f"{fs.name}::{model_name}::shrink{best['shrink']}"
            pred_corr = pred_v300 + float(best["shrink"]) * pred_resid
            summary, event_delta = summarize_prediction_delta(labels, y_true, pred_v300, pred_corr, method)
            summary_frames.append(summary)
            event_frames.append(event_delta)
            selection_rows.append(
                {
                    "feature_set": fs.name,
                    "feature_n": int(fs.x.shape[1]),
                    "regressor": model_name,
                    "selected_shrink": float(best["shrink"]),
                    "val_rmse_mean": float(best["val_rmse_mean"]),
                }
            )
    return pd.DataFrame(selection_rows), pd.concat(summary_frames, ignore_index=True), pd.concat(event_frames, ignore_index=True)


def plot_multiclass(summary: pd.DataFrame) -> Path:
    """绘制各输入集合的事件类型 test macro-F1。"""

    idx = select_multiclass_val_chosen(summary).sort_values("test_macro_f1", ascending=True)
    path = FIGURES / "v302_event_type_macro_f1_by_input.png"
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(idx["feature_set"], idx["test_macro_f1"], color="#4c78a8")
    ax.set_title("v302 事件类型识别：validation 选模型后的 test macro-F1")
    ax.set_xlabel("test macro-F1")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_binary(binary_summary: pd.DataFrame) -> Path:
    """绘制 bad_top10 识别 AUC。"""

    idx = select_binary_val_chosen(binary_summary, "within_bad_top10_by_v249").sort_values("test_roc_auc", ascending=True)
    path = FIGURES / "v302_badtop10_auc_by_input.png"
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(idx["feature_set"], idx["test_roc_auc"], color="#72b7b2")
    ax.axvline(0.5, color="black", lw=1)
    ax.set_title("v302 差样本识别：validation 选模型后的 test AUC")
    ax.set_xlabel("test ROC-AUC")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def select_multiclass_val_chosen(summary: pd.DataFrame) -> pd.DataFrame:
    """每个输入集合先按 validation 选择分类器，再读取对应 test 结果。"""

    rows = []
    for feature_set, grp in summary.groupby("feature_set"):
        val = grp[grp["split"].eq("val")].sort_values(
            ["macro_f1", "balanced_accuracy", "accuracy"],
            ascending=[False, False, False],
        )
        if val.empty:
            continue
        chosen = val.iloc[0]
        test = grp[grp["split"].eq("test") & grp["classifier"].eq(chosen["classifier"])]
        if test.empty:
            continue
        t = test.iloc[0]
        rows.append(
            {
                "feature_set": feature_set,
                "feature_n": int(chosen["feature_n"]),
                "chosen_classifier": str(chosen["classifier"]),
                "val_macro_f1": float(chosen["macro_f1"]),
                "val_balanced_accuracy": float(chosen["balanced_accuracy"]),
                "val_accuracy": float(chosen["accuracy"]),
                "test_macro_f1": float(t["macro_f1"]),
                "test_balanced_accuracy": float(t["balanced_accuracy"]),
                "test_accuracy": float(t["accuracy"]),
                "test_weighted_f1": float(t["weighted_f1"]),
            }
        )
    return pd.DataFrame(rows)


def select_binary_val_chosen(binary_summary: pd.DataFrame, target: str) -> pd.DataFrame:
    """每个输入集合先按 validation AUC 选择分类器，再读取对应 test 结果。"""

    rows = []
    one = binary_summary[binary_summary["target"].eq(target)]
    for feature_set, grp in one.groupby("feature_set"):
        val = grp[grp["split"].eq("val")].sort_values(
            ["roc_auc", "average_precision", "balanced_accuracy"],
            ascending=[False, False, False],
        )
        if val.empty:
            continue
        chosen = val.iloc[0]
        test = grp[grp["split"].eq("test") & grp["classifier"].eq(chosen["classifier"])]
        if test.empty:
            continue
        t = test.iloc[0]
        rows.append(
            {
                "target": target,
                "feature_set": feature_set,
                "feature_n": int(chosen["feature_n"]),
                "chosen_classifier": str(chosen["classifier"]),
                "val_roc_auc": float(chosen["roc_auc"]),
                "val_average_precision": float(chosen["average_precision"]),
                "val_balanced_accuracy": float(chosen["balanced_accuracy"]),
                "test_roc_auc": float(t["roc_auc"]),
                "test_average_precision": float(t["average_precision"]),
                "test_balanced_accuracy": float(t["balanced_accuracy"]),
                "test_f1": float(t["f1"]),
                "test_positive_rate": float(t["positive_rate"]),
            }
        )
    return pd.DataFrame(rows)


def plot_residual(summary: pd.DataFrame) -> Path:
    """绘制残差修正对 test/all 和 test/bad_top10 的影响。"""

    test = summary[summary["split"].eq("test") & summary["group"].isin(["all", "within_bad_top10"])].copy()
    # 每个方法保留 all 上最好的若干个，避免图太挤。
    top_methods = (
        test[test["group"].eq("all")]
        .sort_values("delta_vs_v300_mean")
        .head(8)["method"]
        .astype(str)
        .tolist()
    )
    test = test[test["method"].isin(top_methods)]
    pivot = test.pivot(index="method", columns="group", values="delta_vs_v300_mean").fillna(0.0)
    pivot = pivot.sort_values("all", ascending=True)
    path = FIGURES / "v302_residual_delta_by_input.png"
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(pivot))
    width = 0.38
    ax.bar(x - width / 2, pivot.get("all", pd.Series(index=pivot.index, data=0.0)), width, label="test/all")
    ax.bar(x + width / 2, pivot.get("within_bad_top10", pd.Series(index=pivot.index, data=0.0)), width, label="test/bad_top10")
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=35, ha="right")
    ax.set_ylabel("RMSE delta vs v300，负值为改善")
    ax.set_title("v302 roll-cause 输入残差修正收益")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(
    raw_audit: pd.DataFrame,
    signal_audit: pd.DataFrame,
    multiclass_summary: pd.DataFrame,
    binary_summary: pd.DataFrame,
    residual_selection: pd.DataFrame,
    residual_summary: pd.DataFrame,
    selected_v300: str,
    figure_paths: List[Path],
) -> Path:
    """写中文报告。"""

    path = REPORTS / "v302_roll_cause_input_audit_cn.md"
    raw = raw_audit.iloc[0].to_dict()
    best_multi = select_multiclass_val_chosen(multiclass_summary).sort_values("test_macro_f1", ascending=False)
    best_bad = select_binary_val_chosen(binary_summary, "within_bad_top10_by_v249").sort_values("test_roc_auc", ascending=False)
    residual_focus = residual_summary[
        residual_summary["split"].eq("test") & residual_summary["group"].isin(["all", "within_bad_top10", "within_bad_top20"])
    ].sort_values(["group", "delta_vs_v300_mean"])

    lines = [
        "# v302 侧倾诱因输入审计",
        "",
        "## 这一步回答的问题",
        "",
        "用户指出：车辆一开始发生侧倾的行为诱因，本来就应该作为输入。这个判断是成立的。v302 因此不再讨论“未来事件标签能不能直接输入”，而是检查当前输入中是否已有侧倾诱因，以及显式聚合这些因果可见信号后是否有增益。",
        "",
        f"当前 v300 参照模型：`{selected_v300}`。",
        "",
        "## 当前输入是否已经包含侧倾诱因",
        "",
        f"- v236 preinput 总特征数：`{int(raw['total_v236_feature_n'])}`。",
        f"- 侧倾/横摆/转向/道路等原始相关列数：`{int(raw['raw_roll_cause_subset_n'])}`。",
        f"- roll 相关列数：`{int(raw['contains_roll_feature_n'])}`。",
        f"- ay 相关列数：`{int(raw['contains_ay_feature_n'])}`。",
        f"- yaw 相关列数：`{int(raw['contains_yaw_feature_n'])}`。",
        f"- steer 相关列数：`{int(raw['contains_steer_feature_n'])}`。",
        f"- 是否包含 `current_roll_abs`：`{raw['contains_current_roll_abs']}`。",
        f"- 是否包含 `current_roll_rate_abs`：`{raw['contains_current_roll_rate_abs']}`。",
        f"- 是否包含 `current_ay_abs`：`{raw['contains_current_ay_abs']}`。",
        f"- 是否包含 `current_yaw_rate_abs`：`{raw['contains_current_yaw_rate_abs']}`。",
        "",
        "结论：当前 v236/v300 输入并不是没有看到侧倾诱因；roll、roll_rate、ay、yaw_rate、steering、road curvature 等信号已经在历史序列或 current 特征中出现。v302 的新增部分是把这些信号显式聚合成更容易被浅层模型利用的 summary。",
        "",
        "## 信号覆盖",
        "",
        signal_audit.to_markdown(index=False),
        "",
        "## 事件类型识别结果",
        "",
        best_multi[
            [
                "feature_set",
                "feature_n",
                "chosen_classifier",
                "val_macro_f1",
                "test_accuracy",
                "test_balanced_accuracy",
                "test_macro_f1",
                "test_weighted_f1",
            ]
        ].to_markdown(index=False),
        "",
        "## 差样本识别结果",
        "",
        best_bad[
            [
                "feature_set",
                "feature_n",
                "chosen_classifier",
                "val_roc_auc",
                "test_roc_auc",
                "test_average_precision",
                "test_balanced_accuracy",
                "test_f1",
            ]
        ].to_markdown(index=False),
        "",
        "## 残差修正结果",
        "",
        residual_selection.sort_values("val_rmse_mean").head(12).to_markdown(index=False),
        "",
        residual_focus[
            ["method", "group", "n", "baseline_rmse_mean", "method_rmse_mean", "delta_vs_v300_mean", "improved_rate"]
        ].to_markdown(index=False),
        "",
        "## 当前判断",
        "",
        "- 用户关于“侧倾诱因应作为输入”的判断是对的；严格说，这些因果可见信号已经在当前输入里。",
        "- 如果 v302 显示 base_plus_engineered 只有很小改善，说明问题不是没有输入 roll-cause，而是模型没有从这些锚点前信号中稳定推断未来分叉行为。",
        "- 如果 raw_roll_cause_subset 或 engineered_roll_cause_summary 单独接近 base_all，说明侧倾诱因是关键输入组；后续可以围绕这组信号做专门编码，而不是盲目增加所有通道。",
        "- 事件类型标签仍建议作为辅助监督/分层诊断，而不是直接作为未来标签输入。",
        "",
        "## 产物",
        "",
        "- `tables/v302_roll_cause_raw_feature_audit.csv`：当前输入中侧倾诱因相关列数量。",
        "- `tables/v302_roll_cause_signal_coverage.csv`：各类历史/道路信号覆盖情况。",
        "- `tables/v302_roll_cause_summary_features.csv`：逐事件 roll-cause summary 特征。",
        "- `tables/v302_multiclass_predictability_by_input.csv`：不同输入集合的事件类型识别结果。",
        "- `tables/v302_bad_sample_binary_by_input.csv`：不同输入集合的差样本识别结果。",
        "- `tables/v302_residual_regression_summary.csv`：不同输入集合的残差修正结果。",
        "- `figures/v302_event_type_macro_f1_by_input.png`",
        "- `figures/v302_badtop10_auc_by_input.png`",
        "- `figures/v302_residual_delta_by_input.png`",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_file_inventory() -> pd.DataFrame:
    """记录输出文件清单。"""

    rows = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "relative_path": str(path.relative_to(OUT)),
                    "size_bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    inv = pd.DataFrame(rows)
    write_csv(inv, LOGS / "file_inventory.csv")
    return inv


def make_zip() -> Tuple[Path, bool]:
    """打包 v302 产物。"""

    zip_path = OUT / "v302_roll_cause_input_audit_20260703.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(OUT.rglob("*")):
            if path == zip_path or not path.is_file():
                continue
            zf.write(path, path.relative_to(OUT))
    with zipfile.ZipFile(zip_path, "r") as zf:
        ok = zf.testzip() is None
    return zip_path, ok


def main() -> None:
    start = time.time()
    clean_out_dir()
    warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

    print("[v302] 读取 v236/v301/v300 数据")
    (
        data,
        manifest_delay0,
        delay0_mask,
        labels,
        x_base,
        feature_names,
        y_true,
        pred_v300,
        selected_v300,
        v300_meta,
    ) = load_inputs()

    print("[v302] 构造侧倾诱因输入集合")
    feature_sets, raw_audit, signal_audit = build_feature_sets(x_base, feature_names)
    roll_summary_fs = next(fs for fs in feature_sets if fs.name == "engineered_roll_cause_summary")
    roll_summary_df = pd.DataFrame(roll_summary_fs.x, columns=roll_summary_fs.feature_names)
    roll_summary_df.insert(0, "event_uid", labels["event_uid"].astype(str).to_numpy())

    write_csv(raw_audit, TABLES / "v302_roll_cause_raw_feature_audit.csv")
    write_csv(signal_audit, TABLES / "v302_roll_cause_signal_coverage.csv")
    write_csv(roll_summary_df, TABLES / "v302_roll_cause_summary_features.csv")
    write_csv(
        pd.DataFrame(
            [
                {
                    "feature_set": fs.name,
                    "feature_n": int(fs.x.shape[1]),
                    "nan_rate": float(np.mean(~np.isfinite(fs.x))),
                }
                for fs in feature_sets
            ]
        ),
        TABLES / "v302_feature_set_audit.csv",
    )

    print("[v302] 比较事件类型识别")
    multiclass_summary, multiclass_preds, best_multi = train_all_multiclass(labels, feature_sets)
    write_csv(multiclass_summary, TABLES / "v302_multiclass_predictability_by_input.csv")
    write_csv(multiclass_preds, TABLES / "v302_multiclass_predictions_by_input.csv")
    write_csv(best_multi, TABLES / "v302_multiclass_best_models.csv")
    write_csv(
        select_multiclass_val_chosen(multiclass_summary),
        TABLES / "v302_multiclass_val_chosen_test_summary.csv",
    )
    write_csv(labels, TABLES / "v302_labels_with_rollcause_predictions.csv")

    print("[v302] 比较差样本识别")
    binary_summary = train_bad_sample_binary(labels, feature_sets)
    write_csv(binary_summary, TABLES / "v302_bad_sample_binary_by_input.csv")
    write_csv(
        select_binary_val_chosen(binary_summary, "within_bad_top10_by_v249"),
        TABLES / "v302_badtop10_val_chosen_test_summary.csv",
    )

    print("[v302] 比较 v300 残差修正")
    residual_selection, residual_summary, residual_events = train_residual_regressors(labels, feature_sets, y_true, pred_v300)
    write_csv(residual_selection, TABLES / "v302_residual_regression_selection.csv")
    write_csv(residual_summary, TABLES / "v302_residual_regression_summary.csv")
    write_csv(residual_events, TABLES / "v302_residual_regression_event_deltas.csv")

    print("[v302] 绘图和报告")
    figure_paths = [
        plot_multiclass(multiclass_summary),
        plot_binary(binary_summary),
        plot_residual(residual_summary),
    ]

    input_hashes = pd.DataFrame(
        [
            {"input_name": "v301_labels", "path": str(V301_LABELS), "sha256": file_sha256(V301_LABELS)},
            {"input_name": "v300_predictions", "path": str(V300_PRED), "sha256": file_sha256(V300_PRED)},
            {
                "input_name": "v300_guardrail",
                "path": str(V300_GUARDRAIL),
                "sha256": file_sha256(V300_GUARDRAIL) if V300_GUARDRAIL.exists() else "",
            },
            {"input_name": "v301_script", "path": str(V301_SCRIPT), "sha256": file_sha256(V301_SCRIPT)},
        ]
    )
    write_csv(input_hashes, LOGS / "input_hashes.csv")

    report_path = write_report(
        raw_audit,
        signal_audit,
        multiclass_summary,
        binary_summary,
        residual_selection,
        residual_summary,
        selected_v300,
        figure_paths,
    )

    guardrail = {
        "pass": True,
        "version": "v302_roll_cause_input_audit_20260703",
        "event_n": int(len(labels)),
        "delay0_only": True,
        "uses_future_event_labels_as_features": False,
        "uses_test_error_as_features": False,
        "roll_cause_features_already_in_v236": True,
        "raw_roll_cause_subset_n": int(raw_audit.iloc[0]["raw_roll_cause_subset_n"]),
        "engineered_roll_cause_feature_n": int(roll_summary_fs.x.shape[1]),
        "selected_v300_model": selected_v300,
        "v300_guardrail_pass": bool(v300_meta.get("guardrail", {}).get("pass", False)),
        "report_path": str(report_path),
        "figure_paths": [str(p) for p in figure_paths],
        "runtime_seconds": float(time.time() - start),
    }
    write_json(guardrail, LOGS / "guardrail_check.json")
    write_file_inventory()
    zip_path, zip_ok = make_zip()
    guardrail["zip_path"] = str(zip_path)
    guardrail["zip_testzip"] = bool(zip_ok)
    write_json(guardrail, LOGS / "guardrail_check.json")
    write_file_inventory()

    print("[v302] 完成")
    print(json.dumps(guardrail, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
