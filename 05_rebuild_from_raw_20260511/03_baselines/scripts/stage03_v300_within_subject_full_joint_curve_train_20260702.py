#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v300 within-subject split 下的完整 joint-curve 重训。

本脚本回答一个很明确的问题：
1. 不再用旧 v249 的预测结果做 residual 校准，也不再删除样本。
2. 复用 v299 已经生成的“同一被试内部事件级 train/val/test 划分”。
3. 从 v236 rolling 原始输入重新 fit scaler、重新训练 joint curve decoder。
4. 额外比较一个 subject_onehot 候选，用来检验“驾驶员身份/风格线索”是否能弥补锚点前车辆信息不足。

注意：
- 旧 v249 预测只作为跑完后的诊断参照，不参与训练、标准化、模型选择。
- 同一个 event_uid 的 6 个 delay 样本必须全部落在同一个 split，避免事件级泄漏。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import pickle
import re
import shutil
import sys
import time
import zipfile
from dataclasses import dataclass
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
import torch


ROOT = Path(r"F:\data_set_process\data_process")
REBUILD = ROOT / "05_rebuild_from_raw_20260511"
BASELINES = REBUILD / "03_baselines"

V242_SCRIPT = BASELINES / "scripts" / "stage03_v242_joint_curve_decoder_20260626.py"
V249_PRED = BASELINES / "v249_shape_aware_curve_model_20260630" / "v249_shape_aware_predictions.npz"
V299_DIR = BASELINES / "v299_within_subject_split_residual_calibration_20260702"
V299_EVENT_TABLE = V299_DIR / "tables" / "v299_within_subject_split_event_table.csv"
V299_GUARDRAIL = V299_DIR / "logs" / "guardrail_check.json"

OUT = BASELINES / "v300_within_subject_full_joint_curve_train_20260702"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
MODELS = OUT / "models"

SEED = 20260702
DELAY_MS = [0, 200, 400, 600, 800, 1000]

mpl.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
mpl.rcParams["axes.unicode_minus"] = False


def import_module_from_path(module_name: str, path: Path):
    """按文件路径导入已验证的前序脚本，避免复制大段模型代码。"""

    if not path.exists():
        raise FileNotFoundError(f"缺少脚本：{path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法导入脚本：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V242 = import_module_from_path("stage03_v242_joint_curve_decoder_20260626_for_v300", V242_SCRIPT)
V241 = V242.V241
V239 = V242.V239
V238 = V242.V238
FUTURE_GRID = V238.FUTURE_GRID.astype(np.float32)


@dataclass
class PreparedVariant:
    """一个输入版本对应的一整套训练数组。"""

    variant_name: str
    data: object
    point_data: object
    point_masks: Dict[str, np.ndarray]
    sample_masks: Dict[str, np.ndarray]
    scalers: object
    arrays: Dict[str, np.ndarray]
    feature_meta: Dict[str, object]


def ensure_dirs() -> None:
    """创建 v300 输出目录。"""

    for folder in (TABLES, FIGURES, REPORTS, LOGS, MODELS):
        folder.mkdir(parents=True, exist_ok=True)


def clean_out_dir() -> None:
    """只清理 v300 自己的输出目录，不触碰前序实验。"""

    if OUT.exists():
        shutil.rmtree(OUT)
    ensure_dirs()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """统一用 utf-8-sig，方便 Windows Excel 直接打开中文表。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_json(payload: Dict[str, object], path: Path) -> None:
    """保存 JSON 审计文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def file_sha256(path: Path) -> str:
    """计算输入文件哈希，便于结果追溯。"""

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def set_seed(seed: int) -> None:
    """固定 numpy / torch 随机种子。"""

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False


def clone_rolling_data(data, manifest: pd.DataFrame | None = None, x_phase: np.ndarray | None = None, feature_names=None):
    """复制 RollingData，并允许替换 manifest 或 x_phase。"""

    return V238.RollingData(
        manifest=(data.manifest.copy() if manifest is None else manifest.copy()),
        x_hist=data.x_hist.copy(),
        x_road=data.x_road.copy(),
        x_phase=(data.x_phase.copy() if x_phase is None else x_phase.astype(np.float32, copy=True)),
        y_future=data.y_future.copy(),
        pred_v236=data.pred_v236.copy(),
        feature_names=(list(data.feature_names) if feature_names is None else list(feature_names)),
        target_names=list(data.target_names),
    )


def safe_name(value: str) -> str:
    """把被试编号转成稳定的特征名片段。"""

    text = re.sub(r"[^0-9A-Za-z_]+", "_", str(value))
    return text.strip("_") or "unknown"


def apply_v299_within_subject_split(data):
    """
    把 v299 的事件级 within-subject split 映射回全部 rolling 样本。

    v299 表是一行一个 event_uid；v236 rolling 是一个 event_uid 对应 6 个 delay。
    这里要求同一 event 的 6 行必须使用同一个新 split。
    """

    if not V299_EVENT_TABLE.exists():
        raise FileNotFoundError(f"缺少 v299 事件划分表：{V299_EVENT_TABLE}")
    event_table = pd.read_csv(V299_EVENT_TABLE, encoding="utf-8-sig")
    if event_table["event_uid"].duplicated().any():
        dup = event_table.loc[event_table["event_uid"].duplicated(), "event_uid"].head(5).tolist()
        raise AssertionError(f"v299 event table 存在重复 event_uid：{dup}")
    if set(event_table["within_subject_split"].astype(str).unique()) != {"train", "val", "test"}:
        raise AssertionError("v299 within_subject_split 不是完整的 train/val/test")

    event_meta = event_table.set_index("event_uid")
    manifest = data.manifest.copy()
    manifest["original_v236_split"] = manifest["split"].astype(str)
    mapped_split = manifest["event_uid"].astype(str).map(event_meta["within_subject_split"].astype(str))
    if mapped_split.isna().any():
        missing = manifest.loc[mapped_split.isna(), "event_uid"].drop_duplicates().head(10).tolist()
        raise AssertionError(f"rolling manifest 中存在 v299 未覆盖的 event_uid：{missing}")

    manifest["within_subject_split"] = mapped_split.astype(str)
    manifest["split"] = manifest["within_subject_split"]

    # 把 v299 的诊断标签带到 rolling manifest 上，后续只用于分组评估，不用于训练。
    column_map = {
        "split_npz": "v299_original_npz_split",
        "split": "v299_original_event_split",
        "bad_top10": "v299_original_bad_top10",
        "vehicle_ambiguous": "v299_vehicle_ambiguous",
        "bad_top10_vehicle_ambiguous": "v299_bad_top10_vehicle_ambiguous",
        "candidate_pool_gain_gt_005": "v299_candidate_pool_gain_gt_005",
        "within_subject_order": "within_subject_order",
        "subject_event_n": "subject_event_n",
        "within_bad_top10_by_v249": "within_bad_top10_by_v249",
        "within_bad_top20_by_v249": "within_bad_top20_by_v249",
        "oracle_strength_label": "v299_oracle_strength_label",
        "oracle_timing_label": "v299_oracle_timing_label",
        "oracle_error_label": "v299_oracle_error_label",
        "oracle_shape_label": "v299_oracle_shape_label",
        "oracle_direction_label": "v299_oracle_direction_label",
    }
    for src, dst in column_map.items():
        if src in event_meta.columns:
            manifest[dst] = manifest["event_uid"].astype(str).map(event_meta[src])

    for col in [
        "v299_original_bad_top10",
        "v299_vehicle_ambiguous",
        "v299_bad_top10_vehicle_ambiguous",
        "v299_candidate_pool_gain_gt_005",
        "within_bad_top10_by_v249",
        "within_bad_top20_by_v249",
    ]:
        if col in manifest.columns:
            manifest[col] = manifest[col].fillna(0).astype(int)

    return clone_rolling_data(data, manifest=manifest), event_table


def build_split_audit(manifest: pd.DataFrame, event_table: pd.DataFrame) -> pd.DataFrame:
    """生成事件级和 rolling 样本级 split 防泄漏审计表。"""

    event_split_n = manifest.groupby("event_uid")["split"].nunique()
    event_delay_n = manifest.groupby("event_uid")["delay_ms"].nunique()
    rows: List[Dict[str, object]] = []
    for split_name in ["train", "val", "test"]:
        sample_mask = manifest["split"].astype(str).eq(split_name)
        event_mask = event_table["within_subject_split"].astype(str).eq(split_name)
        rows.append(
            {
                "split": split_name,
                "rolling_rows": int(sample_mask.sum()),
                "unique_events_from_rolling": int(manifest.loc[sample_mask, "event_uid"].nunique()),
                "unique_events_from_v299_table": int(event_mask.sum()),
                "unique_subjects": int(manifest.loc[sample_mask, "subject"].astype(str).nunique()),
            }
        )
    rows.append(
        {
            "split": "audit",
            "rolling_rows": int(len(manifest)),
            "unique_events_from_rolling": int(manifest["event_uid"].nunique()),
            "unique_events_from_v299_table": int(event_table["event_uid"].nunique()),
            "unique_subjects": int(manifest["subject"].astype(str).nunique()),
            "event_in_multiple_splits_n": int((event_split_n > 1).sum()),
            "event_without_6_delay_rows_n": int((event_delay_n != 6).sum()),
            "duplicate_event_delay_rows_n": int(manifest.duplicated(["event_uid", "delay_ms"]).sum()),
        }
    )
    return pd.DataFrame(rows)


def add_subject_onehot(data):
    """
    构造带被试身份的输入版本。

    这不是把 test 标签泄漏给模型，而是显式模拟“实车知道当前驾驶员是谁/已有驾驶员档案”的场景。
    模型选择仍然只看 validation，test 只用于最后报告。
    """

    subjects = sorted(data.manifest["subject"].astype(str).unique().tolist())
    subject_to_idx = {s: i for i, s in enumerate(subjects)}
    idx = data.manifest["subject"].astype(str).map(subject_to_idx).to_numpy(dtype=np.int64)
    onehot = np.zeros((len(data.manifest), len(subjects)), dtype=np.float32)
    onehot[np.arange(len(data.manifest)), idx] = 1.0
    x_phase = np.concatenate([data.x_phase.astype(np.float32), onehot], axis=1).astype(np.float32)
    subject_feature_names = [f"subject_onehot_{safe_name(s)}" for s in subjects]
    feature_names = list(data.feature_names) + subject_feature_names
    feature_meta = {
        "uses_subject_onehot": True,
        "subject_count": len(subjects),
        "subject_feature_names": subject_feature_names,
        "subject_values": subjects,
    }
    return clone_rolling_data(data, x_phase=x_phase, feature_names=feature_names), feature_meta


def prepare_variant(variant_name: str, data, feature_meta: Dict[str, object]) -> PreparedVariant:
    """为一个输入版本构造 point 数据、scaler 和神经网络数组。"""

    x_base = V238.build_base_design_matrix(data)
    point_data = V238.build_point_dataset(data, x_base)
    point_masks = V238.split_point_masks(point_data, data.manifest)
    sample_masks = V242.sample_masks(data.manifest)
    for split_name in ["train", "val", "test"]:
        if not bool(point_masks[split_name].any()):
            raise AssertionError(f"{variant_name} point-level {split_name} mask 为空")
        if not bool(sample_masks[split_name].any()):
            raise AssertionError(f"{variant_name} sample-level {split_name} mask 为空")
    scalers = V239.fit_scalers(data, point_data, point_masks)
    arrays = V239.standardize_arrays(data, point_data, scalers)
    return PreparedVariant(
        variant_name=variant_name,
        data=data,
        point_data=point_data,
        point_masks=point_masks,
        sample_masks=sample_masks,
        scalers=scalers,
        arrays=arrays,
        feature_meta=feature_meta,
    )


def load_v249_prediction_aligned(manifest: pd.DataFrame, y_true_curve: np.ndarray) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    读取旧 v249 预测并按 event_uid + delay_ms 对齐。

    旧 v249 不参与训练。这里只作为“旧路线诊断参照”，并在报告里明确标注其原始 split 暴露风险。
    """

    if not V249_PRED.exists():
        raise FileNotFoundError(f"缺少旧 v249 预测文件：{V249_PRED}")
    with np.load(V249_PRED, allow_pickle=False) as z:
        src_event = z["event_uid"].astype(str)
        src_delay = z["delay_ms"].astype(int)
        src_y = z["y_true_steering_delta"].astype(np.float32)
        src_pred = z["pred_v249_best_shape_steering_delta"].astype(np.float32)
        best_shape_model = str(z["best_shape_model"][0])
        source_split = z["split"].astype(str)

    cur_event = manifest["event_uid"].astype(str).to_numpy()
    cur_delay = manifest["delay_ms"].astype(int).to_numpy()
    if np.array_equal(src_event, cur_event) and np.array_equal(src_delay, cur_delay):
        pred = src_pred
        y_max_abs_diff = float(np.nanmax(np.abs(src_y - y_true_curve)))
        source_split_aligned = source_split
    else:
        src_index = {(e, int(d)): i for i, (e, d) in enumerate(zip(src_event, src_delay))}
        pred = np.empty((len(manifest), src_pred.shape[1]), dtype=np.float32)
        source_split_aligned = np.empty(len(manifest), dtype=object)
        src_y_aligned = np.empty_like(pred)
        for i, (e, d) in enumerate(zip(cur_event, cur_delay)):
            key = (e, int(d))
            if key not in src_index:
                raise AssertionError(f"v249 预测缺少 key：{key}")
            j = src_index[key]
            pred[i] = src_pred[j]
            src_y_aligned[i] = src_y[j]
            source_split_aligned[i] = source_split[j]
        y_max_abs_diff = float(np.nanmax(np.abs(src_y_aligned - y_true_curve)))

    if y_max_abs_diff > 1e-5:
        raise AssertionError(f"v249 y_true 与当前 y_true 不一致，max_abs_diff={y_max_abs_diff}")

    info = {
        "v249_best_shape_model": best_shape_model,
        "v249_prediction_file": str(V249_PRED),
        "v249_y_true_max_abs_diff_after_alignment": y_max_abs_diff,
        "within_test_original_v249_train_rate": float(
            np.mean(source_split_aligned[manifest["split"].astype(str).to_numpy() == "test"] == "train"
            )
        ),
        "fixed_v249_predictions_have_original_split_exposure": True,
    }
    return pred.astype(np.float32), info


def validation_score(metrics: pd.DataFrame, model_name: str) -> Dict[str, object]:
    """只用 validation original_remaining/all delays 选择候选模型。"""

    val_all = metrics[
        metrics["model_name"].eq(model_name)
        & metrics["split"].eq("val")
        & metrics["eval_mode"].eq("original_remaining")
        & metrics["bucket"].eq("all")
    ].copy()
    if val_all.empty:
        raise AssertionError(f"{model_name} 没有 validation 指标")
    val_all["score_each_delay"] = val_all.apply(V238.selection_score_from_metric, axis=1)
    weights = val_all["n_samples"].astype(float).to_numpy()
    score = float(np.average(val_all["score_each_delay"].astype(float).to_numpy(), weights=weights))
    return {
        "validation_selection_score": score,
        "val_sample_rmse_weighted": float(np.average(val_all["steer_sample_rmse_mean"], weights=weights)),
        "val_tail_rmse_weighted": float(np.average(val_all["steer_tail_rmse_mean"], weights=weights)),
        "val_strong_under_rate_weighted": float(np.average(val_all["strong_under_rate"], weights=weights)),
        "val_peak_ratio_weighted": float(np.average(val_all["peak_ratio_mean"], weights=weights)),
    }


def build_selection_table(metrics: pd.DataFrame, runs: List[Tuple[object, PreparedVariant]]) -> pd.DataFrame:
    """生成 validation-only 模型选择表。"""

    rows: List[Dict[str, object]] = []
    for run, prepared in runs:
        row = {
            "model_name": run.model_name,
            "input_variant": prepared.variant_name,
            "uses_subject_onehot": bool(prepared.feature_meta.get("uses_subject_onehot", False)),
            "test_used_for_selection": False,
            "selected_by": "validation_original_remaining_only",
            "best_epoch": int(run.best_epoch),
            "best_val_loss": float(run.best_val_loss),
            "training_seconds": float(run.training_seconds),
            "config_json": json.dumps(run.config, ensure_ascii=False, sort_keys=True),
        }
        row.update(validation_score(metrics, run.model_name))
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("validation_selection_score").reset_index(drop=True)
    out["validation_rank"] = np.arange(1, len(out) + 1)
    return out


def attach_v299_event_labels(per_sample: pd.DataFrame, event_table: pd.DataFrame) -> pd.DataFrame:
    """给 per-sample 指标附加 v299 的差样本标签。"""

    event_meta = event_table.set_index("event_uid")
    out = per_sample.copy()
    labels = {
        "within_bad_top10_by_v249": "within_bad_top10_by_v249",
        "within_bad_top20_by_v249": "within_bad_top20_by_v249",
        "bad_top10": "v299_original_bad_top10",
        "vehicle_ambiguous": "v299_vehicle_ambiguous",
        "bad_top10_vehicle_ambiguous": "v299_bad_top10_vehicle_ambiguous",
        "split_npz": "v249_original_split_for_event",
    }
    for src, dst in labels.items():
        if src in event_meta.columns:
            out[dst] = out["event_uid"].map(event_meta[src])
    for col in [
        "within_bad_top10_by_v249",
        "within_bad_top20_by_v249",
        "v299_original_bad_top10",
        "v299_vehicle_ambiguous",
        "v299_bad_top10_vehicle_ambiguous",
    ]:
        if col in out.columns:
            out[col] = out[col].fillna(0).astype(int)
    return out


def build_delay0_group_summary(per_sample: pd.DataFrame) -> pd.DataFrame:
    """按 delay0 事件和重点差样本组汇总 RMSE。"""

    delay0 = per_sample[per_sample["delay_ms"].astype(int).eq(0)].copy()
    groups = {
        "all": lambda df: np.ones(len(df), dtype=bool),
        "within_bad_top10": lambda df: df["within_bad_top10_by_v249"].astype(int).to_numpy() == 1,
        "within_bad_top20": lambda df: df["within_bad_top20_by_v249"].astype(int).to_numpy() == 1,
        "original_v249_bad_top10": lambda df: df["v299_original_bad_top10"].astype(int).to_numpy() == 1,
        "vehicle_ambiguous": lambda df: df["v299_vehicle_ambiguous"].astype(int).to_numpy() == 1,
        "strong_steer": lambda df: df["strong_steer"].astype(bool).to_numpy(),
        "normal_predictable": lambda df: ~df["observe_later_like"].astype(bool).to_numpy()
        & ~df["strong_steer"].astype(bool).to_numpy(),
    }
    rows: List[Dict[str, object]] = []
    for model_name, by_model in delay0.groupby("model_name"):
        for split_name in ["train", "val", "test"]:
            split_df = by_model[by_model["split"].astype(str).eq(split_name)].copy()
            for group_name, mask_fn in groups.items():
                mask = mask_fn(split_df)
                one = split_df.loc[mask].copy()
                if one.empty:
                    rows.append(
                        {
                            "model_name": model_name,
                            "split": split_name,
                            "group": group_name,
                            "n": 0,
                            "sample_rmse_mean": math.nan,
                            "sample_rmse_median": math.nan,
                            "sample_rmse_p90": math.nan,
                            "tail_rmse_mean": math.nan,
                            "strong_under_rate": math.nan,
                            "peak_ratio_mean": math.nan,
                            "direction_acc": math.nan,
                        }
                    )
                    continue
                rows.append(
                    {
                        "model_name": model_name,
                        "split": split_name,
                        "group": group_name,
                        "n": int(len(one)),
                        "sample_rmse_mean": float(one["sample_rmse"].mean()),
                        "sample_rmse_median": float(one["sample_rmse"].median()),
                        "sample_rmse_p90": float(one["sample_rmse"].quantile(0.90)),
                        "tail_rmse_mean": float(one["tail_rmse"].mean()),
                        "strong_under_rate": float(one["strong_under"].astype(bool).mean()),
                        "peak_ratio_mean": float(one["peak_ratio"].mean()),
                        "direction_acc": float(one["direction_ok"].astype(bool).mean()),
                    }
                )
    summary = pd.DataFrame(rows)
    ref = summary[summary["model_name"].eq("v249_existing_old_split_diagnostic")][
        ["split", "group", "sample_rmse_mean", "tail_rmse_mean"]
    ].rename(
        columns={
            "sample_rmse_mean": "v249_diagnostic_sample_rmse_mean",
            "tail_rmse_mean": "v249_diagnostic_tail_rmse_mean",
        }
    )
    summary = summary.merge(ref, on=["split", "group"], how="left")
    summary["delta_sample_rmse_mean_vs_v249_diagnostic"] = (
        summary["sample_rmse_mean"] - summary["v249_diagnostic_sample_rmse_mean"]
    )
    summary["delta_tail_rmse_mean_vs_v249_diagnostic"] = (
        summary["tail_rmse_mean"] - summary["v249_diagnostic_tail_rmse_mean"]
    )
    return summary


def build_delay0_event_wide(per_sample: pd.DataFrame, selected_name: str) -> pd.DataFrame:
    """生成 delay0 每事件宽表，便于人工看差样本。"""

    delay0 = per_sample[per_sample["delay_ms"].astype(int).eq(0)].copy()
    keys = [
        "event_uid",
        "sample_id",
        "split",
        "subject",
        "recording",
        "within_bad_top10_by_v249",
        "within_bad_top20_by_v249",
        "v299_original_bad_top10",
        "v299_vehicle_ambiguous",
        "v249_original_split_for_event",
    ]
    # V238 的 per-sample 表没有 subject/recording 时，从 sample_id/event_uid 仍可定位。
    keys = [k for k in keys if k in delay0.columns]
    metric_cols = ["sample_rmse", "tail_rmse", "peak_ratio", "strong_under", "direction_ok"]
    wide = delay0.pivot_table(index=keys, columns="model_name", values=metric_cols, aggfunc="first")
    wide.columns = [f"{metric}__{model}" for metric, model in wide.columns]
    wide = wide.reset_index()
    ref_col = "sample_rmse__v249_existing_old_split_diagnostic"
    sel_col = f"sample_rmse__{selected_name}"
    if ref_col in wide.columns and sel_col in wide.columns:
        wide[f"delta_sample_rmse__{selected_name}_minus_v249_diagnostic"] = wide[sel_col] - wide[ref_col]
    return wide.sort_values(sel_col if sel_col in wide.columns else keys[0], ascending=False).reset_index(drop=True)


def plot_training_history(runs: List[Tuple[object, PreparedVariant]]) -> Path:
    """绘制训练/验证 loss 曲线。"""

    path = FIGURES / "v300_training_history.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    for run, _prepared in runs:
        hist = run.training_history.copy()
        ax.plot(hist["epoch"], hist["val_loss"], label=f"{run.model_name} val")
        ax.plot(hist["epoch"], hist["train_loss"], linestyle="--", alpha=0.45, label=f"{run.model_name} train")
    ax.set_title("v300 完整重训 loss 曲线")
    ax.set_xlabel("epoch")
    ax.set_ylabel("masked loss")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_delay0_group_bars(summary: pd.DataFrame, selected_name: str, run_names: List[str]) -> Path:
    """绘制 test delay0 关键组 RMSE 柱状图。"""

    path = FIGURES / "v300_test_delay0_group_rmse.png"
    groups = ["all", "within_bad_top10", "within_bad_top20", "original_v249_bad_top10"]
    models = ["v249_existing_old_split_diagnostic"] + run_names
    test = summary[summary["split"].eq("test") & summary["group"].isin(groups) & summary["model_name"].isin(models)].copy()
    x = np.arange(len(groups), dtype=float)
    width = 0.80 / max(1, len(models))
    fig, ax = plt.subplots(figsize=(12, 5))
    for j, model_name in enumerate(models):
        vals = []
        for group_name in groups:
            one = test[test["group"].eq(group_name) & test["model_name"].eq(model_name)]
            vals.append(float(one["sample_rmse_mean"].iloc[0]) if not one.empty else math.nan)
        label = model_name
        if model_name == selected_name:
            label = f"{model_name}（val选择）"
        ax.bar(x + (j - (len(models) - 1) / 2) * width, vals, width=width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=15, ha="right")
    ax.set_ylabel("delay0 original_remaining RMSE")
    ax.set_title("v300 test delay0：差样本组完整重训效果")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_test_bad_top_curves(
    manifest: pd.DataFrame,
    y_true_curve: np.ndarray,
    pred_by_model: Dict[str, np.ndarray],
    per_sample: pd.DataFrame,
    selected_name: str,
    top_n: int = 6,
) -> Path:
    """绘制 selected 模型在 test delay0 中最差的若干条曲线。"""

    path = FIGURES / "v300_test_selected_bad_top6_curves.png"
    selected_per = per_sample[
        per_sample["model_name"].eq(selected_name)
        & per_sample["split"].eq("test")
        & per_sample["delay_ms"].astype(int).eq(0)
    ].copy()
    selected_per = selected_per.sort_values("sample_rmse", ascending=False).head(top_n)
    if selected_per.empty:
        return path

    event_to_row = {
        (str(row["event_uid"]), int(row["delay_ms"])): i for i, row in manifest.reset_index(drop=True).iterrows()
    }
    fig, axes = plt.subplots(len(selected_per), 1, figsize=(12, max(3.0, 2.6 * len(selected_per))), sharex=True)
    if len(selected_per) == 1:
        axes = [axes]
    for ax, (_, row) in zip(axes, selected_per.iterrows()):
        key = (str(row["event_uid"]), 0)
        idx = event_to_row[key]
        ax.plot(FUTURE_GRID, y_true_curve[idx], color="#111111", linewidth=2.0, label="真实")
        if "v249_existing_old_split_diagnostic" in pred_by_model:
            ax.plot(
                FUTURE_GRID,
                pred_by_model["v249_existing_old_split_diagnostic"][idx],
                color="#777777",
                linestyle="--",
                linewidth=1.4,
                label="旧v249诊断参照",
            )
        ax.plot(FUTURE_GRID, pred_by_model[selected_name][idx], color="#d95f02", linewidth=1.8, label="v300选择模型")
        ax.axhline(0.0, color="#bbbbbb", linewidth=0.8)
        ax.grid(True, alpha=0.22)
        ax.set_ylabel("steering_delta")
        ref_rmse = row.get("sample_rmse__v249_existing_old_split_diagnostic", math.nan)
        title = f"{row['event_uid']} | v300 RMSE={row['sample_rmse']:.3f}"
        if np.isfinite(ref_rmse):
            title += f" | old v249 RMSE={ref_rmse:.3f}"
        ax.set_title(title, fontsize=9)
    axes[0].legend(fontsize=8, ncol=3)
    axes[-1].set_xlabel("原始锚点后未来时间 / s")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(
    split_audit: pd.DataFrame,
    selection: pd.DataFrame,
    delay0_summary: pd.DataFrame,
    guardrail: Dict[str, object],
    v249_info: Dict[str, object],
    selected_name: str,
) -> Path:
    """生成中文报告。"""

    path = REPORTS / "v300_within_subject_full_joint_curve_train_cn.md"
    test_all = delay0_summary[
        delay0_summary["model_name"].eq(selected_name)
        & delay0_summary["split"].eq("test")
        & delay0_summary["group"].eq("all")
    ]
    test_bad10 = delay0_summary[
        delay0_summary["model_name"].eq(selected_name)
        & delay0_summary["split"].eq("test")
        & delay0_summary["group"].eq("within_bad_top10")
    ]
    old_all = delay0_summary[
        delay0_summary["model_name"].eq("v249_existing_old_split_diagnostic")
        & delay0_summary["split"].eq("test")
        & delay0_summary["group"].eq("all")
    ]
    old_bad10 = delay0_summary[
        delay0_summary["model_name"].eq("v249_existing_old_split_diagnostic")
        & delay0_summary["split"].eq("test")
        & delay0_summary["group"].eq("within_bad_top10")
    ]

    def metric_text(df: pd.DataFrame) -> str:
        if df.empty:
            return "NA"
        row = df.iloc[0]
        return f"n={int(row['n'])}, RMSE={float(row['sample_rmse_mean']):.4f}, tail={float(row['tail_rmse_mean']):.4f}"

    lines = [
        "# v300 within-subject 完整重训报告",
        "",
        "## 这一步做了什么",
        "",
        "v300 不是 v299 那种固定旧 v249 预测后的 residual 校准，而是把 v299 的同被试事件级划分映射回全部 rolling 样本后，从原始车辆/道路/phase 输入重新训练 joint curve decoder。",
        "",
        "训练了两类输入候选：",
        "",
        "- `no_subject`：只使用现有车辆历史、道路和 phase 输入。",
        "- `subject_onehot`：在现有输入上加入被试身份 one-hot，用来检验驾驶员身份/风格信息是否能补上锚点前车辆信息不足。",
        "",
        "旧 v249 预测只作为结果诊断参照，没有参与 scaler fit、训练、validation 选择或 test 调参。",
        "",
        "## 划分与防泄漏",
        "",
        split_audit.to_markdown(index=False),
        "",
        "核心约束：同一个 `event_uid` 的 6 个 delay 样本全部跟随同一个 `within_subject_split`。",
        "",
        "## Validation-only 选择",
        "",
        selection.to_markdown(index=False),
        "",
        f"validation 选择出的 v300 模型是：`{selected_name}`。",
        "",
        "## delay0 test 关键结果",
        "",
        f"- v300 选择模型 test/all：{metric_text(test_all)}",
        f"- 旧 v249 诊断参照 test/all：{metric_text(old_all)}",
        f"- v300 选择模型 test/within_bad_top10：{metric_text(test_bad10)}",
        f"- 旧 v249 诊断参照 test/within_bad_top10：{metric_text(old_bad10)}",
        "",
        "注意：旧 v249 在 within-subject test 中有原始 split 暴露风险，不能当作正式公平基线；它这里只用于判断旧路线预测形状和当前完整重训之间的差距。",
        "",
        "## 防线结论",
        "",
        f"- 同一事件跨 split 数：`{guardrail['event_in_multiple_splits_n']}`。",
        f"- 旧 v249 是否参与训练：`{guardrail['uses_old_v249_predictions_for_training']}`。",
        f"- 模型选择是否看 test：`{guardrail['candidate_selection_uses_test']}`。",
        f"- within test 中旧 v249 原 train 暴露比例：`{v249_info['within_test_original_v249_train_rate']:.4f}`。",
        "",
        "## 产物",
        "",
        "- `tables/v300_model_selection_validation.csv`：validation-only 选择表。",
        "- `tables/v300_metrics_by_delay_and_bucket.csv`：完整分层指标。",
        "- `tables/v300_delay0_group_summary.csv`：delay0 差样本组汇总。",
        "- `tables/v300_delay0_event_wide_comparison.csv`：每个 delay0 事件的宽表，适合人工审查差样本。",
        "- `figures/v300_test_selected_bad_top6_curves.png`：选择模型 test delay0 最差曲线。",
        "- `v300_within_subject_full_predictions.npz`：完整预测数组。",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_file_inventory() -> pd.DataFrame:
    """记录 v300 输出文件清单。"""

    rows: List[Dict[str, object]] = []
    for file in OUT.rglob("*"):
        if file.is_file():
            rows.append(
                {
                    "relative_path": str(file.relative_to(OUT)),
                    "bytes": int(file.stat().st_size),
                    "sha256": file_sha256(file),
                }
            )
    inventory = pd.DataFrame(rows).sort_values("relative_path").reset_index(drop=True)
    write_csv(inventory, LOGS / "file_inventory.csv")
    return inventory


def make_zip_package() -> Tuple[Path, bool]:
    """打包 v300 产物并做 zip 完整性检查。"""

    zip_path = OUT / "v300_within_subject_full_joint_curve_train_20260702.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file in OUT.rglob("*"):
            if file.is_file() and file != zip_path:
                zf.write(file, file.relative_to(OUT))
    with zipfile.ZipFile(zip_path, "r") as zf:
        ok = zf.testzip() is None
    return zip_path, bool(ok)


def main() -> None:
    start_time = time.time()
    clean_out_dir()
    set_seed(SEED)
    V242.SEED = SEED
    torch.set_num_threads(1)

    print("[v300] 读取 v236 rolling 数据，并套用 v299 within-subject 事件级 split")
    raw_data = V238.load_v236_data()
    data, event_table = apply_v299_within_subject_split(raw_data)
    split_audit = build_split_audit(data.manifest, event_table)
    write_csv(split_audit, TABLES / "v300_within_subject_split_audit.csv")

    task_table, point_counts = V238.build_task_construction_tables(data, V238.build_point_dataset(data, V238.build_base_design_matrix(data)))
    write_csv(task_table, TABLES / "v300_task_construction_audit.csv")
    write_csv(point_counts, TABLES / "v300_point_training_rows_by_delay.csv")

    input_hashes = pd.DataFrame(
        [
            {"input_name": "v242_script", "path": str(V242_SCRIPT), "sha256": file_sha256(V242_SCRIPT)},
            {"input_name": "v249_predictions_diagnostic_only", "path": str(V249_PRED), "sha256": file_sha256(V249_PRED)},
            {"input_name": "v299_event_split_table", "path": str(V299_EVENT_TABLE), "sha256": file_sha256(V299_EVENT_TABLE)},
            {
                "input_name": "v299_guardrail",
                "path": str(V299_GUARDRAIL),
                "sha256": file_sha256(V299_GUARDRAIL) if V299_GUARDRAIL.exists() else "",
            },
        ]
    )
    write_csv(input_hashes, LOGS / "input_hashes.csv")

    print("[v300] 准备 no_subject 与 subject_onehot 两个输入版本")
    no_subject = prepare_variant(
        "no_subject",
        data,
        {
            "uses_subject_onehot": False,
            "subject_count": int(data.manifest["subject"].astype(str).nunique()),
            "description_cn": "仅使用现有车辆历史、道路和 phase 输入",
        },
    )
    subject_data, subject_meta = add_subject_onehot(data)
    subject_onehot = prepare_variant("subject_onehot", subject_data, subject_meta)
    variants = {v.variant_name: v for v in [no_subject, subject_onehot]}

    variant_rows = []
    for prepared in variants.values():
        variant_rows.append(
            {
                "input_variant": prepared.variant_name,
                "uses_subject_onehot": bool(prepared.feature_meta.get("uses_subject_onehot", False)),
                "x_hist_shape": str(tuple(prepared.data.x_hist.shape)),
                "x_road_shape": str(tuple(prepared.data.x_road.shape)),
                "x_phase_shape": str(tuple(prepared.data.x_phase.shape)),
                "feature_name_count": len(prepared.data.feature_names),
                "subject_count": int(prepared.feature_meta.get("subject_count", 0)),
                "feature_meta_json": json.dumps(prepared.feature_meta, ensure_ascii=False),
            }
        )
    write_csv(pd.DataFrame(variant_rows), TABLES / "v300_input_variant_audit.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[v300] 使用设备：{device}")

    configs: List[Tuple[str, str, Dict[str, object]]] = [
        (
            "v300_full_joint_h64_no_subject",
            "no_subject",
            {
                "hidden_dim": 64,
                "n_heads": 4,
                "n_layers": 3,
                "mixer_layers": 2,
                "mlp_hidden": 96,
                "dropout": 0.08,
                "smooth_weight": 0.02,
                "lr": 6e-4,
                "min_lr": 1e-5,
                "weight_decay": 3e-4,
                "batch_size": 384,
                "max_epochs": 80,
                "patience": 12,
            },
        ),
        (
            "v300_full_joint_h64_subject_onehot",
            "subject_onehot",
            {
                "hidden_dim": 64,
                "n_heads": 4,
                "n_layers": 3,
                "mixer_layers": 2,
                "mlp_hidden": 112,
                "dropout": 0.08,
                "smooth_weight": 0.02,
                "lr": 6e-4,
                "min_lr": 1e-5,
                "weight_decay": 3e-4,
                "batch_size": 384,
                "max_epochs": 80,
                "patience": 12,
            },
        ),
        (
            "v300_full_joint_h96_subject_onehot",
            "subject_onehot",
            {
                "hidden_dim": 96,
                "n_heads": 4,
                "n_layers": 4,
                "mixer_layers": 3,
                "mlp_hidden": 144,
                "dropout": 0.11,
                "smooth_weight": 0.04,
                "lr": 5e-4,
                "min_lr": 1e-5,
                "weight_decay": 5e-4,
                "batch_size": 256,
                "max_epochs": 90,
                "patience": 14,
            },
        ),
    ]

    runs: List[Tuple[object, PreparedVariant]] = []
    for run_idx, (model_name, variant_name, config) in enumerate(configs):
        prepared = variants[variant_name]
        run_seed = SEED + run_idx
        set_seed(run_seed)
        V242.SEED = run_seed
        print(f"[v300] training {model_name} | variant={variant_name} | seed={run_seed}")
        run = V242.train_joint_candidate(
            model_name=model_name,
            config=config,
            data=prepared.data,
            point_data=prepared.point_data,
            arrays=prepared.arrays,
            scalers=prepared.scalers,
            masks=prepared.sample_masks,
            device=device,
        )
        runs.append((run, prepared))
        write_csv(run.training_history, TABLES / f"{model_name}_training_history.csv")
        torch.save(
            {
                "model_name": run.model_name,
                "state_dict": run.state_dict,
                "config": run.config,
                "input_variant": prepared.variant_name,
                "feature_meta": prepared.feature_meta,
                "best_epoch": run.best_epoch,
                "best_val_loss": run.best_val_loss,
                "training_seconds": run.training_seconds,
                "seed": run_seed,
            },
            MODELS / f"{model_name}.pt",
        )
        print(f"[v300] {model_name} best_epoch={run.best_epoch} best_val_loss={run.best_val_loss:.6f}")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("[v300] 计算完整分层指标与 validation-only 选择")
    y_true_curve = data.y_future[:, :, 0].astype(np.float32)
    pred_v249, v249_info = load_v249_prediction_aligned(data.manifest, y_true_curve)
    pred_by_model: Dict[str, np.ndarray] = {
        "v236_existing_old_split_diagnostic": data.pred_v236[:, :, 0].astype(np.float32),
        "v249_existing_old_split_diagnostic": pred_v249.astype(np.float32),
    }
    for run, _prepared in runs:
        pred_by_model[run.model_name] = run.pred_curve.astype(np.float32)

    metrics = V238.compute_metrics_table(
        y_true_curve=y_true_curve,
        pred_by_model=pred_by_model,
        manifest=data.manifest,
        eval_modes=["original_remaining", "receding_2s_diagnostic"],
    )
    write_csv(metrics, TABLES / "v300_metrics_by_delay_and_bucket.csv")

    selection = build_selection_table(metrics, runs)
    selected_name = str(selection.iloc[0]["model_name"])
    write_csv(selection, TABLES / "v300_model_selection_validation.csv")

    per_tables = []
    for model_name, pred_curve in pred_by_model.items():
        per = V238.build_per_sample_metrics(
            y_true_curve=y_true_curve,
            pred_curve=pred_curve,
            manifest=data.manifest,
            model_name=model_name,
        )
        per_tables.append(per)
    per_sample = pd.concat(per_tables, ignore_index=True)
    per_sample = attach_v299_event_labels(per_sample, event_table)
    write_csv(per_sample, TABLES / "v300_per_sample_metrics_original_remaining.csv")

    delay0_summary = build_delay0_group_summary(per_sample)
    write_csv(delay0_summary, TABLES / "v300_delay0_group_summary.csv")
    delay0_wide = build_delay0_event_wide(per_sample, selected_name)
    write_csv(delay0_wide, TABLES / "v300_delay0_event_wide_comparison.csv")

    print("[v300] 保存预测数组、模型选择和图像")
    original_remaining_valid, _ = V238.build_original_remaining_mask(data.manifest)
    npz_payload = {
        "y_true_steering_delta": y_true_curve.astype(np.float32),
        "pred_v236_existing_old_split_diagnostic": pred_by_model["v236_existing_old_split_diagnostic"].astype(np.float32),
        "pred_v249_existing_old_split_diagnostic": pred_by_model["v249_existing_old_split_diagnostic"].astype(np.float32),
        "pred_v300_best_within_subject_full": pred_by_model[selected_name].astype(np.float32),
        "best_v300_model": np.array([selected_name]),
        "delay_ms": data.manifest["delay_ms"].astype(int).to_numpy(dtype=np.int32),
        "split": data.manifest["split"].astype(str).to_numpy(),
        "original_v236_split": data.manifest["original_v236_split"].astype(str).to_numpy(),
        "event_uid": data.manifest["event_uid"].astype(str).to_numpy(),
        "subject": data.manifest["subject"].astype(str).to_numpy(),
        "future_grid_s": FUTURE_GRID.astype(np.float32),
        "original_remaining_valid": original_remaining_valid.astype(bool),
    }
    for run, _prepared in runs:
        npz_payload[f"pred_{run.model_name}"] = run.pred_curve.astype(np.float32)
    np.savez_compressed(OUT / "v300_within_subject_full_predictions.npz", **npz_payload)

    with (MODELS / "v300_scalers_and_selection.pkl").open("wb") as f:
        pickle.dump(
            {
                "selection": selection.to_dict(orient="records"),
                "selected_name": selected_name,
                "variant_meta": {name: prepared.feature_meta for name, prepared in variants.items()},
                "scalers": {name: prepared.scalers for name, prepared in variants.items()},
                "v249_info": v249_info,
            },
            f,
        )

    figure_paths = [
        plot_training_history(runs),
        plot_delay0_group_bars(delay0_summary, selected_name, [run.model_name for run, _ in runs]),
        plot_test_bad_top_curves(data.manifest, y_true_curve, pred_by_model, per_sample, selected_name, top_n=6),
    ]

    event_split_n = data.manifest.groupby("event_uid")["split"].nunique()
    event_delay_n = data.manifest.groupby("event_uid")["delay_ms"].nunique()
    guardrail = {
        "pass": bool((event_split_n <= 1).all() and (event_delay_n == 6).all()),
        "version": "v300_within_subject_full_joint_curve_train_20260702",
        "split_method": "v299_within_subject_random_event_split_60_20_20_mapped_to_all_rolling_delays",
        "same_event_never_repeated_across_splits": bool((event_split_n <= 1).all()),
        "event_in_multiple_splits_n": int((event_split_n > 1).sum()),
        "event_without_6_delay_rows_n": int((event_delay_n != 6).sum()),
        "duplicate_event_delay_rows_n": int(data.manifest.duplicated(["event_uid", "delay_ms"]).sum()),
        "event_n": int(data.manifest["event_uid"].nunique()),
        "rolling_sample_n": int(len(data.manifest)),
        "subject_n": int(data.manifest["subject"].astype(str).nunique()),
        "train_rolling_n": int(data.manifest["split"].eq("train").sum()),
        "val_rolling_n": int(data.manifest["split"].eq("val").sum()),
        "test_rolling_n": int(data.manifest["split"].eq("test").sum()),
        "train_event_n": int(event_table["within_subject_split"].eq("train").sum()),
        "val_event_n": int(event_table["within_subject_split"].eq("val").sum()),
        "test_event_n": int(event_table["within_subject_split"].eq("test").sum()),
        "full_model_retrained_from_rolling_inputs": True,
        "uses_old_v249_predictions_for_training": False,
        "uses_old_v249_predictions_for_selection": False,
        "candidate_selection_uses_test": False,
        "old_v249_predictions_diagnostic_only": True,
        "within_test_original_v249_train_rate": v249_info["within_test_original_v249_train_rate"],
        "selected_model": selected_name,
        "selected_model_uses_subject_onehot": bool(
            selection.loc[selection["model_name"].eq(selected_name), "uses_subject_onehot"].iloc[0]
        ),
        "device": str(device),
        "runtime_seconds": float(time.time() - start_time),
        "figure_paths": [str(p) for p in figure_paths],
    }
    write_json(guardrail, LOGS / "guardrail_check.json")
    write_json(v249_info, LOGS / "v249_diagnostic_reference_info.json")
    write_report(split_audit, selection, delay0_summary, guardrail, v249_info, selected_name)

    write_file_inventory()
    zip_path, zip_ok = make_zip_package()
    guardrail["zip_path"] = str(zip_path)
    guardrail["zip_testzip"] = bool(zip_ok)
    write_json(guardrail, LOGS / "guardrail_check.json")
    write_file_inventory()

    print("[v300] 完成")
    print(json.dumps(guardrail, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
