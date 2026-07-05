#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v240 锁定 attention 候选审查包。

本轮不训练模型、不改配置、不做 test-based retuning。

目标：
1. 固定读取 v239_light_attention_h32 的预测与模型权重；
2. 逐样本比较 v236 / v238 / v239，找出改善样本与退化样本；
3. 对 normal_predictable 做更细 no-harm 子桶审查；
4. 对 strong_steer 的 400ms / 1000ms 退化样本做专门审查；
5. 抽取 attention 权重，检查模型是否在合理时间段关注历史与道路预瞄；
6. 形成可交给用户/GPTPro/导师继续判断的中文锁定报告。
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
import torch


ROOT = Path(r"F:\data_set_process\data_process")
REBUILD = ROOT / "05_rebuild_from_raw_20260511"
BASELINES = REBUILD / "03_baselines"

V239_SCRIPT = BASELINES / "scripts" / "stage03_v239_light_attention_noharm_20260626.py"
V239_DIR = BASELINES / "v239_light_attention_noharm_20260626"
V239_PRED = V239_DIR / "v239_light_attention_predictions.npz"
V239_MODEL = V239_DIR / "models" / "v239_best_light_attention_diagnostic.pt"
V239_SELECTION = V239_DIR / "tables" / "v239_model_selection_validation_noharm.csv"
V239_DECISION = V239_DIR / "tables" / "v239_next_model_decision.csv"

OUT = BASELINES / "v240_locked_attention_audit_20260626"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
CASE_FIGURES = FIGURES / "attention_casebook"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"

DELAY_MS = [0, 200, 400, 600, 800, 1000]
SEED = 240

mpl.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
mpl.rcParams["axes.unicode_minus"] = False


def import_v239_module():
    """导入 v239 脚本，复用模型结构和 v238 任务工具。"""

    if not V239_SCRIPT.exists():
        raise FileNotFoundError(f"找不到 v239 脚本：{V239_SCRIPT}")
    spec = importlib.util.spec_from_file_location("stage03_v239_light_attention_noharm_20260626", V239_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法导入 v239 脚本：{V239_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V239 = import_v239_module()
V238 = V239.V238
FUTURE_GRID = V238.FUTURE_GRID
HISTORY_GRID = np.round(np.arange(-3.0, 0.0 + 1e-9, 0.1), 4)


def ensure_dirs() -> None:
    """创建输出目录。"""

    for folder in (TABLES, CASE_FIGURES, REPORTS, LOGS):
        folder.mkdir(parents=True, exist_ok=True)


def clean_out_dir() -> None:
    """只清理 v240 自己的输出目录。"""

    if OUT.exists():
        shutil.rmtree(OUT)
    ensure_dirs()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """CSV 统一用 utf-8-sig 输出。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def file_sha256(path: Path) -> str:
    """计算文件 SHA256。"""

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_name(text: str, max_len: int = 70) -> str:
    """生成适合文件名的短字符串。"""

    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))
    return cleaned[:max_len].strip("_") or "case"


def load_locked_inputs():
    """读取 v236/v238/v239 的锁定输入、预测、attention 模型和标准化参数。"""

    required = [V239_PRED, V239_MODEL, V239_SELECTION, V239_DECISION]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("v240 缺少必要输入：\n" + "\n".join(missing))

    selection = pd.read_csv(V239_SELECTION, encoding="utf-8-sig")
    best_name = str(selection.sort_values("validation_rank").iloc[0]["model_name"])
    decision = pd.read_csv(V239_DECISION, encoding="utf-8-sig")

    data = V238.load_v236_data()
    x_base = V238.build_base_design_matrix(data)
    point_data = V238.build_point_dataset(data, x_base)

    with np.load(V239_PRED, allow_pickle=False) as pred:
        y_true = pred["y_true_steering_delta"].astype(np.float32)
        pred_v236 = pred["pred_v236_steering_delta"].astype(np.float32)
        pred_v238 = pred["pred_v238_steering_delta"].astype(np.float32)
        pred_v239 = pred["pred_v239_best_attention_steering_delta"].astype(np.float32)
        best_from_npz = str(pred["best_attention_model"][0])
    if best_from_npz != best_name:
        raise AssertionError(f"v239 prediction best model 与 selection 不一致：{best_from_npz} vs {best_name}")

    payload = torch.load(V239_MODEL, map_location="cpu", weights_only=False)
    if str(payload["model_name"]) != best_name:
        raise AssertionError(f"v239 model payload 与 selection 不一致：{payload['model_name']} vs {best_name}")
    config = payload["config"]
    scalers = V239.SequenceScalers(**payload["scalers"])
    arrays = V239.standardize_arrays(data, point_data, scalers)

    model = V239.LightTemporalAttention(
        hist_dim=data.x_hist.shape[-1],
        road_dim=data.x_road.shape[-1],
        phase_dim=data.x_phase.shape[-1],
        point_dim=len(V238.POINT_EXTRA_FEATURE_NAMES),
        hidden_dim=int(config["hidden_dim"]),
        head_dim=int(config["head_dim"]),
        dropout=float(config["dropout"]),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()

    return {
        "selection": selection,
        "decision": decision,
        "best_name": best_name,
        "data": data,
        "point_data": point_data,
        "arrays": arrays,
        "model": model,
        "y_true": y_true,
        "pred_v236": pred_v236,
        "pred_v238": pred_v238,
        "pred_v239": pred_v239,
    }


def merge_per_sample_metrics(data, y_true: np.ndarray, pred_v236: np.ndarray, pred_v238: np.ndarray, pred_v239: np.ndarray) -> pd.DataFrame:
    """生成逐 rolling sample 的 v236/v238/v239 对照表。"""

    manifest = data.manifest.reset_index(drop=True).copy()
    base_cols = [
        "event_uid",
        "sample_id",
        "split",
        "delay_ms",
        "observe_later_like",
        "strong_steer",
        "normal_curve",
        "reverse",
        "zero_cross",
        "multi_correction",
        "extreme_peak",
        "high_tail_error",
        "strict_subset",
        "route_event",
        "scene_type",
        "original_anchor_s",
        "observation_s",
    ]
    available = [c for c in base_cols if c in manifest.columns]
    out = manifest[available].copy()
    out.insert(0, "sample_index", np.arange(len(out), dtype=int))
    out["normal_predictable"] = out["normal_curve"].astype(bool) & ~out["observe_later_like"].astype(bool)
    out["reverse_or_multi_correction"] = (
        out["reverse"].astype(bool) | out["zero_cross"].astype(bool) | out["multi_correction"].astype(bool)
    )

    for model_name, pred in [
        ("v236", pred_v236),
        ("v238", pred_v238),
        ("v239", pred_v239),
    ]:
        metrics = V238.build_per_sample_metrics(y_true, pred, manifest, model_name)
        out[f"{model_name}_sample_rmse"] = metrics["sample_rmse"].astype(float)
        out[f"{model_name}_tail_rmse"] = metrics["tail_rmse"].astype(float)
        out[f"{model_name}_true_peak_abs"] = metrics["true_peak_abs"].astype(float)
        out[f"{model_name}_pred_peak_abs"] = metrics["pred_peak_abs"].astype(float)
        out[f"{model_name}_peak_ratio"] = metrics["peak_ratio"].astype(float)
        out[f"{model_name}_true_peak_t"] = metrics["true_peak_t"].astype(float)
        out[f"{model_name}_pred_peak_t"] = metrics["pred_peak_t"].astype(float)
        out[f"{model_name}_direction_ok"] = metrics["direction_ok"].astype(bool)
        out[f"{model_name}_strong_under"] = metrics["strong_under"].astype(bool)

    for ref in ["v236", "v238"]:
        out[f"delta_sample_v239_minus_{ref}"] = out["v239_sample_rmse"] - out[f"{ref}_sample_rmse"]
        out[f"delta_tail_v239_minus_{ref}"] = out["v239_tail_rmse"] - out[f"{ref}_tail_rmse"]
        out[f"delta_peak_ratio_v239_minus_{ref}"] = out["v239_peak_ratio"] - out[f"{ref}_peak_ratio"]
    return out


def aggregate_mask(df: pd.DataFrame, mask: np.ndarray, split: str, bucket: str, delay_ms: int) -> Dict[str, object] | None:
    """聚合一个 split/bucket/delay 的逐样本指标。"""

    one = df[mask].copy()
    if one.empty:
        return None
    return {
        "split": split,
        "bucket": bucket,
        "delay_ms": int(delay_ms),
        "n": int(len(one)),
        "v236_sample_rmse_mean": float(one["v236_sample_rmse"].mean()),
        "v239_sample_rmse_mean": float(one["v239_sample_rmse"].mean()),
        "delta_sample_v239_minus_v236": float(one["delta_sample_v239_minus_v236"].mean()),
        "v236_tail_rmse_mean": float(one["v236_tail_rmse"].mean()),
        "v239_tail_rmse_mean": float(one["v239_tail_rmse"].mean()),
        "delta_tail_v239_minus_v236": float(one["delta_tail_v239_minus_v236"].mean()),
        "v236_peak_ratio_mean": float(one["v236_peak_ratio"].mean()),
        "v239_peak_ratio_mean": float(one["v239_peak_ratio"].mean()),
        "delta_peak_ratio_v239_minus_v236": float(one["delta_peak_ratio_v239_minus_v236"].mean()),
        "regression_rate_tail_gt_0": float((one["delta_tail_v239_minus_v236"] > 0).mean()),
        "improvement_rate_tail_lt_0": float((one["delta_tail_v239_minus_v236"] < 0).mean()),
    }


def build_subbucket_summary(per_sample: pd.DataFrame) -> pd.DataFrame:
    """生成 test split 的核心子桶 no-harm 审查表。"""

    buckets = {
        "all": np.ones(len(per_sample), dtype=bool),
        "observe_later_like": per_sample["observe_later_like"].astype(bool).to_numpy(),
        "strong_steer": per_sample["strong_steer"].astype(bool).to_numpy(),
        "normal_predictable": per_sample["normal_predictable"].astype(bool).to_numpy(),
        "reverse_or_multi_correction": per_sample["reverse_or_multi_correction"].astype(bool).to_numpy(),
        "reverse": per_sample["reverse"].astype(bool).to_numpy(),
        "zero_cross": per_sample["zero_cross"].astype(bool).to_numpy(),
        "multi_correction": per_sample["multi_correction"].astype(bool).to_numpy(),
        "extreme_peak": per_sample["extreme_peak"].astype(bool).to_numpy(),
        "high_tail_error": per_sample["high_tail_error"].astype(bool).to_numpy(),
        "strict_subset": per_sample["strict_subset"].astype(bool).to_numpy(),
    }
    rows: List[Dict[str, object]] = []
    split_values = per_sample["split"].astype(str).to_numpy()
    delay_values = per_sample["delay_ms"].astype(int).to_numpy()
    for split in ["val", "test"]:
        for delay_ms in DELAY_MS:
            split_delay = (split_values == split) & (delay_values == delay_ms)
            for bucket, bucket_mask in buckets.items():
                item = aggregate_mask(per_sample, split_delay & bucket_mask, split, bucket, delay_ms)
                if item is not None:
                    item["noharm_sample_pass"] = bool(item["delta_sample_v239_minus_v236"] <= 0.02)
                    item["noharm_tail_pass"] = bool(item["delta_tail_v239_minus_v236"] <= 0.02)
                    item["noharm_pass"] = bool(item["noharm_sample_pass"] and item["noharm_tail_pass"])
                    rows.append(item)
    return pd.DataFrame(rows)


def build_case_tables(per_sample: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """抽取改善、退化和重点审查样本表。"""

    test = per_sample[per_sample["split"].astype(str).eq("test")].copy()
    tables = {
        "top_observe_later_improvements": test[test["observe_later_like"].astype(bool)]
        .sort_values("delta_tail_v239_minus_v236")
        .head(15),
        "top_normal_improvements": test[test["normal_predictable"].astype(bool)]
        .sort_values("delta_tail_v239_minus_v236")
        .head(15),
        "worst_regressions": test.sort_values("delta_tail_v239_minus_v236", ascending=False).head(20),
        "worst_v239_residuals": test.sort_values("v239_tail_rmse", ascending=False).head(20),
        "strong_400_1000_regressions": test[
            test["strong_steer"].astype(bool)
            & test["delay_ms"].astype(int).isin([400, 1000])
            & (test["delta_tail_v239_minus_v236"] > 0)
        ]
        .sort_values("delta_tail_v239_minus_v236", ascending=False)
        .head(20),
    }
    return tables


def point_index_for_sample_time(point_data, sample_index: int, time_index: int) -> int:
    """根据 v238 point-level 展开顺序找到 point index。"""

    idx = int(sample_index) * len(FUTURE_GRID) + int(time_index)
    if int(point_data.sample_index_all[idx]) != int(sample_index) or int(point_data.time_index_all[idx]) != int(time_index):
        raise AssertionError("point index 映射与 v238 展开顺序不一致")
    return idx


def attention_for_case(model, arrays: Dict[str, np.ndarray], point_data, sample_index: int, time_index: int) -> Tuple[np.ndarray, np.ndarray]:
    """抽取一个样本在指定 future point 上的历史/道路 attention 权重。"""

    point_idx = point_index_for_sample_time(point_data, sample_index, time_index)
    sidx = int(sample_index)
    with torch.no_grad():
        hist = torch.from_numpy(arrays["hist"][sidx : sidx + 1]).float()
        road = torch.from_numpy(arrays["road"][sidx : sidx + 1]).float()
        phase = torch.from_numpy(arrays["phase"][sidx : sidx + 1]).float()
        point = torch.from_numpy(arrays["point"][point_idx : point_idx + 1]).float()
        hist_emb = model.hist_proj(hist)
        road_emb = model.road_proj(road)
        query = model.query(torch.cat([phase, point], dim=1))
        _, hist_weight = model.attend(hist_emb, query, model.hist_score)
        _, road_weight = model.attend(road_emb, query, model.road_score)
    return hist_weight.squeeze(0).numpy(), road_weight.squeeze(0).numpy()


def entropy(weights: np.ndarray) -> float:
    """标准化熵，越低表示注意力越集中。"""

    w = np.asarray(weights, dtype=float)
    w = w / max(float(w.sum()), 1e-12)
    ent = -float(np.sum(w * np.log(np.maximum(w, 1e-12))))
    return ent / math.log(len(w))


def choose_attention_cases(case_tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """选择生成 attention casebook 图的代表样本。"""

    specs = [
        ("observe_later_top_improve", case_tables["top_observe_later_improvements"].head(6)),
        ("normal_top_improve", case_tables["top_normal_improvements"].head(4)),
        ("strong_regression_400_1000", case_tables["strong_400_1000_regressions"].head(6)),
        ("worst_v239_residual", case_tables["worst_v239_residuals"].head(6)),
    ]
    rows = []
    seen = set()
    for group_name, df in specs:
        for _, row in df.iterrows():
            key = (str(row["event_uid"]), int(row["delay_ms"]))
            if key in seen:
                continue
            seen.add(key)
            item = row.to_dict()
            item["case_group"] = group_name
            rows.append(item)
    return pd.DataFrame(rows)


def plot_case_figure(
    row: pd.Series,
    y_true: np.ndarray,
    pred_v236: np.ndarray,
    pred_v238: np.ndarray,
    pred_v239: np.ndarray,
    hist_weight: np.ndarray,
    road_weight: np.ndarray,
    path: Path,
) -> None:
    """生成单个 case 的曲线 + attention 权重图。"""

    sample_index = int(row["sample_index"])
    delay_s = int(row["delay_ms"]) / 1000.0
    original_rel = delay_s + FUTURE_GRID
    valid = original_rel <= 2.0 + 1e-9

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 9.0), gridspec_kw={"height_ratios": [2.0, 1.0, 1.0]})
    ax = axes[0]
    ax.plot(original_rel[valid], y_true[sample_index, valid], label="true", color="#111111", linewidth=2.0)
    ax.plot(original_rel[valid], pred_v236[sample_index, valid], label="v236", color="#777777", linewidth=1.6)
    ax.plot(original_rel[valid], pred_v238[sample_index, valid], label="v238", color="#1f77b4", linewidth=1.6)
    ax.plot(original_rel[valid], pred_v239[sample_index, valid], label="v239 attention", color="#d62728", linewidth=1.8)
    ax.axvline(delay_s, color="#999999", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlim(0, 2.05)
    ax.set_xlabel("original anchor relative time (s)")
    ax.set_ylabel("steering delta")
    ax.set_title(
        f"{row['case_group']} | delay={int(row['delay_ms'])}ms | "
        f"tail_delta={float(row['delta_tail_v239_minus_v236']):+.3f}"
    )
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=4, fontsize=8)

    axes[1].bar(HISTORY_GRID, hist_weight, width=0.07, color="#9467bd")
    axes[1].set_xlim(float(HISTORY_GRID.min()) - 0.05, 0.05)
    axes[1].set_ylabel("hist attn")
    axes[1].set_xlabel("history time relative to observation (s)")
    axes[1].grid(alpha=0.20)

    axes[2].bar(FUTURE_GRID, road_weight, width=0.07, color="#2ca02c")
    axes[2].set_xlim(-0.05, 2.05)
    axes[2].set_ylabel("road attn")
    axes[2].set_xlabel("road preview time relative to observation (s)")
    axes[2].grid(alpha=0.20)

    fig.suptitle(str(row["event_uid"])[:120], fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_attention_casebook(inputs: Dict[str, object], case_tables: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame, List[Path]]:
    """为代表样本抽取 attention 权重并生成 casebook 图。"""

    data = inputs["data"]
    point_data = inputs["point_data"]
    arrays = inputs["arrays"]
    model = inputs["model"]
    y_true = inputs["y_true"]
    pred_v236 = inputs["pred_v236"]
    pred_v238 = inputs["pred_v238"]
    pred_v239 = inputs["pred_v239"]

    cases = choose_attention_cases(case_tables)
    focus_rows: List[Dict[str, object]] = []
    case_rows: List[Dict[str, object]] = []
    figure_paths: List[Path] = []
    for idx, row in cases.reset_index(drop=True).iterrows():
        sample_index = int(row["sample_index"])
        delay_s = int(row["delay_ms"]) / 1000.0
        original_rel = delay_s + FUTURE_GRID
        valid = original_rel <= 2.0 + 1e-9
        true_curve = y_true[sample_index]
        valid_indices = np.where(valid)[0]
        peak_local = int(np.argmax(np.abs(true_curve[valid])))
        time_index = int(valid_indices[peak_local])
        hist_weight, road_weight = attention_for_case(model, arrays, point_data, sample_index, time_index)

        hist_top = int(np.argmax(hist_weight))
        road_top = int(np.argmax(road_weight))
        case_id = f"{idx:02d}_{safe_name(row['case_group'])}_{safe_name(row['event_uid'], 45)}_{int(row['delay_ms'])}ms"
        fig_path = CASE_FIGURES / f"{case_id}.png"
        plot_case_figure(row, y_true, pred_v236, pred_v238, pred_v239, hist_weight, road_weight, fig_path)
        figure_paths.append(fig_path)

        focus_rows.append(
            {
                "case_id": case_id,
                "case_group": row["case_group"],
                "event_uid": row["event_uid"],
                "split": row["split"],
                "delay_ms": int(row["delay_ms"]),
                "sample_index": sample_index,
                "attention_future_time_index": time_index,
                "attention_original_rel_s": float(original_rel[time_index]),
                "hist_top_rel_s": float(HISTORY_GRID[hist_top]),
                "hist_top_weight": float(hist_weight[hist_top]),
                "hist_entropy_norm": entropy(hist_weight),
                "hist_mass_last_0p5s": float(hist_weight[HISTORY_GRID >= -0.5].sum()),
                "hist_mass_last_1p0s": float(hist_weight[HISTORY_GRID >= -1.0].sum()),
                "road_top_rel_s": float(FUTURE_GRID[road_top]),
                "road_top_weight": float(road_weight[road_top]),
                "road_entropy_norm": entropy(road_weight),
                "road_mass_0to0p8s": float(road_weight[FUTURE_GRID <= 0.8].sum()),
                "road_mass_0to1p2s": float(road_weight[FUTURE_GRID <= 1.2].sum()),
            }
        )
        item = {
            "case_id": case_id,
            "case_group": row["case_group"],
            "event_uid": row["event_uid"],
            "delay_ms": int(row["delay_ms"]),
            "split": row["split"],
            "v239_tail_rmse": float(row["v239_tail_rmse"]),
            "delta_tail_v239_minus_v236": float(row["delta_tail_v239_minus_v236"]),
            "delta_sample_v239_minus_v236": float(row["delta_sample_v239_minus_v236"]),
            "observe_later_like": bool(row["observe_later_like"]),
            "strong_steer": bool(row["strong_steer"]),
            "normal_predictable": bool(row["normal_predictable"]),
            "reverse_or_multi_correction": bool(row["reverse_or_multi_correction"]),
            "figure_path": str(fig_path.relative_to(OUT)).replace("\\", "/"),
            "human_review_decision": "",
            "human_note_cn": "",
        }
        case_rows.append(item)

    return pd.DataFrame(case_rows), pd.DataFrame(focus_rows), figure_paths


def build_locked_summary(per_sample: pd.DataFrame, subbucket: pd.DataFrame) -> pd.DataFrame:
    """生成 v240 锁定总体摘要。"""

    test = subbucket[subbucket["split"].eq("test")].copy()
    rows = []
    for bucket in ["all", "observe_later_like", "strong_steer", "normal_predictable"]:
        one = test[test["bucket"].eq(bucket)]
        rows.append(
            {
                "scope": bucket,
                "delay_count": int(one["delay_ms"].nunique()),
                "mean_delta_tail_v239_minus_v236": float(one["delta_tail_v239_minus_v236"].mean()),
                "max_delta_tail_v239_minus_v236": float(one["delta_tail_v239_minus_v236"].max()),
                "mean_delta_sample_v239_minus_v236": float(one["delta_sample_v239_minus_v236"].mean()),
                "all_delay_tail_noharm_pass": bool((one["delta_tail_v239_minus_v236"] <= 0.02).all()),
                "all_delay_sample_noharm_pass": bool((one["delta_sample_v239_minus_v236"] <= 0.02).all()),
            }
        )
    strong_reg = per_sample[
        per_sample["split"].eq("test")
        & per_sample["strong_steer"].astype(bool)
        & per_sample["delay_ms"].astype(int).isin([400, 1000])
        & (per_sample["delta_tail_v239_minus_v236"] > 0)
    ]
    rows.append(
        {
            "scope": "strong_400_1000_positive_regression_cases",
            "delay_count": 2,
            "mean_delta_tail_v239_minus_v236": float(strong_reg["delta_tail_v239_minus_v236"].mean()) if not strong_reg.empty else 0.0,
            "max_delta_tail_v239_minus_v236": float(strong_reg["delta_tail_v239_minus_v236"].max()) if not strong_reg.empty else 0.0,
            "mean_delta_sample_v239_minus_v236": float(strong_reg["delta_sample_v239_minus_v236"].mean()) if not strong_reg.empty else 0.0,
            "all_delay_tail_noharm_pass": False,
            "all_delay_sample_noharm_pass": False,
            "case_count": int(len(strong_reg)),
        }
    )
    return pd.DataFrame(rows)


def build_next_decision(summary: pd.DataFrame, casebook: pd.DataFrame) -> pd.DataFrame:
    """形成 v240 后的下一步决策。"""

    normal = summary[summary["scope"].eq("normal_predictable")].iloc[0]
    observe = summary[summary["scope"].eq("observe_later_like")].iloc[0]
    strong_reg_rows = summary[summary["scope"].eq("strong_400_1000_positive_regression_cases")]
    strong_case_count = int(strong_reg_rows.iloc[0].get("case_count", 0)) if not strong_reg_rows.empty else 0
    rows = [
        {
            "decision_item": "attention_candidate_survives_locked_audit",
            "decision": bool(normal["all_delay_tail_noharm_pass"] and observe["all_delay_tail_noharm_pass"]),
            "reason": "normal_predictable and observe_later_like pass locked test no-harm by delay; attention remains a valid next-stage candidate.",
        },
        {
            "decision_item": "formal_replacement_allowed",
            "decision": False,
            "reason": "v240 is locked audit/casebook only. Formal headline remains v225/v226 until robustness and manual case review are completed.",
        },
        {
            "decision_item": "strong_exception_requires_review",
            "decision": strong_case_count > 0,
            "reason": f"strong_steer 400/1000ms contains {strong_case_count} positive-regression test cases; inspect casebook before claiming strong bucket solved.",
        },
        {
            "decision_item": "recommended_next_task",
            "decision": "v241_attention_case_manual_review_and_robustness_ci",
            "reason": "Use v240 casebook for manual review, then run robustness/CI; do not expand architecture before resolving strong exceptions.",
        },
    ]
    return pd.DataFrame(rows)


def build_guardrail_json(split_check: pd.DataFrame) -> Dict[str, object]:
    """v240 审查边界。"""

    checks = {
        "stage": "v240_locked_attention_audit",
        "trained_new_model": False,
        "changed_model_config": False,
        "test_used_for_selection": False,
        "gate_router_selector_created": False,
        "response_type_hard_routing_created": False,
        "formal_headline_changed": False,
        "same_event_uid_cross_split_count": int(split_check["split_check_status"].eq("fail").sum()),
        "pass": False,
    }
    checks["pass"] = (
        not checks["trained_new_model"]
        and not checks["changed_model_config"]
        and not checks["test_used_for_selection"]
        and not checks["gate_router_selector_created"]
        and not checks["response_type_hard_routing_created"]
        and not checks["formal_headline_changed"]
        and checks["same_event_uid_cross_split_count"] == 0
    )
    return checks


def write_input_hashes() -> None:
    """记录关键输入哈希。"""

    paths = [V239_SCRIPT, V239_PRED, V239_MODEL, V239_SELECTION, V239_DECISION, V238.V236_ARRAYS, V238.V236_MANIFEST]
    rows = [{"path": str(path), "sha256": file_sha256(path), "bytes": int(path.stat().st_size)} for path in paths]
    write_csv(pd.DataFrame(rows), LOGS / "input_file_hashes.csv")


def file_inventory() -> Dict[str, object]:
    """输出目录文件清单。"""

    entries = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.suffix.lower() != ".zip":
            entries.append(
                {
                    "relative_path": str(path.relative_to(OUT)).replace("\\", "/"),
                    "bytes": int(path.stat().st_size),
                    "sha256": file_sha256(path),
                }
            )
    return {"output_dir": str(OUT), "file_count_excluding_zip": len(entries), "files": entries}


def zip_outputs() -> Path:
    """打包 v240 输出。"""

    zip_path = OUT / "v240_locked_attention_audit_pack.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(OUT.rglob("*")):
            if path.is_file() and path != zip_path:
                zf.write(path, path.relative_to(OUT))
    with zipfile.ZipFile(zip_path, "r") as zf:
        bad = zf.testzip()
    if bad is not None:
        raise RuntimeError(f"ZIP 校验失败：{bad}")
    return zip_path


def write_report(
    locked_summary: pd.DataFrame,
    subbucket: pd.DataFrame,
    casebook: pd.DataFrame,
    attention_focus: pd.DataFrame,
    next_decision: pd.DataFrame,
    guardrail: Dict[str, object],
    zip_path: Path,
) -> None:
    """写中文锁定审查报告。"""

    lines: List[str] = []
    lines.append("# v240 锁定 attention 候选审查报告")
    lines.append("")
    lines.append("## 本轮边界")
    lines.append("")
    lines.append("- 本轮不训练模型，不改 attention 配置，不用 test 选择模型。")
    lines.append("- 本轮只读取 v239 已锁定的 `v239_light_attention_h32` 预测和权重，做样本级审查。")
    lines.append("- 本轮不创建 gate/router/selector，不做响应类型硬路由，不改变 formal headline。")
    lines.append("")
    lines.append("## 锁定数字结论")
    lines.append("")
    for row in locked_summary.itertuples(index=False):
        lines.append(
            f"- `{row.scope}`：mean tail delta={float(row.mean_delta_tail_v239_minus_v236):+.6f}，"
            f"max tail delta={float(row.max_delta_tail_v239_minus_v236):+.6f}，"
            f"tail no-harm all-delay={bool(row.all_delay_tail_noharm_pass)}。"
        )
    lines.append("")
    lines.append("## 重点发现")
    lines.append("")
    obs = locked_summary[locked_summary["scope"].eq("observe_later_like")].iloc[0]
    normal = locked_summary[locked_summary["scope"].eq("normal_predictable")].iloc[0]
    strong_reg = locked_summary[locked_summary["scope"].eq("strong_400_1000_positive_regression_cases")].iloc[0]
    lines.append(
        f"- observe_later_like 锁定审查通过：平均 tail delta `{float(obs.mean_delta_tail_v239_minus_v236):+.6f}`，所有 delay tail no-harm 为 `{bool(obs.all_delay_tail_noharm_pass)}`。"
    )
    lines.append(
        f"- normal_predictable 锁定 no-harm 通过：平均 tail delta `{float(normal.mean_delta_tail_v239_minus_v236):+.6f}`，所有 delay tail no-harm 为 `{bool(normal.all_delay_tail_noharm_pass)}`。"
    )
    lines.append(
        f"- strong_steer 仍有例外：400/1000ms 正向退化样本数 `{int(strong_reg.get('case_count', 0))}`，需要人工看 casebook。"
    )
    if not attention_focus.empty:
        lines.append(
            f"- attention 代表样本平均历史最后 1 秒权重 `{float(attention_focus['hist_mass_last_1p0s'].mean()):.3f}`，"
            f"道路 0-1.2 秒权重 `{float(attention_focus['road_mass_0to1p2s'].mean()):.3f}`。"
        )
    lines.append("")
    lines.append("## 下一步决策")
    lines.append("")
    for row in next_decision.itertuples(index=False):
        lines.append(f"- `{row.decision_item}`: `{row.decision}`；{row.reason}")
    lines.append("")
    lines.append("## 代表图")
    lines.append("")
    lines.append(f"- attention casebook 图数：`{len(casebook)}`，目录：`figures/attention_casebook/`。")
    lines.append("- 每张图包含 true/v236/v238/v239 曲线、历史 attention 权重和道路预瞄 attention 权重。")
    lines.append("")
    lines.append("## Guardrail")
    lines.append("")
    for key, value in guardrail.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## 输出")
    lines.append("")
    lines.append("- `tables/v240_locked_overall_summary.csv`")
    lines.append("- `tables/v240_subbucket_noharm_audit.csv`")
    lines.append("- `tables/v240_per_sample_locked_metrics.csv`")
    lines.append("- `tables/v240_top_observe_later_improvements.csv`")
    lines.append("- `tables/v240_strong_400_1000_regressions.csv`")
    lines.append("- `tables/v240_attention_casebook_index.csv`")
    lines.append("- `tables/v240_attention_time_focus_summary.csv`")
    lines.append("- `tables/v240_next_decision.csv`")
    lines.append(f"- ZIP：`{zip_path.name}`")
    lines.append("")
    (REPORTS / "v240_locked_attention_audit_cn.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    np.random.seed(SEED)
    clean_out_dir()
    print("[v240] loading locked v239 inputs")
    inputs = load_locked_inputs()
    data = inputs["data"]

    print("[v240] computing per-sample locked metrics")
    per_sample = merge_per_sample_metrics(
        data=data,
        y_true=inputs["y_true"],
        pred_v236=inputs["pred_v236"],
        pred_v238=inputs["pred_v238"],
        pred_v239=inputs["pred_v239"],
    )
    subbucket = build_subbucket_summary(per_sample)
    case_tables = build_case_tables(per_sample)

    print("[v240] extracting attention weights and casebook figures")
    casebook, attention_focus, figure_paths = build_attention_casebook(inputs, case_tables)
    locked_summary = build_locked_summary(per_sample, subbucket)
    next_decision = build_next_decision(locked_summary, casebook)
    split_check = V238.split_integrity_check(data.manifest)
    guardrail = build_guardrail_json(split_check)
    if not bool(guardrail["pass"]):
        raise AssertionError("v240 guardrail 未通过：\n" + json.dumps(guardrail, ensure_ascii=False, indent=2))

    print("[v240] writing outputs")
    write_csv(locked_summary, TABLES / "v240_locked_overall_summary.csv")
    write_csv(subbucket, TABLES / "v240_subbucket_noharm_audit.csv")
    write_csv(per_sample, TABLES / "v240_per_sample_locked_metrics.csv")
    for name, table in case_tables.items():
        write_csv(table, TABLES / f"v240_{name}.csv")
    write_csv(casebook, TABLES / "v240_attention_casebook_index.csv")
    write_csv(attention_focus, TABLES / "v240_attention_time_focus_summary.csv")
    write_csv(next_decision, TABLES / "v240_next_decision.csv")
    write_csv(split_check, TABLES / "v240_split_integrity_check.csv")

    write_input_hashes()
    leakage = {
        "same_event_uid_cross_split_count": int(split_check["split_check_status"].eq("fail").sum()),
        "test_used_for_selection": False,
        "pass": int(split_check["split_check_status"].eq("fail").sum()) == 0,
    }
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    (LOGS / "leakage_check.json").write_text(json.dumps(leakage, ensure_ascii=False, indent=2), encoding="utf-8")
    run_manifest = {
        "stage": "v240_locked_attention_audit",
        "created_by": Path(__file__).name,
        "output_dir": str(OUT),
        "source_v239_dir": str(V239_DIR),
        "best_attention_model": inputs["best_name"],
        "n_rolling_samples": int(len(data.manifest)),
        "n_events": int(data.manifest["event_uid"].nunique()),
        "casebook_figures": [str(path.relative_to(OUT)).replace("\\", "/") for path in figure_paths],
    }
    (LOGS / "run_manifest.json").write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (LOGS / "file_inventory.json").write_text(json.dumps(file_inventory(), ensure_ascii=False, indent=2), encoding="utf-8")
    zip_path = zip_outputs()
    write_report(locked_summary, subbucket, casebook, attention_focus, next_decision, guardrail, zip_path)
    (LOGS / "file_inventory.json").write_text(json.dumps(file_inventory(), ensure_ascii=False, indent=2), encoding="utf-8")
    zip_path = zip_outputs()

    print("[v240] finished")
    print(f"output_dir={OUT}")
    print(f"report={REPORTS / 'v240_locked_attention_audit_cn.md'}")
    print(f"casebook_figures={len(figure_paths)}")
    print(f"zip={zip_path}")


if __name__ == "__main__":
    main()
