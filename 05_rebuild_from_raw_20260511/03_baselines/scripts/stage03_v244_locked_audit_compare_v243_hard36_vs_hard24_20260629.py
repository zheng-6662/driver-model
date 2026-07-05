#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v244：固定 v243 产物，对 hard36 与 hard24 做 locked audit 对比。

本脚本不训练模型、不调权重、不改 validation 选择规则。

为什么要做这一轮：
- v243 的 validation 规则选择了 `v243_metric_hard36_guard08`；
- 但 v243 的 test 稳定性补充显示 `v243_metric_hard24_guard04` 在 hard bucket 上更均衡；
- 因此本轮只读取 v243 已落盘的表和预测，做“候选比较 / 证据完整性 / 下一步决策”。

重要限制：
- v243 的 npz 只保存了 best guarded 模型，也就是 hard36 的完整曲线预测；
- hard24 只有 aggregate metrics，没有完整曲线预测、checkpoint 和逐样本 delta；
- 所以本轮能完整做 hard36 的逐样本风险审计，但 hard24 只能做 aggregate locked test 对比。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"F:\data_set_process\data_process")
REBUILD = ROOT / "05_rebuild_from_raw_20260511"
BASELINES = REBUILD / "03_baselines"

V243_DIR = BASELINES / "v243_v241_guarded_finetune_20260626"
V243_PRED = V243_DIR / "v243_v241_guarded_finetune_predictions.npz"
V243_SELECTION = V243_DIR / "tables" / "v243_model_selection_validation_guarded.csv"
V243_COMPARE = V243_DIR / "tables" / "v243_compare_vs_v236_v239_v241_original_remaining.csv"
V243_ROBUST = V243_DIR / "tables" / "v243_candidate_test_robustness_summary.csv"
V243_PER_SAMPLE = V243_DIR / "tables" / "v243_per_sample_delta_vs_v241.csv"
V243_GUARDRAIL = V243_DIR / "logs" / "guardrail_check.json"
V243_LEAKAGE = V243_DIR / "logs" / "leakage_check.json"
V243_REPORT = V243_DIR / "reports" / "v243_v241_guarded_finetune_cn.md"

OUT = BASELINES / "v244_locked_audit_compare_v243_hard36_vs_hard24_20260629"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"

V241_NAME = "v241_tcn_mha_h96"
HARD24 = "v243_metric_hard24_guard04"
HARD30 = "v243_metric_hard30_guard06_anchor04"
HARD36 = "v243_metric_hard36_guard08"
CANDIDATES = [HARD24, HARD30, HARD36]
COMPARE_FOCUS = [HARD24, HARD36]
CORE_BUCKETS = ["all", "normal_predictable", "observe_later_like", "strong_steer"]

mpl.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
mpl.rcParams["axes.unicode_minus"] = False


def ensure_dirs() -> None:
    """创建 v244 输出目录。"""

    for folder in (TABLES, FIGURES, REPORTS, LOGS):
        folder.mkdir(parents=True, exist_ok=True)


def clean_out_dir() -> None:
    """只清理 v244 自己的输出目录。"""

    if OUT.exists():
        shutil.rmtree(OUT)
    ensure_dirs()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """CSV 使用 utf-8-sig，方便 Windows Excel 打开中文说明。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def file_sha256(path: Path) -> str:
    """计算 SHA256，便于追溯输入产物。"""

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_inputs() -> None:
    """确认 v244 所需的 v243 产物存在。"""

    missing = [
        str(path)
        for path in [
            V243_PRED,
            V243_SELECTION,
            V243_COMPARE,
            V243_ROBUST,
            V243_PER_SAMPLE,
            V243_GUARDRAIL,
            V243_LEAKAGE,
            V243_REPORT,
        ]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("v244 缺少输入文件：\n" + "\n".join(missing))


def safe_float(value: object, default: float = math.nan) -> float:
    """安全转 float。"""

    try:
        out = float(value)
    except Exception:
        return default
    if not np.isfinite(out):
        return default
    return out


def load_prediction_availability() -> pd.DataFrame:
    """
    审查 v243 npz 里保存了哪些候选预测。

    这一步非常关键：hard24 没有完整预测就不能做逐样本 casebook。
    """

    with np.load(V243_PRED, allow_pickle=False) as pred:
        keys = list(pred.files)
        best_model = str(pred["best_guarded_model"][0])

    rows = []
    rows.append(
        {
            "artifact": "v243_prediction_npz",
            "path": str(V243_PRED),
            "available": True,
            "detail": "npz exists and was readable",
        }
    )
    for model in CANDIDATES:
        direct_key = f"pred_{model}_steering_delta"
        if model == best_model:
            saved = "pred_v243_best_guarded_steering_delta" in keys
            detail = "saved as best guarded prediction"
        else:
            saved = direct_key in keys
            detail = "not saved unless explicit per-candidate key exists"
        rows.append(
            {
                "artifact": f"full_curve_prediction__{model}",
                "path": str(V243_PRED),
                "available": bool(saved),
                "detail": detail,
            }
        )
    rows.append(
        {
            "artifact": f"per_sample_delta__{HARD36}",
            "path": str(V243_PER_SAMPLE),
            "available": True,
            "detail": "v243 per-sample table corresponds to best guarded model hard36",
        }
    )
    rows.append(
        {
            "artifact": f"per_sample_delta__{HARD24}",
            "path": "",
            "available": False,
            "detail": "hard24 full prediction/checkpoint was not saved in v243, so per-sample hard24 audit is unavailable without replaying training",
        }
    )
    rows.append(
        {
            "artifact": "aggregate_metrics_for_all_v243_candidates",
            "path": str(V243_COMPARE),
            "available": True,
            "detail": "hard24/hard30/hard36 aggregate test metrics are available",
        }
    )
    return pd.DataFrame(rows)


def build_validation_test_summary(selection: pd.DataFrame, robust: pd.DataFrame) -> pd.DataFrame:
    """把 validation 结论和 locked test 稳定性放到一张候选级对照表。"""

    rows: List[Dict[str, object]] = []
    sel = selection.set_index("model_name")
    for model in CANDIDATES:
        row: Dict[str, object] = {
            "model_name": model,
            "validation_rank": int(sel.loc[model, "validation_rank"]),
            "validation_selection_score": safe_float(sel.loc[model, "validation_selection_score"]),
            "accepted_as_next_candidate_by_validation": bool(sel.loc[model, "accepted_as_next_candidate"]),
            "best_epoch": int(sel.loc[model, "best_epoch"]),
            "normal_max_tail_delta_vs_v241_val": safe_float(sel.loc[model, "normal_max_tail_delta_vs_v241"]),
            "all_mean_tail_delta_vs_v241_val_0to800": safe_float(sel.loc[model, "all_mean_tail_delta_vs_v241_0to800"]),
            "observe_mean_tail_delta_vs_v241_val_0to800": safe_float(
                sel.loc[model, "observe_later_mean_tail_delta_vs_v241_0to800"]
            ),
            "strong_exception_mean_tail_delta_vs_v241_val_400_1000": safe_float(
                sel.loc[model, "strong_exception_mean_tail_delta_vs_v241_400_1000"]
            ),
            "val_all_tail_regression_rate_vs_v241": safe_float(sel.loc[model, "val_all_tail_regression_rate_vs_v241"]),
        }
        one = robust[robust["model_name"].eq(model)].set_index("bucket")
        for bucket in CORE_BUCKETS:
            row[f"test_{bucket}_mean_tail_delta_vs_v241"] = safe_float(
                one.loc[bucket, "mean_tail_delta_test_vs_v241"]
            )
            row[f"test_{bucket}_max_tail_delta_vs_v241"] = safe_float(
                one.loc[bucket, "max_tail_delta_test_vs_v241"]
            )
            row[f"test_{bucket}_worse_delay_count_vs_v241"] = int(
                one.loc[bucket, "n_delay_tail_worse_vs_v241"]
            )
        row["test_hard_bucket_mean_tail_delta_vs_v241"] = float(
            np.mean(
                [
                    row["test_observe_later_like_mean_tail_delta_vs_v241"],
                    row["test_strong_steer_mean_tail_delta_vs_v241"],
                ]
            )
        )
        row["test_hard_bucket_worse_delay_count_vs_v241"] = int(
            row["test_observe_later_like_worse_delay_count_vs_v241"]
            + row["test_strong_steer_worse_delay_count_vs_v241"]
        )
        row["test_all_and_hard_stability_score_lower_is_better"] = float(
            row["test_all_worse_delay_count_vs_v241"] * 0.50
            + row["test_hard_bucket_worse_delay_count_vs_v241"]
            + max(0.0, row["test_hard_bucket_mean_tail_delta_vs_v241"]) * 100.0
            + max(0.0, row["test_all_mean_tail_delta_vs_v241"]) * 100.0
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    out = out.sort_values(
        [
            "test_all_and_hard_stability_score_lower_is_better",
            "test_hard_bucket_worse_delay_count_vs_v241",
            "test_all_mean_tail_delta_vs_v241",
        ]
    ).reset_index(drop=True)
    out["locked_test_stability_rank"] = np.arange(1, len(out) + 1)
    return out


def build_per_delay_focus(compare: pd.DataFrame) -> pd.DataFrame:
    """生成 hard24 vs hard36 的 per-delay 对比表。"""

    rows: List[Dict[str, object]] = []
    for _, row in compare[compare["bucket"].isin(CORE_BUCKETS)].sort_values(["bucket", "delay_ms"]).iterrows():
        item: Dict[str, object] = {
            "bucket": row["bucket"],
            "delay_ms": int(row["delay_ms"]),
        }
        for model in COMPARE_FOCUS:
            item[f"tail_rmse__{model}"] = safe_float(row[f"steer_tail_rmse_mean__{model}"])
            item[f"sample_rmse__{model}"] = safe_float(row[f"steer_sample_rmse_mean__{model}"])
            item[f"tail_delta_vs_v241__{model}"] = safe_float(
                row[f"delta_steer_tail_rmse_mean__{model}_minus_{V241_NAME}"]
            )
            item[f"sample_delta_vs_v241__{model}"] = safe_float(
                row[f"delta_steer_sample_rmse_mean__{model}_minus_{V241_NAME}"]
            )
        item["tail_delta_hard36_minus_hard24"] = (
            item[f"tail_delta_vs_v241__{HARD36}"] - item[f"tail_delta_vs_v241__{HARD24}"]
        )
        item["sample_delta_hard36_minus_hard24"] = (
            item[f"sample_delta_vs_v241__{HARD36}"] - item[f"sample_delta_vs_v241__{HARD24}"]
        )
        item["preferred_by_tail_rmse"] = HARD36 if item[f"tail_rmse__{HARD36}"] < item[f"tail_rmse__{HARD24}"] else HARD24
        item["preferred_by_sample_rmse"] = (
            HARD36 if item[f"sample_rmse__{HARD36}"] < item[f"sample_rmse__{HARD24}"] else HARD24
        )
        rows.append(item)
    return pd.DataFrame(rows)


def build_bucket_decision_matrix(per_delay: pd.DataFrame) -> pd.DataFrame:
    """按 bucket 汇总 hard24/hard36 的 locked test 偏好。"""

    rows = []
    for bucket, one in per_delay.groupby("bucket"):
        hard24_mean = float(one[f"tail_delta_vs_v241__{HARD24}"].mean())
        hard36_mean = float(one[f"tail_delta_vs_v241__{HARD36}"].mean())
        hard24_worse = int((one[f"tail_delta_vs_v241__{HARD24}"] > 0.0).sum())
        hard36_worse = int((one[f"tail_delta_vs_v241__{HARD36}"] > 0.0).sum())
        preferred = HARD36 if hard36_mean < hard24_mean else HARD24
        rows.append(
            {
                "bucket": bucket,
                "hard24_mean_tail_delta_vs_v241": hard24_mean,
                "hard36_mean_tail_delta_vs_v241": hard36_mean,
                "hard36_minus_hard24_mean_tail_delta": hard36_mean - hard24_mean,
                "hard24_worse_delay_count": hard24_worse,
                "hard36_worse_delay_count": hard36_worse,
                "preferred_by_locked_test_tail_mean": preferred,
                "interpretation_cn": (
                    "hard36 更适合 normal，但不是 hard bucket 最稳"
                    if bucket == "normal_predictable" and preferred == HARD36
                    else "hard24 在该 bucket 更稳，尤其要关注 hard36 的迁移风险"
                    if preferred == HARD24
                    else "hard36 在该 bucket 平均更好"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("bucket").reset_index(drop=True)


def bucket_mask(df: pd.DataFrame, bucket: str) -> np.ndarray:
    """给 hard36 per-sample 表生成 bucket mask。"""

    if bucket == "all":
        return np.ones(len(df), dtype=bool)
    if bucket == "observe_later_like":
        return df["observe_later_like"].astype(bool).to_numpy()
    if bucket == "normal_predictable":
        return (~df["observe_later_like"].astype(bool) & ~df["strong_steer"].astype(bool)).to_numpy()
    if bucket == "strong_steer":
        return df["strong_steer"].astype(bool).to_numpy()
    if bucket == "zero_cross_or_reverse_or_multi":
        return (
            df["zero_cross"].astype(bool) | df["reverse"].astype(bool) | df["multi_correction"].astype(bool)
        ).to_numpy()
    raise ValueError(f"unknown bucket: {bucket}")


def build_hard36_per_sample_risk(per_sample: pd.DataFrame) -> pd.DataFrame:
    """汇总 hard36 的逐样本回退风险。hard24 没有逐样本表，不能做同级分析。"""

    rows = []
    for split in ["val", "test"]:
        part = per_sample[per_sample["split"].eq(split)].copy()
        for bucket in ["all", "normal_predictable", "observe_later_like", "strong_steer", "zero_cross_or_reverse_or_multi"]:
            one = part.loc[bucket_mask(part, bucket)].copy()
            if one.empty:
                continue
            delta = one["delta_tail_v243_minus_v241"].astype(float)
            rows.append(
                {
                    "model_name": HARD36,
                    "split": split,
                    "bucket": bucket,
                    "n": int(len(one)),
                    "tail_regression_count_vs_v241": int((delta > 0.0).sum()),
                    "tail_regression_rate_vs_v241": float((delta > 0.0).mean()),
                    "mean_tail_delta_vs_v241": float(delta.mean()),
                    "p90_tail_delta_vs_v241": float(delta.quantile(0.90)),
                    "max_tail_delta_vs_v241": float(delta.max()),
                }
            )
    return pd.DataFrame(rows)


def build_hard36_worst_regressions(per_sample: pd.DataFrame, n: int = 80) -> pd.DataFrame:
    """输出 hard36 在 test 上相对 v241 最差的逐样本回退。"""

    keep_cols = [
        "event_uid",
        "sample_id",
        "split",
        "delay_ms",
        "observe_later_like",
        "strong_steer",
        "reverse",
        "zero_cross",
        "multi_correction",
        "extreme_peak",
        "tail_rmse_v241",
        "tail_rmse_v243",
        "delta_tail_v243_minus_v241",
        "sample_rmse_v241",
        "sample_rmse_v243",
        "delta_sample_v243_minus_v241",
        "peak_ratio_v241",
        "peak_ratio_v243",
    ]
    return (
        per_sample[per_sample["split"].eq("test")]
        .sort_values("delta_tail_v243_minus_v241", ascending=False)
        .head(n)[keep_cols]
        .reset_index(drop=True)
    )


def build_missing_granular_table(availability: pd.DataFrame) -> pd.DataFrame:
    """把 hard24 granular 缺口写成机器可读表。"""

    return pd.DataFrame(
        [
            {
                "item": "hard24_full_curve_prediction",
                "available": bool(
                    availability[
                        availability["artifact"].eq(f"full_curve_prediction__{HARD24}")
                    ]["available"].iloc[0]
                ),
                "impact_cn": "无法计算 hard24 逐样本 tail delta、worst regression casebook、同一事件曲线图。",
                "safe_next_step_cn": "如需完整 hard24 audit，应重放 v243 训练并保存 all-candidate predictions/checkpoints；不得改权重或基于 test 调参。",
            },
            {
                "item": "hard24_checkpoint",
                "available": False,
                "impact_cn": "无法从 checkpoint 直接重建 hard24 预测。",
                "safe_next_step_cn": "修改 v243 脚本保存每个候选 checkpoint，或仅做 aggregate-level audit。",
            },
            {
                "item": "hard24_per_sample_delta",
                "available": False,
                "impact_cn": "无法和 hard36 做同级别逐样本风险对照。",
                "safe_next_step_cn": "本轮 report 明确限制，不把 hard24 直接 claim 成 formal replacement。",
            },
        ]
    )


def build_next_decision(
    validation_test: pd.DataFrame,
    bucket_matrix: pd.DataFrame,
    missing_granular: pd.DataFrame,
) -> pd.DataFrame:
    """生成 v244 的决策表。"""

    hard36 = validation_test[validation_test["model_name"].eq(HARD36)].iloc[0]
    hard24 = validation_test[validation_test["model_name"].eq(HARD24)].iloc[0]
    hard24_granular_ok = bool(missing_granular["available"].astype(bool).all())
    hard36_hard_worse = int(hard36["test_hard_bucket_worse_delay_count_vs_v241"])
    hard24_hard_worse = int(hard24["test_hard_bucket_worse_delay_count_vs_v241"])
    return pd.DataFrame(
        [
            {
                "decision_item": "validation_selected_candidate",
                "decision": HARD36,
                "reason_cn": "hard36 是 v243 validation 规则下排名第一的 accepted candidate。",
            },
            {
                "decision_item": "locked_test_more_stable_candidate",
                "decision": HARD24,
                "reason_cn": f"hard24 在 observe/strong hard bucket 的变差 delay 数为 {hard24_hard_worse}/12，低于 hard36 的 {hard36_hard_worse}/12。",
            },
            {
                "decision_item": "promote_hard36_as_formal_replacement_now",
                "decision": False,
                "reason_cn": "hard36 虽通过 validation，但 locked test 上 observe_later_like 6/6 个 delay tail 变差、strong_steer 5/6 个 delay tail 变差。",
            },
            {
                "decision_item": "promote_hard24_as_formal_replacement_now",
                "decision": False,
                "reason_cn": "hard24 aggregate test 更稳，但 hard24 没有保存完整预测/checkpoint/逐样本表，且不能用 test 反向改 validation 选择。",
            },
            {
                "decision_item": "keep_v241_as_default_until_granular_audit",
                "decision": True,
                "reason_cn": "v243 的 hard24/hard36 结论还存在 validation-vs-test 选择冲突；正式替代前必须补齐 hard24 granular audit。",
            },
            {
                "decision_item": "recommended_next_step",
                "decision": "replay_v243_save_all_candidates_then_locked_audit_or_keep_aggregate_v244_as_limit",
                "reason_cn": "若要继续推进 v243，应只重放保存 hard24/hard36 全候选预测和 checkpoint，不改超参，不用 test 调参。",
            },
            {
                "decision_item": "hard24_granular_artifact_complete",
                "decision": hard24_granular_ok,
                "reason_cn": "当前 hard24 缺少完整曲线预测、checkpoint 和逐样本 delta。",
            },
        ]
    )


def plot_test_mean_tail_delta(validation_test: pd.DataFrame) -> Path:
    """画各候选在 test bucket 上相对 v241 的 mean tail delta。"""

    rows = []
    for _, row in validation_test.iterrows():
        for bucket in CORE_BUCKETS:
            rows.append(
                {
                    "model_name": row["model_name"],
                    "bucket": bucket,
                    "mean_tail_delta": row[f"test_{bucket}_mean_tail_delta_vs_v241"],
                }
            )
    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(11.0, 5.4))
    x = np.arange(len(CORE_BUCKETS))
    width = 0.25
    colors = {HARD24: "#1f77b4", HARD30: "#9467bd", HARD36: "#d62728"}
    for idx, model in enumerate(CANDIDATES):
        vals = [
            float(plot_df[plot_df["model_name"].eq(model) & plot_df["bucket"].eq(bucket)]["mean_tail_delta"].iloc[0])
            for bucket in CORE_BUCKETS
        ]
        ax.bar(x + (idx - 1) * width, vals, width=width, label=model.replace("v243_metric_", ""), color=colors[model])
    ax.axhline(0.0, color="#333333", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(CORE_BUCKETS, rotation=15, ha="right")
    ax.set_ylabel("Test mean tail delta vs v241")
    ax.set_title("v244 locked audit: candidate test mean tail delta")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = FIGURES / "v244_candidate_test_mean_tail_delta.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_per_delay_hard24_vs_hard36(per_delay: pd.DataFrame) -> Path:
    """画 hard24 / hard36 各 bucket per-delay tail delta。"""

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), sharex=True)
    axes = axes.ravel()
    for ax, bucket in zip(axes, CORE_BUCKETS):
        one = per_delay[per_delay["bucket"].eq(bucket)].sort_values("delay_ms")
        ax.plot(
            one["delay_ms"],
            one[f"tail_delta_vs_v241__{HARD24}"],
            marker="o",
            label="hard24",
            color="#1f77b4",
        )
        ax.plot(
            one["delay_ms"],
            one[f"tail_delta_vs_v241__{HARD36}"],
            marker="o",
            label="hard36",
            color="#d62728",
        )
        ax.axhline(0.0, color="#333333", linewidth=0.8)
        ax.set_title(bucket)
        ax.set_xlabel("delay ms")
        ax.set_ylabel("tail delta vs v241")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = FIGURES / "v244_per_delay_tail_delta_hard24_vs_hard36.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_validation_vs_test_tradeoff(validation_test: pd.DataFrame) -> Path:
    """画 validation score 与 locked test hard bucket 退化之间的 tradeoff。"""

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    colors = {HARD24: "#1f77b4", HARD30: "#9467bd", HARD36: "#d62728"}
    for _, row in validation_test.iterrows():
        model = str(row["model_name"])
        ax.scatter(
            row["validation_selection_score"],
            row["test_hard_bucket_mean_tail_delta_vs_v241"],
            s=90,
            color=colors.get(model, "#777777"),
            label=model.replace("v243_metric_", ""),
        )
        ax.annotate(
            model.replace("v243_metric_", ""),
            (row["validation_selection_score"], row["test_hard_bucket_mean_tail_delta_vs_v241"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Validation selection score lower is better")
    ax.set_ylabel("Test hard-bucket mean tail delta vs v241")
    ax.set_title("v244 validation vs locked-test tradeoff")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = FIGURES / "v244_validation_vs_test_tradeoff.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_input_hashes() -> None:
    """记录输入文件哈希。"""

    rows = []
    for path in [
        V243_PRED,
        V243_SELECTION,
        V243_COMPARE,
        V243_ROBUST,
        V243_PER_SAMPLE,
        V243_GUARDRAIL,
        V243_LEAKAGE,
        V243_REPORT,
    ]:
        rows.append({"path": str(path), "bytes": int(path.stat().st_size), "sha256": file_sha256(path)})
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
    """打包 v244 输出并做 ZIP 完整性校验。"""

    zip_path = OUT / "v244_locked_audit_compare_v243_hard36_vs_hard24_pack.zip"
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


def build_guardrail_json(availability: pd.DataFrame, leakage: Dict[str, object]) -> Dict[str, object]:
    """记录 v244 的方法边界。"""

    checks = {
        "stage": "v244_locked_audit_compare_v243_hard36_vs_hard24",
        "source_stage": "v243_v241_guarded_finetune",
        "new_model_trained": False,
        "hyperparameter_changed": False,
        "test_used_for_retuning": False,
        "test_used_for_locked_audit_reporting": True,
        "gate_router_selector_created": False,
        "sample_deleted": False,
        "formal_headline_changed": False,
        "hard24_granular_prediction_available": bool(
            availability[availability["artifact"].eq(f"full_curve_prediction__{HARD24}")]["available"].iloc[0]
        ),
        "source_v243_leakage_pass": bool(leakage.get("pass", False)),
        "same_event_uid_cross_split_count": int(leakage.get("same_event_uid_cross_split_count", -1)),
    }
    checks["pass"] = (
        not checks["new_model_trained"]
        and not checks["hyperparameter_changed"]
        and not checks["test_used_for_retuning"]
        and not checks["gate_router_selector_created"]
        and not checks["sample_deleted"]
        and not checks["formal_headline_changed"]
        and checks["source_v243_leakage_pass"]
        and checks["same_event_uid_cross_split_count"] == 0
    )
    return checks


def write_report(
    validation_test: pd.DataFrame,
    per_delay: pd.DataFrame,
    bucket_matrix: pd.DataFrame,
    hard36_risk: pd.DataFrame,
    missing_granular: pd.DataFrame,
    next_decision: pd.DataFrame,
    guardrail: Dict[str, object],
    zip_path: Path,
) -> None:
    """写中文审计报告。"""

    hard36 = validation_test[validation_test["model_name"].eq(HARD36)].iloc[0]
    hard24 = validation_test[validation_test["model_name"].eq(HARD24)].iloc[0]
    hard30 = validation_test[validation_test["model_name"].eq(HARD30)].iloc[0]
    lines: List[str] = []
    lines.append("# v244 locked audit：v243 hard36 vs hard24 对比报告")
    lines.append("")
    lines.append("## 本轮做了什么")
    lines.append("")
    lines.append("- 只读取 v243 已落盘产物，不训练模型，不调权重，不改 validation 规则。")
    lines.append("- 对比 validation-selected `v243_metric_hard36_guard08` 和 conservative/test-robust `v243_metric_hard24_guard04`。")
    lines.append("- 同时保留 `v243_metric_hard30_guard06_anchor04` 作为参考，因为它在 all/normal 上也很强。")
    lines.append("- 本轮是 locked audit/reporting，不是新模型实验。")
    lines.append("")
    lines.append("## 关键限制")
    lines.append("")
    lines.append("- v243 的 npz 只保存了 best guarded 预测，也就是 hard36。")
    lines.append("- hard24 没有完整曲线预测、checkpoint 和逐样本 delta，因此 hard24 只能做 aggregate 对比，不能做同级别 per-sample casebook。")
    lines.append("- 这个限制不影响 aggregate test 结论，但会阻止把 hard24 直接升级为 formal replacement。")
    lines.append("")
    lines.append("## 候选级结论")
    lines.append("")
    lines.append(
        f"- validation-selected：`{HARD36}`，validation score={float(hard36.validation_selection_score):.6f}，"
        f"best_epoch={int(hard36.best_epoch)}。"
    )
    lines.append(
        f"- hard36 test：all={float(hard36.test_all_mean_tail_delta_vs_v241):+.6f}，"
        f"normal={float(hard36.test_normal_predictable_mean_tail_delta_vs_v241):+.6f}，"
        f"observe={float(hard36.test_observe_later_like_mean_tail_delta_vs_v241):+.6f}，"
        f"strong={float(hard36.test_strong_steer_mean_tail_delta_vs_v241):+.6f}。"
    )
    lines.append(
        f"- hard24 test：all={float(hard24.test_all_mean_tail_delta_vs_v241):+.6f}，"
        f"normal={float(hard24.test_normal_predictable_mean_tail_delta_vs_v241):+.6f}，"
        f"observe={float(hard24.test_observe_later_like_mean_tail_delta_vs_v241):+.6f}，"
        f"strong={float(hard24.test_strong_steer_mean_tail_delta_vs_v241):+.6f}。"
    )
    lines.append(
        f"- hard30 test 参考：all={float(hard30.test_all_mean_tail_delta_vs_v241):+.6f}，"
        f"normal={float(hard30.test_normal_predictable_mean_tail_delta_vs_v241):+.6f}，"
        f"observe={float(hard30.test_observe_later_like_mean_tail_delta_vs_v241):+.6f}，"
        f"strong={float(hard30.test_strong_steer_mean_tail_delta_vs_v241):+.6f}。"
    )
    lines.append("")
    lines.append("## Bucket 判断")
    lines.append("")
    for _, row in bucket_matrix.iterrows():
        lines.append(
            f"- {row.bucket}: hard24 mean tail delta={float(row.hard24_mean_tail_delta_vs_v241):+.6f}，"
            f"hard36 mean tail delta={float(row.hard36_mean_tail_delta_vs_v241):+.6f}，"
            f"preferred={row.preferred_by_locked_test_tail_mean}；{row.interpretation_cn}。"
        )
    lines.append("")
    lines.append("## hard36 逐样本风险")
    lines.append("")
    test_risk = hard36_risk[hard36_risk["split"].eq("test")].copy()
    for _, row in test_risk.iterrows():
        lines.append(
            f"- {row.bucket}: n={int(row.n)}，tail regression rate={float(row.tail_regression_rate_vs_v241):.3f}，"
            f"mean delta={float(row.mean_tail_delta_vs_v241):+.6f}，"
            f"max delta={float(row.max_tail_delta_vs_v241):+.6f}。"
        )
    lines.append("")
    lines.append("## 决策")
    lines.append("")
    for _, row in next_decision.iterrows():
        lines.append(f"- `{row.decision_item}`: `{row.decision}`。{row.reason_cn}")
    lines.append("")
    lines.append("## Guardrail")
    lines.append("")
    for key, value in guardrail.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## 主要产物")
    lines.append("")
    lines.append("- `tables/v244_validation_vs_test_candidate_compare.csv`")
    lines.append("- `tables/v244_per_delay_hard24_hard36_compare.csv`")
    lines.append("- `tables/v244_bucket_decision_matrix.csv`")
    lines.append("- `tables/v244_hard36_per_sample_risk_summary.csv`")
    lines.append("- `tables/v244_hard36_worst_regressions_vs_v241.csv`")
    lines.append("- `tables/v244_missing_hard24_granular_audit.csv`")
    lines.append("- `tables/v244_next_decision.csv`")
    lines.append(f"- ZIP：`{zip_path.name}`")
    lines.append("")
    (REPORTS / "v244_locked_audit_compare_v243_hard36_vs_hard24_cn.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    clean_out_dir()
    require_inputs()
    print("[v244] loading v243 artifacts")

    selection = pd.read_csv(V243_SELECTION, encoding="utf-8-sig")
    compare = pd.read_csv(V243_COMPARE, encoding="utf-8-sig")
    robust = pd.read_csv(V243_ROBUST, encoding="utf-8-sig")
    per_sample = pd.read_csv(V243_PER_SAMPLE, encoding="utf-8-sig")
    leakage = json.loads(V243_LEAKAGE.read_text(encoding="utf-8"))
    source_guardrail = json.loads(V243_GUARDRAIL.read_text(encoding="utf-8"))

    availability = load_prediction_availability()
    validation_test = build_validation_test_summary(selection, robust)
    per_delay = build_per_delay_focus(compare)
    bucket_matrix = build_bucket_decision_matrix(per_delay)
    hard36_risk = build_hard36_per_sample_risk(per_sample)
    hard36_worst = build_hard36_worst_regressions(per_sample, n=100)
    missing_granular = build_missing_granular_table(availability)
    next_decision = build_next_decision(validation_test, bucket_matrix, missing_granular)
    guardrail = build_guardrail_json(availability, leakage)
    if not bool(guardrail["pass"]):
        raise AssertionError("v244 guardrail 未通过：\n" + json.dumps(guardrail, ensure_ascii=False, indent=2))

    print("[v244] writing tables and figures")
    write_csv(availability, TABLES / "v244_prediction_availability_audit.csv")
    write_csv(validation_test, TABLES / "v244_validation_vs_test_candidate_compare.csv")
    write_csv(per_delay, TABLES / "v244_per_delay_hard24_hard36_compare.csv")
    write_csv(bucket_matrix, TABLES / "v244_bucket_decision_matrix.csv")
    write_csv(hard36_risk, TABLES / "v244_hard36_per_sample_risk_summary.csv")
    write_csv(hard36_worst, TABLES / "v244_hard36_worst_regressions_vs_v241.csv")
    write_csv(missing_granular, TABLES / "v244_missing_hard24_granular_audit.csv")
    write_csv(next_decision, TABLES / "v244_next_decision.csv")

    figures = [
        plot_test_mean_tail_delta(validation_test),
        plot_per_delay_hard24_vs_hard36(per_delay),
        plot_validation_vs_test_tradeoff(validation_test),
    ]

    write_input_hashes()
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    run_manifest = {
        "stage": "v244_locked_audit_compare_v243_hard36_vs_hard24",
        "created_by": Path(__file__).name,
        "output_dir": str(OUT),
        "source_v243_dir": str(V243_DIR),
        "new_model_trained": False,
        "candidate_focus": [HARD24, HARD36],
        "reference_model": V241_NAME,
        "source_v243_guardrail_pass": bool(source_guardrail.get("pass", False)),
        "figures": [str(path.relative_to(OUT)).replace("\\", "/") for path in figures],
    }
    (LOGS / "run_manifest.json").write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (LOGS / "file_inventory.json").write_text(json.dumps(file_inventory(), ensure_ascii=False, indent=2), encoding="utf-8")
    zip_path = zip_outputs()
    write_report(
        validation_test=validation_test,
        per_delay=per_delay,
        bucket_matrix=bucket_matrix,
        hard36_risk=hard36_risk,
        missing_granular=missing_granular,
        next_decision=next_decision,
        guardrail=guardrail,
        zip_path=zip_path,
    )
    (LOGS / "file_inventory.json").write_text(json.dumps(file_inventory(), ensure_ascii=False, indent=2), encoding="utf-8")
    zip_path = zip_outputs()

    print("[v244] finished")
    print(f"output_dir={OUT}")
    print(f"report={REPORTS / 'v244_locked_audit_compare_v243_hard36_vs_hard24_cn.md'}")
    print(f"zip={zip_path}")


if __name__ == "__main__":
    main()
