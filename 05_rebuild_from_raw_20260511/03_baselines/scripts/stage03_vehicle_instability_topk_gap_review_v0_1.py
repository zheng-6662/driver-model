# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd


ROOT = Path(r"F:/data_set_process/data_process/05_rebuild_from_raw_20260511")
TOPK_DIR = ROOT / "03_baselines" / "stage03_vehicle_instability_topk_vehicle_transformer_v0_1"
TOPK_TABLE_DIR = TOPK_DIR / "tables"
OUT_ROOT = ROOT / "03_baselines" / "stage03_vehicle_instability_topk_gap_review_v0_1"
TABLE_DIR = OUT_ROOT / "tables"
FIG_DIR = OUT_ROOT / "figures"
LOG_DIR = OUT_ROOT / "logs"
REPORT_ROOT = ROOT / "09_reports"

MANIFEST_PATH = ROOT / "02_samples" / "vehicle_instability_response_task_decision_v0_1" / "tables" / "sample_response_task_manifest.csv"
RESPONSE_LABEL_PATH = ROOT / "03_baselines" / "stage03_vehicle_instability_response_decomposition_labels_v0_1" / "tables" / "response_decomposition_sample_labels.csv"

TRACK_ID = "B_response3s_strict_core"
RBF_MODEL = "rbf_kernel_ridge_context_no_subject"
TOP1_MODEL = "topk_vehicle_transformer_top1_no_subject"
BESTK_MODEL = "topk_vehicle_transformer_best_of_3_oracle"
OUTPUT_VERSION = "stage03_vehicle_instability_topk_gap_review_v0_1"


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIG_DIR, LOG_DIR, REPORT_ROOT]:
        path.mkdir(parents=True, exist_ok=True)


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    diag = pd.read_csv(TOPK_TABLE_DIR / "topk_vehicle_transformer_branch_diagnostics.csv")
    per_sample = pd.read_csv(TOPK_TABLE_DIR / "topk_vehicle_transformer_per_sample_metrics.csv")
    manifest = pd.read_csv(MANIFEST_PATH)
    labels = pd.read_csv(RESPONSE_LABEL_PATH)
    labels = labels[labels["track_id"].astype(str) == TRACK_ID].copy()
    return diag, per_sample, manifest, labels


def prefixed_metrics(per_sample: pd.DataFrame, model_name: str, prefix: str) -> pd.DataFrame:
    keep = [
        "sample_id",
        "sample_rmse",
        "wrong_side",
        "large_response_recalled",
        "severe_amp_under",
        "peak_amp_abs_error",
        "peak_time_abs_error_s",
        "onset_delay_abs_error_s",
        "tail_abs_error",
        "tail_drift_risk",
        "zero_crossing_mismatch",
        "reversal_count_exact",
        "pred_multi_segment",
        "pred_peak_abs",
        "gt_peak_abs",
        "is_large_response",
        "is_difficult_peak_top20",
    ]
    out = per_sample[per_sample["model_name"].astype(str) == model_name][keep].copy()
    out = out.rename(columns={c: f"{prefix}_{c}" for c in keep if c != "sample_id"})
    return out


def build_detail() -> tuple[pd.DataFrame, dict[str, float]]:
    diag, per_sample, manifest, labels = read_inputs()
    detail = diag.copy()
    detail["top1_bestk_gap"] = detail["rmse_top1_branch"] - detail["rmse_bestk"]
    for model, prefix in [(RBF_MODEL, "rbf"), (TOP1_MODEL, "top1"), (BESTK_MODEL, "bestk")]:
        detail = detail.merge(prefixed_metrics(per_sample, model, prefix), on="sample_id", how="left", validate="one_to_one")
    manifest_cols = [
        "sample_id",
        "event_level",
        "road_type_anchor",
        "old_v400_phase_mode",
        "road_design_module_name",
        "road_design_instance_name",
        "road_design_risk_class",
        "road_design_mapping_reliability",
        "anchor_time_rel_s",
        "curvature_anchor",
        "median_speed_kmh_window",
    ]
    detail = detail.merge(manifest[[c for c in manifest_cols if c in manifest.columns]].drop_duplicates("sample_id"), on="sample_id", how="left")
    label_cols = [
        "sample_id",
        "computed_morphology",
        "amplitude_bucket",
        "response_family_target",
        "peak_time_bucket",
        "onset_bucket",
        "tail_state",
        "peak_abs",
        "peak_signed",
        "reversal_count",
    ]
    detail = detail.merge(labels[[c for c in label_cols if c in labels.columns]].drop_duplicates("sample_id"), on="sample_id", how="left")
    detail["top1_minus_rbf_rmse"] = detail["top1_sample_rmse"] - detail["rbf_sample_rmse"]
    detail["bestk_minus_rbf_rmse"] = detail["bestk_sample_rmse"] - detail["rbf_sample_rmse"]
    detail["bestk_gain_over_top1"] = detail["top1_sample_rmse"] - detail["bestk_sample_rmse"]
    detail["bestk_gain_over_rbf"] = detail["rbf_sample_rmse"] - detail["bestk_sample_rmse"]
    train = detail[detail["split"] == "train"].copy()
    thresholds = {
        "gap_train_p75": float(train["top1_bestk_gap"].quantile(0.75)),
        "gap_train_p90": float(train["top1_bestk_gap"].quantile(0.90)),
        "branch_spread_train_p75": float(train["branch_spread_mean"].quantile(0.75)),
        "top1_prob_train_p25": float(train["top1_prob"].quantile(0.25)),
        "prob_margin_train_p25": float(train["prob_margin"].quantile(0.25)),
    }
    for col in ["top1_prob", "prob_margin", "branch_spread_mean", "branch_spread_peak"]:
        mu = float(train[col].mean())
        sigma = float(train[col].std())
        if not np.isfinite(sigma) or sigma < 1e-9:
            sigma = 1.0
        detail[f"z_{col}"] = (pd.to_numeric(detail[col], errors="coerce") - mu) / sigma
    detail["simple_reliability_risk_score"] = (
        -detail["z_top1_prob"] - detail["z_prob_margin"] + detail["z_branch_spread_mean"] + detail["z_branch_spread_peak"]
    )
    risk_thr = float(detail.loc[detail["split"] == "train", "simple_reliability_risk_score"].quantile(0.75))
    thresholds["risk_score_train_p75"] = risk_thr
    detail["high_gap_train_p75"] = (detail["top1_bestk_gap"] >= thresholds["gap_train_p75"]).astype(int)
    detail["high_risk_train_p75"] = (detail["simple_reliability_risk_score"] >= risk_thr).astype(int)
    detail["low_confidence_train_rule"] = (
        (detail["top1_prob"] <= thresholds["top1_prob_train_p25"])
        | (detail["prob_margin"] <= thresholds["prob_margin_train_p25"])
        | (detail["branch_spread_mean"] >= thresholds["branch_spread_train_p75"])
    ).astype(int)
    return detail, thresholds


def summarize(detail: pd.DataFrame, thresholds: dict[str, float]) -> dict[str, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for split, grp in detail.groupby("split"):
        rows.append(
            {
                "split": split,
                "n_samples": int(len(grp)),
                "top1_matches_best_rate": float(grp["top1_matches_best"].mean()),
                "mean_top1_bestk_gap": float(grp["top1_bestk_gap"].mean()),
                "median_top1_bestk_gap": float(grp["top1_bestk_gap"].median()),
                "p75_top1_bestk_gap": float(grp["top1_bestk_gap"].quantile(0.75)),
                "mean_top1_minus_rbf_rmse": float(grp["top1_minus_rbf_rmse"].mean()),
                "mean_bestk_gain_over_rbf": float(grp["bestk_gain_over_rbf"].mean()),
                "high_gap_train_p75_rate": float(grp["high_gap_train_p75"].mean()),
                "high_risk_train_p75_rate": float(grp["high_risk_train_p75"].mean()),
                "high_risk_captures_high_gap_rate": float(
                    grp.loc[grp["high_gap_train_p75"] == 1, "high_risk_train_p75"].mean()
                ) if (grp["high_gap_train_p75"] == 1).any() else float("nan"),
                "low_conf_rule_captures_high_gap_rate": float(
                    grp.loc[grp["high_gap_train_p75"] == 1, "low_confidence_train_rule"].mean()
                ) if (grp["high_gap_train_p75"] == 1).any() else float("nan"),
            }
        )
    overall = pd.DataFrame(rows)
    thresh_df = pd.DataFrame([thresholds])

    corr_rows: list[dict[str, Any]] = []
    corr_cols = ["top1_prob", "prob_margin", "branch_spread_mean", "branch_spread_peak", "simple_reliability_risk_score"]
    targets = ["top1_bestk_gap", "top1_minus_rbf_rmse", "top1_matches_best"]
    for split, grp in detail.groupby("split"):
        for col in corr_cols:
            for target in targets:
                corr_rows.append(
                    {
                        "split": split,
                        "feature": col,
                        "target": target,
                        "pearson_corr": float(pd.to_numeric(grp[col], errors="coerce").corr(pd.to_numeric(grp[target], errors="coerce"), method="pearson")),
                        "spearman_corr": float(pd.to_numeric(grp[col], errors="coerce").corr(pd.to_numeric(grp[target], errors="coerce"), method="spearman")),
                    }
                )
    corr = pd.DataFrame(corr_rows)

    bucket_rows: list[pd.DataFrame] = []
    for split, grp in detail.groupby("split"):
        for field in ["top1_matches_best", "high_gap_train_p75", "high_risk_train_p75", "low_confidence_train_rule"]:
            part = (
                grp.groupby(field)
                .agg(
                    n_samples=("sample_id", "count"),
                    mean_gap=("top1_bestk_gap", "mean"),
                    mean_top1_rmse=("top1_sample_rmse", "mean"),
                    mean_bestk_rmse=("bestk_sample_rmse", "mean"),
                    mean_rbf_rmse=("rbf_sample_rmse", "mean"),
                    wrong_side_top1_rate=("top1_wrong_side", "mean"),
                    severe_under_top1_rate=("top1_severe_amp_under", "mean"),
                )
                .reset_index()
            )
            part["split"] = split
            part["bucket_field"] = field
            part = part.rename(columns={field: "bucket_value"})
            bucket_rows.append(part)
    bucket_summary = pd.concat(bucket_rows, ignore_index=True) if bucket_rows else pd.DataFrame()

    def group_summary(field: str) -> pd.DataFrame:
        if field not in detail.columns:
            return pd.DataFrame()
        grp = detail[detail["split"] == "test"].groupby(field, dropna=False)
        return (
            grp.agg(
                n_samples=("sample_id", "count"),
                top1_matches_best_rate=("top1_matches_best", "mean"),
                mean_gap=("top1_bestk_gap", "mean"),
                mean_top1_rmse=("top1_sample_rmse", "mean"),
                mean_bestk_rmse=("bestk_sample_rmse", "mean"),
                mean_rbf_rmse=("rbf_sample_rmse", "mean"),
                high_gap_rate=("high_gap_train_p75", "mean"),
            )
            .reset_index()
            .sort_values(["mean_gap", "n_samples"], ascending=[False, False])
        )

    outputs = {
        "overall_summary": overall,
        "thresholds": thresh_df,
        "correlations": corr,
        "bucket_summary": bucket_summary,
        "subject_summary": group_summary("subject"),
        "road_module_summary": group_summary("road_design_module_name"),
        "response_family_summary": group_summary("response_family_target"),
        "amplitude_bucket_summary": group_summary("amplitude_bucket"),
    }
    return outputs


def plot_top_gap(path: Path, top_gap: pd.DataFrame) -> None:
    top = top_gap.head(15).copy()
    labels = [sid.split("__")[-2] if "__" in sid else sid[-8:] for sid in top["sample_id"].astype(str)]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    x = np.arange(len(top))
    ax.bar(x, top["top1_bestk_gap"], color="#d62728", alpha=0.82, label="top1-bestK gap")
    ax.plot(x, top["top1_minus_rbf_rmse"], color="#1f77b4", marker="o", linewidth=1.1, label="top1-RBF")
    ax.set_xticks(x, labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("RMSE delta")
    ax.set_title("Largest top-1 vs best-of-K gaps on test")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_reliability_scatter(path: Path, detail: pd.DataFrame) -> None:
    df = detail[detail["split"] == "test"].copy()
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    colors = np.where(df["top1_matches_best"] == 1, "#2ca02c", "#d62728")
    ax.scatter(df["simple_reliability_risk_score"], df["top1_bestk_gap"], c=colors, s=52, alpha=0.82)
    ax.set_xlabel("Simple reliability risk score")
    ax.set_ylabel("Top1 - bestK sample RMSE")
    ax.set_title("Reliability risk vs top-K regret")
    ax.grid(True, alpha=0.24)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_branch_confusion(path: Path, detail: pd.DataFrame) -> None:
    df = detail[detail["split"] == "test"].copy()
    branches = sorted(set(df["top1_branch"].astype(int)).union(set(df["best_branch_oracle"].astype(int))))
    mat = np.zeros((len(branches), len(branches)), dtype=int)
    for _, row in df.iterrows():
        i = branches.index(int(row["top1_branch"]))
        j = branches.index(int(row["best_branch_oracle"]))
        mat[i, j] += 1
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(len(branches)), [f"best {b}" for b in branches])
    ax.set_yticks(range(len(branches)), [f"top1 {b}" for b in branches])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center", fontsize=11)
    ax.set_title("Top1 branch vs oracle branch on test")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_error_flags(path: Path, detail: pd.DataFrame) -> None:
    df = detail[detail["split"] == "test"].copy()
    groups = {
        "top1=best": df[df["top1_matches_best"] == 1],
        "top1!=best": df[df["top1_matches_best"] == 0],
    }
    fields = [
        ("top1_wrong_side", "wrong side"),
        ("top1_severe_amp_under", "under amp"),
        ("top1_tail_drift_risk", "tail drift"),
        ("top1_zero_crossing_mismatch", "zero mismatch"),
        ("top1_reversal_count_exact", "rev exact"),
    ]
    x = np.arange(len(fields))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for offset, (name, grp) in zip([-width / 2, width / 2], groups.items()):
        vals = [float(grp[col].mean()) if len(grp) else np.nan for col, _ in fields]
        ax.bar(x + offset, vals, width=width, label=name, alpha=0.82)
    ax.set_xticks(x, [label for _, label in fields], rotation=20, ha="right")
    ax.set_ylabel("Rate")
    ax.set_title("Top-1 physical errors by branch choice correctness")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_reports(detail: pd.DataFrame, outputs: dict[str, pd.DataFrame], figures: dict[str, str]) -> None:
    test_summary = outputs["overall_summary"][outputs["overall_summary"]["split"] == "test"].iloc[0]
    top_gap = detail[detail["split"] == "test"].sort_values("top1_bestk_gap", ascending=False)
    top1_bad = top_gap.iloc[0]

    user = f"""# 阶段 3 用户查看版：top-K top1/bestK 差距复盘 v0.1

## 这个阶段为什么做

top-K v0.1 的 best-of-3 很好，但 top-1 没有超过 RBF。这个阶段不训练新模型，只复盘“模型明明有好候选，但为什么 top-1 没选中”的样本和可靠性信号。

## 这个阶段检查了什么

- top-1 分支和 best-of-3 分支是否一致。
- top-1 与 best-of-3 的 RMSE 差距。
- top-1 概率、概率间隔、分支分散度是否能提示风险。
- 差距样本是否集中在某些被试、道路模块、响应类型或物理错误。

## 目前发现了什么

- test top-1 与 best-of-3 一致率={float(test_summary['top1_matches_best_rate']):.3f}。
- test 平均 top1-bestK gap={float(test_summary['mean_top1_bestk_gap']):.6f}。
- test 高风险分数捕捉高 gap 样本比例={float(test_summary['high_risk_captures_high_gap_rate']):.3f}。
- test 低置信规则捕捉高 gap 样本比例={float(test_summary['low_conf_rule_captures_high_gap_rate']):.3f}。
- 最大差距样本 `{top1_bad['sample_id']}` 的 top1-bestK gap={float(top1_bad['top1_bestk_gap']):.6f}。

## 哪些结果可信

可信的是：top-K v0.1 的主要瓶颈不是完全没有候选，而是选择头和可靠性判断不足；这由逐样本 bestK、top1 分支、概率和物理错误共同支持。

## 哪些结果还不能下结论

不能说当前可靠性规则已经可部署。这里的简单风险分数只是诊断线索，下一步如果要使用，必须在 train/val 上固定规则后再 test 评价，不能用 test 重新调参。

## 下一阶段是否可以继续

可以继续阶段 3，建议下一步做“可靠性/选择头 v0.2”或“关键点条件多假设”。仍不能进入连续风格、生理或 EEG 增量结论。

## 推荐优先查看

1. `{figures['top_gap']}`
2. `{figures['risk_scatter']}`
3. `{figures['branch_confusion']}`
4. `{figures['error_flags']}`
5. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_gap_review_v0_1/tables/topk_gap_review_sample_detail.csv`
"""

    tech = f"""# 阶段 3 技术报告：top-K top1/bestK 差距复盘 v0.1

## 输入

- `topk_vehicle_transformer_branch_diagnostics.csv`
- `topk_vehicle_transformer_per_sample_metrics.csv`
- `sample_response_task_manifest.csv`
- `response_decomposition_sample_labels.csv`

未训练新模型，未使用 subject ID 作为训练特征，未使用生理、脑电、连续风格、服务器或服务器密码文件。

## 诊断规则

- `top1_bestk_gap = rmse_top1_branch - rmse_bestk`
- 高 gap 阈值来自 train split 的 75 分位。
- 简单风险分数使用 train split 标准化：`-z(top1_prob) - z(prob_margin) + z(branch_spread_mean) + z(branch_spread_peak)`。
- 高风险阈值来自 train split 的 75 分位。

## test 摘要

| 指标 | 数值 |
|---|---:|
| top1 与 bestK 一致率 | {float(test_summary['top1_matches_best_rate']):.6f} |
| 平均 top1-bestK gap | {float(test_summary['mean_top1_bestk_gap']):.6f} |
| 平均 top1-RBF RMSE 差 | {float(test_summary['mean_top1_minus_rbf_rmse']):.6f} |
| 平均 bestK over RBF 收益 | {float(test_summary['mean_bestk_gain_over_rbf']):.6f} |
| 高风险捕捉高 gap 比例 | {float(test_summary['high_risk_captures_high_gap_rate']):.6f} |
| 低置信规则捕捉高 gap 比例 | {float(test_summary['low_conf_rule_captures_high_gap_rate']):.6f} |

## 结论

top-K v0.1 的下一步应集中在选择机制和可靠性估计，而不是继续把 best-of-K 上限当成模型效果。当前结果仍然阻塞连续风格、生理和 EEG 增量验证。
"""
    (REPORT_ROOT / "stage03_vehicle_instability_topk_gap_review_user_summary_cn.md").write_text(user, encoding="utf-8")
    (REPORT_ROOT / "stage03_vehicle_instability_topk_gap_review_v0_1_cn.md").write_text(tech, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    detail, thresholds = build_detail()
    outputs = summarize(detail, thresholds)

    detail.to_csv(TABLE_DIR / "topk_gap_review_sample_detail.csv", index=False, encoding="utf-8-sig")
    for name, table in outputs.items():
        table.to_csv(TABLE_DIR / f"topk_gap_review_{name}.csv", index=False, encoding="utf-8-sig")
    top_gap = detail[detail["split"] == "test"].sort_values("top1_bestk_gap", ascending=False)
    top_gap.head(20).to_csv(TABLE_DIR / "topk_gap_review_top_gap_samples.csv", index=False, encoding="utf-8-sig")
    top1_worse_rbf = detail[(detail["split"] == "test") & (detail["top1_minus_rbf_rmse"] > 0)].sort_values("top1_minus_rbf_rmse", ascending=False)
    top1_worse_rbf.to_csv(TABLE_DIR / "topk_gap_review_top1_worse_than_rbf_samples.csv", index=False, encoding="utf-8-sig")

    figures = {
        "top_gap": str(FIG_DIR / "topk_gap_top_samples.png").replace("\\", "/"),
        "risk_scatter": str(FIG_DIR / "topk_gap_risk_scatter.png").replace("\\", "/"),
        "branch_confusion": str(FIG_DIR / "topk_gap_branch_confusion.png").replace("\\", "/"),
        "error_flags": str(FIG_DIR / "topk_gap_error_flags.png").replace("\\", "/"),
    }
    plot_top_gap(Path(figures["top_gap"]), top_gap)
    plot_reliability_scatter(Path(figures["risk_scatter"]), detail)
    plot_branch_confusion(Path(figures["branch_confusion"]), detail)
    plot_error_flags(Path(figures["error_flags"]), detail)
    write_reports(detail, outputs, figures)

    summary = {
        "output_version": OUTPUT_VERSION,
        "input_version": "stage03_vehicle_instability_topk_vehicle_transformer_v0_1",
        "track_id": TRACK_ID,
        "server_used": False,
        "credential_file_read": False,
        "raw_files_modified": False,
        "thresholds": thresholds,
        "sample_detail_path": str(TABLE_DIR / "topk_gap_review_sample_detail.csv").replace("\\", "/"),
        "figures": figures,
    }
    (LOG_DIR / "topk_gap_review_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
