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
IN_DIR = ROOT / "03_baselines" / "stage03_vehicle_instability_keypoint_residual_vehicle_transformer_v0_1"
PER_SAMPLE_PATH = IN_DIR / "tables" / "keypoint_residual_vehicle_transformer_per_sample_metrics.csv"
OUT_ROOT = ROOT / "03_baselines" / "stage03_vehicle_instability_keypoint_vs_rbf_bad_sample_review_v0_1"
TABLE_DIR = OUT_ROOT / "tables"
FIG_DIR = OUT_ROOT / "figures"
LOG_DIR = OUT_ROOT / "logs"
REPORT_ROOT = ROOT / "09_reports"

TRACK_ID = "B_response3s_strict_core"
RBF_MODEL = "rbf_kernel_ridge_context_no_subject"
KEYPOINT_MODEL = "keypoint_residual_vehicle_transformer_no_subject"


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIG_DIR, LOG_DIR, REPORT_ROOT]:
        path.mkdir(parents=True, exist_ok=True)


def load_pair_table() -> pd.DataFrame:
    df = pd.read_csv(PER_SAMPLE_PATH)
    df = df[
        (df["track_id"] == TRACK_ID)
        & (df["split"] == "test")
        & (df["model_name"].isin([RBF_MODEL, KEYPOINT_MODEL]))
    ].copy()
    if df.empty:
        raise RuntimeError("no B-track test rows found")
    metrics = [
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
    ]
    id_cols = [
        "sample_id",
        "event_uid",
        "subject",
        "session_stamp",
        "window_config_id",
        "gt_peak_abs",
        "gt_reversal_count",
        "gt_multi_segment",
        "is_large_response",
        "is_difficult_peak_top20",
    ]
    base = df[id_cols].drop_duplicates("sample_id").set_index("sample_id")
    wide = df.pivot(index="sample_id", columns="model_name", values=metrics)
    wide.columns = [f"{metric}__{model}" for metric, model in wide.columns]
    out = base.join(wide).reset_index()
    for metric in metrics:
        out[f"{metric}__delta_keypoint_minus_rbf"] = out[f"{metric}__{KEYPOINT_MODEL}"] - out[f"{metric}__{RBF_MODEL}"]
    out["rmse_change"] = np.select(
        [
            out["sample_rmse__delta_keypoint_minus_rbf"] <= -0.05,
            out["sample_rmse__delta_keypoint_minus_rbf"] >= 0.05,
        ],
        ["improved", "degraded"],
        default="similar",
    )
    out["direction_change"] = np.select(
        [
            (out[f"wrong_side__{RBF_MODEL}"] == 1) & (out[f"wrong_side__{KEYPOINT_MODEL}"] == 0),
            (out[f"wrong_side__{RBF_MODEL}"] == 0) & (out[f"wrong_side__{KEYPOINT_MODEL}"] == 1),
        ],
        ["fixed_wrong_side", "new_wrong_side"],
        default="unchanged",
    )
    out["large_recall_change"] = np.select(
        [
            (out[f"large_response_recalled__{RBF_MODEL}"] == 0) & (out[f"large_response_recalled__{KEYPOINT_MODEL}"] == 1),
            (out[f"large_response_recalled__{RBF_MODEL}"] == 1) & (out[f"large_response_recalled__{KEYPOINT_MODEL}"] == 0),
        ],
        ["fixed_large_recall", "lost_large_recall"],
        default="unchanged",
    )
    out["tail_drift_change"] = np.select(
        [
            (out[f"tail_drift_risk__{RBF_MODEL}"] == 1) & (out[f"tail_drift_risk__{KEYPOINT_MODEL}"] == 0),
            (out[f"tail_drift_risk__{RBF_MODEL}"] == 0) & (out[f"tail_drift_risk__{KEYPOINT_MODEL}"] == 1),
        ],
        ["fixed_tail_drift", "new_tail_drift"],
        default="unchanged",
    )
    out["amp_under_change"] = np.select(
        [
            (out[f"severe_amp_under__{RBF_MODEL}"] == 1) & (out[f"severe_amp_under__{KEYPOINT_MODEL}"] == 0),
            (out[f"severe_amp_under__{RBF_MODEL}"] == 0) & (out[f"severe_amp_under__{KEYPOINT_MODEL}"] == 1),
        ],
        ["fixed_under_amp", "new_under_amp"],
        default="unchanged",
    )
    out["peak_time_change"] = np.select(
        [
            out["peak_time_abs_error_s__delta_keypoint_minus_rbf"] <= -0.10,
            out["peak_time_abs_error_s__delta_keypoint_minus_rbf"] >= 0.10,
        ],
        ["improved_peak_time", "degraded_peak_time"],
        default="similar",
    )
    out["onset_change"] = np.select(
        [
            out["onset_delay_abs_error_s__delta_keypoint_minus_rbf"] <= -0.10,
            out["onset_delay_abs_error_s__delta_keypoint_minus_rbf"] >= 0.10,
        ],
        ["improved_onset", "degraded_onset"],
        default="similar",
    )
    return out


def summarize_changes(delta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for col in ["rmse_change", "direction_change", "large_recall_change", "tail_drift_change", "amp_under_change", "peak_time_change", "onset_change"]:
        counts = delta[col].value_counts().to_dict()
        for label, n in sorted(counts.items()):
            rows.append({"change_type": col, "label": label, "n_samples": int(n), "rate": float(n / len(delta))})
    change_counts = pd.DataFrame(rows)
    subject_summary = (
        delta.groupby("subject")
        .agg(
            n_samples=("sample_id", "count"),
            rmse_delta_mean=("sample_rmse__delta_keypoint_minus_rbf", "mean"),
            rmse_improved_rate=("rmse_change", lambda x: float((x == "improved").mean())),
            wrong_side_fixed=("direction_change", lambda x: int((x == "fixed_wrong_side").sum())),
            new_wrong_side=("direction_change", lambda x: int((x == "new_wrong_side").sum())),
            large_recall_fixed=("large_recall_change", lambda x: int((x == "fixed_large_recall").sum())),
            large_recall_lost=("large_recall_change", lambda x: int((x == "lost_large_recall").sum())),
        )
        .reset_index()
        .sort_values(["rmse_delta_mean", "n_samples"], ascending=[True, False])
    )
    event_summary = pd.DataFrame(
        [
            {
                "n_test_samples": int(len(delta)),
                "rmse_delta_mean_keypoint_minus_rbf": float(delta["sample_rmse__delta_keypoint_minus_rbf"].mean()),
                "rmse_delta_median_keypoint_minus_rbf": float(delta["sample_rmse__delta_keypoint_minus_rbf"].median()),
                "rmse_improved_n": int((delta["rmse_change"] == "improved").sum()),
                "rmse_degraded_n": int((delta["rmse_change"] == "degraded").sum()),
                "wrong_side_fixed_n": int((delta["direction_change"] == "fixed_wrong_side").sum()),
                "new_wrong_side_n": int((delta["direction_change"] == "new_wrong_side").sum()),
                "large_recall_fixed_n": int((delta["large_recall_change"] == "fixed_large_recall").sum()),
                "large_recall_lost_n": int((delta["large_recall_change"] == "lost_large_recall").sum()),
                "tail_drift_fixed_n": int((delta["tail_drift_change"] == "fixed_tail_drift").sum()),
                "new_tail_drift_n": int((delta["tail_drift_change"] == "new_tail_drift").sum()),
            }
        ]
    )
    return change_counts, subject_summary, event_summary


def plot_delta(delta: pd.DataFrame, change_counts: pd.DataFrame) -> tuple[Path, Path]:
    ordered = pd.concat(
        [
            delta.nsmallest(10, "sample_rmse__delta_keypoint_minus_rbf"),
            delta.nlargest(10, "sample_rmse__delta_keypoint_minus_rbf"),
        ],
        ignore_index=True,
    )
    ordered = ordered.drop_duplicates("sample_id")
    labels = [f"{r.subject}\n{float(r.sample_rmse__delta_keypoint_minus_rbf):+.2f}" for r in ordered.itertuples()]
    colors = ["#2ca02c" if v < 0 else "#d62728" for v in ordered["sample_rmse__delta_keypoint_minus_rbf"]]
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.bar(np.arange(len(ordered)), ordered["sample_rmse__delta_keypoint_minus_rbf"], color=colors)
    ax.axhline(0, color="#333333", linewidth=0.9)
    ax.set_xticks(np.arange(len(ordered)))
    ax.set_xticklabels(labels, rotation=0, fontsize=8)
    ax.set_ylabel("RMSE delta: keypoint - RBF")
    ax.set_title("B track: keypoint residual vs RBF per-sample RMSE delta")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    delta_fig = FIG_DIR / "keypoint_vs_rbf_rmse_delta_top_samples.png"
    fig.savefig(delta_fig, dpi=170)
    plt.close(fig)

    focus = change_counts[change_counts["change_type"].isin(["direction_change", "large_recall_change", "tail_drift_change", "amp_under_change", "peak_time_change", "onset_change"])].copy()
    focus["name"] = focus["change_type"].str.replace("_change", "", regex=False) + "\n" + focus["label"]
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.bar(focus["name"], focus["n_samples"], color="#4c78a8")
    ax.set_ylabel("n test samples")
    ax.set_title("B track: keypoint residual vs RBF error-change counts")
    ax.tick_params(axis="x", labelrotation=35, labelsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    change_fig = FIG_DIR / "keypoint_vs_rbf_error_change_counts.png"
    fig.savefig(change_fig, dpi=170)
    plt.close(fig)
    return delta_fig, change_fig


def write_reports(delta: pd.DataFrame, change_counts: pd.DataFrame, subject_summary: pd.DataFrame, event_summary: pd.DataFrame, figures: dict[str, str]) -> None:
    top_improved = delta.nsmallest(8, "sample_rmse__delta_keypoint_minus_rbf")[
        ["sample_id", "subject", "sample_rmse__delta_keypoint_minus_rbf", "direction_change", "large_recall_change", "tail_drift_change", "peak_time_change", "onset_change"]
    ]
    top_degraded = delta.nlargest(8, "sample_rmse__delta_keypoint_minus_rbf")[
        ["sample_id", "subject", "sample_rmse__delta_keypoint_minus_rbf", "direction_change", "large_recall_change", "tail_drift_change", "peak_time_change", "onset_change"]
    ]
    one = event_summary.iloc[0]
    report = f"""# 阶段 3：keypoint+residual vs RBF 坏样本差异复盘 v0.1

生成时间：2026-05-13

## 为什么做

keypoint+residual 在 B 轨道 test 上 RMSE 仍略差于 RBF KRR，但错侧率和大幅响应召回更好。因此需要逐样本检查它到底修复了哪些物理错误、又在哪些样本上退化。

## 输入

- 逐样本指标：`{PER_SAMPLE_PATH.as_posix()}`
- 范围：B 轨道 `response3s_strict_core_candidate` 的 session-level test 40 个样本。
- 对照：`{KEYPOINT_MODEL}` vs `{RBF_MODEL}`。
- 本轮只读已有车辆-only 评估表，不训练模型，不使用生理、脑电、连续风格、服务器或服务器密码文件。

## 总结

- test 样本数：{int(one['n_test_samples'])}
- keypoint - RBF 样本 RMSE 平均差：{float(one['rmse_delta_mean_keypoint_minus_rbf']):.6f}
- RMSE 明显改善样本数：{int(one['rmse_improved_n'])}
- RMSE 明显退化样本数：{int(one['rmse_degraded_n'])}
- 修复错侧样本数：{int(one['wrong_side_fixed_n'])}
- 新增错侧样本数：{int(one['new_wrong_side_n'])}
- 修复大幅响应召回样本数：{int(one['large_recall_fixed_n'])}
- 丢失大幅响应召回样本数：{int(one['large_recall_lost_n'])}
- 修复尾段漂移样本数：{int(one['tail_drift_fixed_n'])}
- 新增尾段漂移样本数：{int(one['new_tail_drift_n'])}

## 变化计数

```text
{change_counts.to_string(index=False)}
```

## 分被试摘要

```text
{subject_summary.to_string(index=False)}
```

## RMSE 改善最大的样本

```text
{top_improved.to_string(index=False)}
```

## RMSE 退化最大的样本

```text
{top_degraded.to_string(index=False)}
```

## 图

- RMSE 差异 Top 样本：`{figures.get('rmse_delta', '')}`
- 错误变化计数：`{figures.get('change_counts', '')}`

## 结论边界

这个复盘只说明 keypoint+residual 与 RBF 在 B 轨道 test 样本上的错误转移关系。它不能证明连续风格、生理或 EEG 有效，也不能替代后续多 seed/多切分验证。
"""
    (REPORT_ROOT / "stage03_vehicle_instability_keypoint_vs_rbf_bad_sample_review_v0_1_cn.md").write_text(report, encoding="utf-8")

    user = f"""# 阶段 3 用户查看版：keypoint+residual 和 RBF 的坏样本差异

## 为什么做

keypoint+residual 的整体 RMSE 还没有超过 RBF，但错侧率和大幅响应召回更好。所以这一步不是再训练，而是逐个样本看：它到底救了哪些样本，又弄坏了哪些样本。

## 这次检查了什么

- 只看 B 轨道 test 的 40 个样本。
- 对比 keypoint+residual 和 RBF KRR 的逐样本 RMSE、错侧、大幅响应召回、幅值不足、峰值时间、启动延迟、尾段漂移和反向修正。
- 不使用生理、脑电、连续风格，也不连接服务器。

## 目前发现

- keypoint - RBF 的样本 RMSE 平均差：{float(one['rmse_delta_mean_keypoint_minus_rbf']):.6f}，说明整体上 keypoint 仍略差。
- RMSE 明显改善 {int(one['rmse_improved_n'])} 个样本，明显退化 {int(one['rmse_degraded_n'])} 个样本。
- keypoint 修复错侧 {int(one['wrong_side_fixed_n'])} 个样本，新增错侧 {int(one['new_wrong_side_n'])} 个样本。
- keypoint 修复大幅响应召回 {int(one['large_recall_fixed_n'])} 个样本，丢失大幅响应召回 {int(one['large_recall_lost_n'])} 个样本。

## 哪些结果可信

可信的是：keypoint+residual 的收益主要体现在方向和大幅响应召回；但它不是全局压倒 RBF，因为 RMSE 和困难样本仍没有赢。

## 哪些还不能下结论

还不能说结构模型已经解决车辆-only 问题，更不能说生理或风格有效。下一步如果继续模型，应看多假设或可靠性识别是否能保住 keypoint 的方向/大幅响应收益，同时减少退化样本。

## 推荐优先查看

1. `{figures.get('rmse_delta', '')}`
2. `{figures.get('change_counts', '')}`
3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_keypoint_vs_rbf_bad_sample_review_v0_1/tables/keypoint_vs_rbf_sample_delta.csv`
"""
    (REPORT_ROOT / "stage03_vehicle_instability_keypoint_vs_rbf_bad_sample_review_user_summary_cn.md").write_text(user, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    delta = load_pair_table()
    change_counts, subject_summary, event_summary = summarize_changes(delta)
    delta_fig, change_fig = plot_delta(delta, change_counts)
    figures = {
        "rmse_delta": str(delta_fig).replace("\\", "/"),
        "change_counts": str(change_fig).replace("\\", "/"),
    }

    delta.sort_values("sample_rmse__delta_keypoint_minus_rbf").to_csv(TABLE_DIR / "keypoint_vs_rbf_sample_delta.csv", index=False, encoding="utf-8-sig")
    change_counts.to_csv(TABLE_DIR / "keypoint_vs_rbf_change_counts.csv", index=False, encoding="utf-8-sig")
    subject_summary.to_csv(TABLE_DIR / "keypoint_vs_rbf_subject_summary.csv", index=False, encoding="utf-8-sig")
    event_summary.to_csv(TABLE_DIR / "keypoint_vs_rbf_overall_summary.csv", index=False, encoding="utf-8-sig")
    delta.nsmallest(12, "sample_rmse__delta_keypoint_minus_rbf").to_csv(TABLE_DIR / "keypoint_vs_rbf_top_improved.csv", index=False, encoding="utf-8-sig")
    delta.nlargest(12, "sample_rmse__delta_keypoint_minus_rbf").to_csv(TABLE_DIR / "keypoint_vs_rbf_top_degraded.csv", index=False, encoding="utf-8-sig")

    summary = {
        "output_version": "stage03_vehicle_instability_keypoint_vs_rbf_bad_sample_review_v0_1",
        "track_id": TRACK_ID,
        "rbf_model": RBF_MODEL,
        "keypoint_model": KEYPOINT_MODEL,
        "n_test_samples": int(len(delta)),
        "overall_summary": event_summary.to_dict(orient="records")[0],
        "server_used": False,
        "credential_file_read": False,
        "raw_files_modified": False,
        "figures": figures,
        "tables": {
            "sample_delta": str(TABLE_DIR / "keypoint_vs_rbf_sample_delta.csv").replace("\\", "/"),
            "change_counts": str(TABLE_DIR / "keypoint_vs_rbf_change_counts.csv").replace("\\", "/"),
            "subject_summary": str(TABLE_DIR / "keypoint_vs_rbf_subject_summary.csv").replace("\\", "/"),
        },
    }
    (LOG_DIR / "keypoint_vs_rbf_bad_sample_review_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_reports(delta, change_counts, subject_summary, event_summary, figures)
    print(event_summary.to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
