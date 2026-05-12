from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_ROOT = PROJECT_ROOT / "03_baselines"
REPORT_ROOT = PROJECT_ROOT / "09_reports"
ROBUSTNESS_ROOT = BASELINE_ROOT / "stage03_vehicle_instability_strong_vehicle_robustness_v0_1"
OUTPUT_ROOT = BASELINE_ROOT / "stage03_vehicle_instability_robustness_bad_sample_review_v0_1"
TABLE_DIR = OUTPUT_ROOT / "tables"
FIG_DIR = OUTPUT_ROOT / "figures"
LOG_DIR = OUTPUT_ROOT / "logs"

PER_SAMPLE_PATH = ROBUSTNESS_ROOT / "tables" / "strong_vehicle_robustness_per_sample_metrics.csv"
DECISION_PATH = ROBUSTNESS_ROOT / "tables" / "strong_vehicle_robustness_decision_table.csv"

ERROR_FLAGS = [
    "high_rmse_top20",
    "wrong_side_flag",
    "severe_amp_under_flag",
    "tail_drift_flag",
    "zero_crossing_mismatch_flag",
    "reversal_mismatch_flag",
    "multi_segment_mismatch_flag",
    "large_response_missed_flag",
    "peak_time_large_error_flag",
    "onset_delay_large_error_flag",
]


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIG_DIR, LOG_DIR, REPORT_ROOT]:
        path.mkdir(parents=True, exist_ok=True)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not PER_SAMPLE_PATH.exists():
        raise FileNotFoundError(PER_SAMPLE_PATH)
    per = pd.read_csv(PER_SAMPLE_PATH)
    decision = pd.read_csv(DECISION_PATH)
    return per, decision


def add_error_flags(per: pd.DataFrame) -> pd.DataFrame:
    out = per.copy()
    out = out[out["split"] == "test"].copy()
    out["config_model"] = out["robustness_config_id"].astype(str) + "::" + out["model_name"].astype(str)
    out["high_rmse_top20"] = False
    for _, idx in out.groupby("config_model").groups.items():
        vals = out.loc[idx, "sample_rmse"].astype(float)
        threshold = vals.quantile(0.8)
        out.loc[idx, "high_rmse_top20"] = vals >= threshold
    out["wrong_side_flag"] = out["wrong_side"].astype(int) == 1
    out["severe_amp_under_flag"] = out["severe_amp_under"].astype(int) == 1
    out["tail_drift_flag"] = out["tail_drift_risk"].astype(int) == 1
    out["zero_crossing_mismatch_flag"] = out["zero_crossing_mismatch"].astype(int) == 1
    out["reversal_mismatch_flag"] = out["reversal_count_exact"].astype(int) == 0
    out["multi_segment_mismatch_flag"] = (
        out["gt_multi_segment"].astype(int) != out["pred_multi_segment"].astype(int)
    )
    out["large_response_missed_flag"] = (
        (out["is_large_response"].astype(int) == 1)
        & (out["large_response_recalled"].astype(int) == 0)
    )
    out["peak_time_large_error_flag"] = out["peak_time_abs_error_s"].astype(float) >= 0.6
    out["onset_delay_large_error_flag"] = out["onset_delay_abs_error_s"].astype(float) >= 0.6
    return out


def build_recurrence(flagged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for event_uid, group in flagged.groupby("event_uid"):
        top = group[group["high_rmse_top20"]].copy()
        first = group.iloc[0]
        possible = group["config_model"].nunique()
        bad_models = top["config_model"].nunique()
        row = {
            "event_uid": event_uid,
            "subject": first.get("subject", ""),
            "session_stamp": first.get("session_stamp", ""),
            "possible_config_model_count": int(possible),
            "high_rmse_top20_count": int(bad_models),
            "high_rmse_top20_rate": float(bad_models / possible) if possible else np.nan,
            "mean_sample_rmse": float(group["sample_rmse"].mean()),
            "max_sample_rmse": float(group["sample_rmse"].max()),
            "mean_gt_peak_abs": float(group["gt_peak_abs"].mean()),
            "is_large_response_any": int(group["is_large_response"].astype(int).max()),
            "wrong_side_count": int(group["wrong_side_flag"].sum()),
            "severe_amp_under_count": int(group["severe_amp_under_flag"].sum()),
            "tail_drift_count": int(group["tail_drift_flag"].sum()),
            "zero_crossing_mismatch_count": int(group["zero_crossing_mismatch_flag"].sum()),
            "reversal_mismatch_count": int(group["reversal_mismatch_flag"].sum()),
            "multi_segment_mismatch_count": int(group["multi_segment_mismatch_flag"].sum()),
            "large_response_missed_count": int(group["large_response_missed_flag"].sum()),
            "peak_time_large_error_count": int(group["peak_time_large_error_flag"].sum()),
            "onset_delay_large_error_count": int(group["onset_delay_large_error_flag"].sum()),
            "configs_in_bad_top20": ";".join(sorted(top["robustness_config_id"].astype(str).unique())),
            "models_in_bad_top20": ";".join(sorted(top["model_name"].astype(str).unique())),
        }
        rows.append(row)
    out = pd.DataFrame(rows)
    out = out.sort_values(
        ["high_rmse_top20_count", "max_sample_rmse", "mean_sample_rmse"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    out.insert(0, "recurrence_rank", np.arange(1, len(out) + 1))
    return out


def build_error_summary(flagged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (config, model), group in flagged.groupby(["robustness_config_id", "model_name"]):
        row = {
            "robustness_config_id": config,
            "model_name": model,
            "n_samples": int(len(group)),
            "mean_sample_rmse": float(group["sample_rmse"].mean()),
        }
        for flag in ERROR_FLAGS:
            row[f"{flag}_count"] = int(group[flag].sum())
            row[f"{flag}_rate"] = float(group[flag].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["robustness_config_id", "model_name"]).reset_index(drop=True)


def build_subject_summary(flagged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subject, group in flagged.groupby("subject"):
        row = {
            "subject": subject,
            "n_rows": int(len(group)),
            "n_events": int(group["event_uid"].nunique()),
            "mean_sample_rmse": float(group["sample_rmse"].mean()),
            "high_rmse_top20_rate": float(group["high_rmse_top20"].mean()),
            "wrong_side_rate": float(group["wrong_side_flag"].mean()),
            "severe_amp_under_rate": float(group["severe_amp_under_flag"].mean()),
            "tail_drift_rate": float(group["tail_drift_flag"].mean()),
            "reversal_mismatch_rate": float(group["reversal_mismatch_flag"].mean()),
            "multi_segment_mismatch_rate": float(group["multi_segment_mismatch_flag"].mean()),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["high_rmse_top20_rate", "mean_sample_rmse"], ascending=[False, False]
    ).reset_index(drop=True)


def build_matrix(flagged: pd.DataFrame, recurrence: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    top_events = recurrence.head(top_n)["event_uid"].tolist()
    sub = flagged[flagged["event_uid"].isin(top_events)].copy()
    sub["cell"] = np.where(sub["high_rmse_top20"], sub["sample_rmse"].round(3).astype(str), "")
    matrix = sub.pivot_table(
        index="event_uid",
        columns="config_model",
        values="cell",
        aggfunc=lambda vals: next((v for v in vals if v != ""), ""),
        fill_value="",
    )
    matrix = matrix.reindex(top_events)
    matrix.insert(0, "subject", recurrence.set_index("event_uid").loc[top_events, "subject"])
    matrix.insert(1, "high_rmse_top20_count", recurrence.set_index("event_uid").loc[top_events, "high_rmse_top20_count"])
    return matrix.reset_index()


def representative_samples(recurrence: pd.DataFrame, flagged: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    rows = []
    for _, rec in recurrence.head(top_n).iterrows():
        event_rows = flagged[flagged["event_uid"] == rec["event_uid"]].sort_values(
            "sample_rmse", ascending=False
        )
        worst = event_rows.iloc[0]
        rows.append(
            {
                "recurrence_rank": rec["recurrence_rank"],
                "event_uid": rec["event_uid"],
                "subject": rec["subject"],
                "session_stamp": rec["session_stamp"],
                "high_rmse_top20_count": rec["high_rmse_top20_count"],
                "high_rmse_top20_rate": rec["high_rmse_top20_rate"],
                "worst_config": worst["robustness_config_id"],
                "worst_model": worst["model_name"],
                "worst_sample_id": worst["sample_id"],
                "worst_sample_rmse": float(worst["sample_rmse"]),
                "worst_gt_peak_abs": float(worst["gt_peak_abs"]),
                "worst_pred_peak_abs": float(worst["pred_peak_abs"]),
                "wrong_side_any": int(event_rows["wrong_side_flag"].any()),
                "severe_amp_under_any": int(event_rows["severe_amp_under_flag"].any()),
                "tail_drift_any": int(event_rows["tail_drift_flag"].any()),
                "reversal_mismatch_any": int(event_rows["reversal_mismatch_flag"].any()),
                "multi_segment_mismatch_any": int(event_rows["multi_segment_mismatch_flag"].any()),
                "large_response_missed_any": int(event_rows["large_response_missed_flag"].any()),
            }
        )
    return pd.DataFrame(rows)


def plot_recurrence(recurrence: pd.DataFrame) -> None:
    top = recurrence.head(25).copy().iloc[::-1]
    labels = top["subject"].astype(str) + " | " + top["event_uid"].astype(str).str[-16:]
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    ax.barh(labels, top["high_rmse_top20_count"], color="#355c9a")
    ax.set_xlabel("top20 high-RMSE occurrence count")
    ax.set_title("Most recurrent bad events across robustness configs/models")
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(FIG_DIR / "robustness_recurrent_bad_events.png", dpi=180)
    plt.close(fig)


def plot_error_heatmap(error_summary: pd.DataFrame) -> None:
    flag_rate_cols = [f"{flag}_rate" for flag in ERROR_FLAGS[1:]]
    sub = error_summary.copy()
    sub["row_label"] = sub["robustness_config_id"] + "::" + sub["model_name"].str.replace("_context_no_subject", "", regex=False)
    mat = sub[flag_rate_cols].astype(float)
    fig, ax = plt.subplots(figsize=(14, 8), constrained_layout=True)
    im = ax.imshow(mat.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_yticks(range(len(sub)), sub["row_label"], fontsize=7)
    ax.set_xticks(range(len(flag_rate_cols)), [c.replace("_rate", "") for c in flag_rate_cols], rotation=35, ha="right")
    ax.set_title("Physical error rates by robustness config and model")
    fig.colorbar(im, ax=ax, label="error rate")
    fig.savefig(FIG_DIR / "robustness_error_flag_heatmap.png", dpi=180)
    plt.close(fig)


def plot_subject_summary(subject_summary: pd.DataFrame) -> None:
    top = subject_summary.head(20).copy().iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    ax.barh(top["subject"], top["high_rmse_top20_rate"], color="#8f3f4a")
    ax.set_xlabel("top20 high-RMSE row rate")
    ax.set_title("Subjects with highest recurrent bad-sample rates")
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(FIG_DIR / "robustness_subject_bad_rate.png", dpi=180)
    plt.close(fig)


def plot_bad_matrix(matrix: pd.DataFrame) -> None:
    config_cols = [c for c in matrix.columns if "::" in c]
    values = matrix[config_cols].astype(str).ne("").to_numpy(dtype=int)
    row_labels = matrix["subject"].astype(str) + " | " + matrix["event_uid"].astype(str).str[-14:]
    fig, ax = plt.subplots(figsize=(14, max(7, 0.28 * len(matrix))), constrained_layout=True)
    im = ax.imshow(values, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_yticks(range(len(matrix)), row_labels, fontsize=7)
    ax.set_xticks(range(len(config_cols)), [c.replace("_context_no_subject", "") for c in config_cols], rotation=45, ha="right", fontsize=7)
    ax.set_title("Top recurrent bad-event matrix (1 = top20 high RMSE)")
    fig.colorbar(im, ax=ax, label="bad top20")
    fig.savefig(FIG_DIR / "robustness_bad_event_matrix.png", dpi=180)
    plt.close(fig)


def table_to_md(df: pd.DataFrame, cols: list[str], n: int = 10) -> str:
    sub = df[cols].head(n).copy()
    for col in sub.columns:
        if pd.api.types.is_float_dtype(sub[col]):
            sub[col] = sub[col].map(lambda x: f"{x:.6f}")
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in sub.values]
    return "\n".join([header, sep] + rows)


def write_reports(
    recurrence: pd.DataFrame,
    representatives: pd.DataFrame,
    error_summary: pd.DataFrame,
    subject_summary: pd.DataFrame,
) -> None:
    top = recurrence.iloc[0]
    top_rep = representatives.iloc[0]
    report = f"""# 阶段 3 稳健性坏样本复盘 v0.1

生成时间：2026-05-12

## 目的

本轮不训练新模型，而是从强车辆稳健性逐样本指标中找出跨配置、跨模型反复失败的事件。目标是区分：当前问题更像事件锚点/样本质量问题、车辆历史信息不足，还是模型结构无法表达反向修正和多段修正。

## 方法

- 输入：`{PER_SAMPLE_PATH.as_posix()}`。
- 只使用 test split 的逐样本指标。
- 对每个 `robustness_config_id::model_name` 单独取 sample RMSE top20% 作为高误差事件。
- 按 `event_uid` 聚合跨模型、跨窗口、跨 split 的复发次数。
- 统计错侧、严重幅值不足、尾段漂移、零线穿越错误、反向修正不匹配、多段修正不匹配、大幅响应漏召回、峰值时间大误差、启动延迟大误差。

## 主要发现

- 复发最高的事件是 `{top['event_uid']}`，subject=`{top['subject']}`，进入 top20 高误差的 config-model 次数为 {int(top['high_rmse_top20_count'])}/{int(top['possible_config_model_count'])}。
- 该事件最差的一次出现在 `{top_rep['worst_config']}` + `{top_rep['worst_model']}`，sample RMSE={top_rep['worst_sample_rmse']:.6f}。
- 高频坏样本不是只由单一模型造成；需要优先画这些事件的原始车辆轨迹、锚点、方向盘标签和预测曲线。

## 复发坏样本 Top10

{table_to_md(representatives, [
    "recurrence_rank",
    "event_uid",
    "subject",
    "high_rmse_top20_count",
    "worst_config",
    "worst_model",
    "worst_sample_rmse",
    "wrong_side_any",
    "severe_amp_under_any",
    "reversal_mismatch_any",
    "multi_segment_mismatch_any",
])}

## 分被试坏样本率 Top10

{table_to_md(subject_summary, [
    "subject",
    "n_events",
    "high_rmse_top20_rate",
    "mean_sample_rmse",
    "wrong_side_rate",
    "severe_amp_under_rate",
    "reversal_mismatch_rate",
    "multi_segment_mismatch_rate",
])}

## 产物

- 复发坏样本总表：`{(TABLE_DIR / "robustness_bad_event_recurrence.csv").as_posix()}`
- 代表坏样本表：`{(TABLE_DIR / "robustness_representative_bad_events.csv").as_posix()}`
- 物理错误汇总：`{(TABLE_DIR / "robustness_error_flag_summary_by_config_model.csv").as_posix()}`
- 分被试汇总：`{(TABLE_DIR / "robustness_subject_bad_summary.csv").as_posix()}`
- 坏样本矩阵：`{(TABLE_DIR / "robustness_bad_event_matrix.csv").as_posix()}`
- 复发事件图：`{(FIG_DIR / "robustness_recurrent_bad_events.png").as_posix()}`
- 物理错误热图：`{(FIG_DIR / "robustness_error_flag_heatmap.png").as_posix()}`
- 分被试坏样本率图：`{(FIG_DIR / "robustness_subject_bad_rate.png").as_posix()}`
- 坏样本矩阵图：`{(FIG_DIR / "robustness_bad_event_matrix.png").as_posix()}`

## 下一步

下一步应对代表坏样本表中的前 10-20 个事件画原始车辆时序、锚点、GT 方向盘响应和主要候选预测曲线。只有确认失败不是锚点错误或样本质量问题后，才应进入结构化车辆模型设计。
"""

    user_summary = f"""# 阶段 3 用户查看版：稳健性坏样本复盘

## 为什么做

前面发现 RBF/KNN 在多个切分和窗口下 RMSE 都能压过 formal ridge，但它们仍可能只是记住了相似模板，或者只改善普通样本。这个阶段专门找“反复失败”的事件。

## 检查了什么

- 每个模型/配置中 RMSE 最高的 top20% 样本。
- 哪些事件在多个模型、多个窗口、多个切分中反复成为坏样本。
- 坏样本里常见的是错侧、幅值不足、尾段漂移、反向修正错误还是多段修正错误。

## 目前发现

复发最高的坏事件是 `{top['event_uid']}`，subject=`{top['subject']}`，在 {int(top['high_rmse_top20_count'])}/{int(top['possible_config_model_count'])} 个 config-model 对照中进入 top20 高误差。

## 哪些结果可信

这个复盘只读取已经生成的逐样本指标，不训练新模型，不使用生理、脑电、连续风格或驾驶员 ID。它适合用来决定下一步优先看哪些坏样本。

## 哪些还不能下结论

现在还不能说这些坏样本一定是模型结构问题。它们也可能来自事件锚点偏差、标签窗口没有覆盖完整响应、或原始车辆数据局部异常。必须继续画原始波形和预测曲线确认。

## 下一步

优先对代表坏样本表前 10-20 个事件画详细曲线：事件锚点、车辆姿态、方向盘 GT、RBF/KNN/template 预测；Transformer 只作为已经单独跑过的参照，必要时另行叠加。确认问题来源后，再决定结构化响应模型怎么设计。

## 推荐优先查看

1. `{(TABLE_DIR / "robustness_representative_bad_events.csv").as_posix()}`
2. `{(TABLE_DIR / "robustness_bad_event_recurrence.csv").as_posix()}`
3. `{(FIG_DIR / "robustness_recurrent_bad_events.png").as_posix()}`
4. `{(FIG_DIR / "robustness_error_flag_heatmap.png").as_posix()}`
5. `{(FIG_DIR / "robustness_bad_event_matrix.png").as_posix()}`
"""

    (REPORT_ROOT / "stage03_vehicle_instability_robustness_bad_sample_review_v0_1_cn.md").write_text(
        report, encoding="utf-8"
    )
    (REPORT_ROOT / "stage03_vehicle_instability_robustness_bad_sample_review_user_summary_cn.md").write_text(
        user_summary, encoding="utf-8"
    )


def main() -> None:
    ensure_dirs()
    per, decision = load_inputs()
    flagged = add_error_flags(per)
    recurrence = build_recurrence(flagged)
    error_summary = build_error_summary(flagged)
    subject_summary = build_subject_summary(flagged)
    matrix = build_matrix(flagged, recurrence, top_n=30)
    reps = representative_samples(recurrence, flagged, top_n=30)

    flagged.to_csv(TABLE_DIR / "robustness_test_per_sample_with_error_flags.csv", index=False, encoding="utf-8-sig")
    recurrence.to_csv(TABLE_DIR / "robustness_bad_event_recurrence.csv", index=False, encoding="utf-8-sig")
    error_summary.to_csv(TABLE_DIR / "robustness_error_flag_summary_by_config_model.csv", index=False, encoding="utf-8-sig")
    subject_summary.to_csv(TABLE_DIR / "robustness_subject_bad_summary.csv", index=False, encoding="utf-8-sig")
    matrix.to_csv(TABLE_DIR / "robustness_bad_event_matrix.csv", index=False, encoding="utf-8-sig")
    reps.to_csv(TABLE_DIR / "robustness_representative_bad_events.csv", index=False, encoding="utf-8-sig")

    plot_recurrence(recurrence)
    plot_error_heatmap(error_summary)
    plot_subject_summary(subject_summary)
    plot_bad_matrix(matrix)
    write_reports(recurrence, reps, error_summary, subject_summary)

    top = recurrence.iloc[0]
    summary = {
        "source_per_sample": str(PER_SAMPLE_PATH),
        "n_test_rows": int(len(flagged)),
        "n_unique_events": int(flagged["event_uid"].nunique()),
        "n_config_models": int(flagged["config_model"].nunique()),
        "top_event_uid": str(top["event_uid"]),
        "top_event_subject": str(top["subject"]),
        "top_event_high_rmse_top20_count": int(top["high_rmse_top20_count"]),
        "top_event_possible_config_model_count": int(top["possible_config_model_count"]),
        "server_used": False,
        "credential_file_read": False,
        "uses_subject_id": False,
        "uses_physio": False,
        "uses_eeg": False,
        "uses_continuous_style": False,
        "raw_files_modified": False,
        "representative_bad_events": str(TABLE_DIR / "robustness_representative_bad_events.csv"),
    }
    (LOG_DIR / "robustness_bad_sample_review_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
