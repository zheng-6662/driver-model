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
SAMPLE_ROOT = ROOT / "02_samples"
BASE_SAMPLE_DIR = SAMPLE_ROOT / "vehicle_instability_highconf_v0_1"
SAMPLES_PATH = BASE_SAMPLE_DIR / "tables" / "samples_master.csv"
AUDIT_ROOT = ROOT / "03_baselines" / "stage03_vehicle_instability_label_window_coverage_audit_v0_1"
EVENT_POLICY_PATH = AUDIT_ROOT / "tables" / "label_window_event_policy_table.csv"
OUT_ROOT = SAMPLE_ROOT / "vehicle_instability_response_task_decision_v0_1"
TABLE_DIR = OUT_ROOT / "tables"
FIG_DIR = OUT_ROOT / "figures"
LOG_DIR = OUT_ROOT / "logs"
REPORT_ROOT = ROOT / "09_reports"

PRE1 = "pre1_label2_event_trigger"
PRE2 = "pre2_label2_old_main"
PRE3 = "pre3_label3_response_coverage"

TASK_CLASS_CN = {
    "instant2s_core_clean": "2秒即时响应核心样本，2秒和3秒标签都相对稳定",
    "instant2s_ok_but_long_event_context": "2秒即时响应可用，但完整响应/长事件仍需复核",
    "switch_to_3s_late_peak_core": "2秒漏掉后续峰值，优先转为3秒响应覆盖候选",
    "switch_to_3s_continuing_response_core": "2秒后仍有明显变化，优先转为3秒响应覆盖候选",
    "manual_2s_tail_or_anchor_review": "2秒尾段或锚点需复核，暂不进核心训练",
    "long_or_unsettled_review": "3秒仍未稳定或长事件复杂，需回到样本规则复核/拆分",
}


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIG_DIR, LOG_DIR, REPORT_ROOT]:
        path.mkdir(parents=True, exist_ok=True)


def classify_event(row: pd.Series) -> str:
    label2 = bool(row["label_window_2s_needs_review"])
    label3 = bool(row["label3_still_needs_review"])
    late_peak = bool(row["old_2s_may_miss_future_peak"])
    post2_change = bool(row["old_2s_may_miss_future_change"])
    if not label2 and not label3:
        return "instant2s_core_clean"
    if not label2 and label3:
        return "instant2s_ok_but_long_event_context"
    if late_peak and not label3:
        return "switch_to_3s_late_peak_core"
    if post2_change and not label3:
        return "switch_to_3s_continuing_response_core"
    if label3:
        return "long_or_unsettled_review"
    return "manual_2s_tail_or_anchor_review"


def decision_fields(task_class: str) -> dict[str, Any]:
    include_instant2 = task_class in {
        "instant2s_core_clean",
        "instant2s_ok_but_long_event_context",
    }
    include_3s_candidate = task_class in {
        "instant2s_core_clean",
        "switch_to_3s_late_peak_core",
        "switch_to_3s_continuing_response_core",
        "manual_2s_tail_or_anchor_review",
    }
    include_3s_strict = task_class in {
        "instant2s_core_clean",
        "switch_to_3s_late_peak_core",
        "switch_to_3s_continuing_response_core",
    }
    long_review = task_class == "long_or_unsettled_review"
    manual_review = task_class in {
        "manual_2s_tail_or_anchor_review",
        "instant2s_ok_but_long_event_context",
        "long_or_unsettled_review",
    }
    if task_class == "instant2s_core_clean":
        track = "A_instant2s_core_and_3s_ok"
        action = "can_use_for_instant2s_core; also safe as 3s simple-response candidate"
    elif task_class == "instant2s_ok_but_long_event_context":
        track = "A_instant2s_only_with_long_event_flag"
        action = "can_use_for_2s_instant_response; do not use as complete-response claim without review"
    elif task_class in {"switch_to_3s_late_peak_core", "switch_to_3s_continuing_response_core"}:
        track = "B_response3s_core_candidate"
        action = "prefer 3s response-coverage label; rerun vehicle-only baseline before model escalation"
    elif task_class == "manual_2s_tail_or_anchor_review":
        track = "C_manual_2s_anchor_or_tail_review"
        action = "review anchor/tail before using as core sample"
    else:
        track = "D_long_event_or_unsettled_review"
        action = "return to stage2 sample rules; consider splitting onset response and sustained control"
    return {
        "response_task_track": track,
        "include_instant2s_core_candidate": include_instant2,
        "include_response3s_candidate": include_3s_candidate,
        "include_response3s_strict_core": include_3s_strict,
        "requires_manual_window_or_anchor_review": manual_review,
        "requires_long_event_split_review": long_review,
        "recommended_training_action": action,
    }


def build_event_decision(policy: pd.DataFrame) -> pd.DataFrame:
    out = policy.copy()
    out["response_task_class"] = out.apply(classify_event, axis=1)
    decisions = pd.DataFrame([decision_fields(v) for v in out["response_task_class"]])
    out = pd.concat([out.reset_index(drop=True), decisions], axis=1)
    out["response_task_class_cn"] = out["response_task_class"].map(TASK_CLASS_CN)
    out["do_not_claim_complete_response_without_review"] = (
        out["requires_manual_window_or_anchor_review"] | out["requires_long_event_split_review"]
    )
    out["stage3_main_recommendation_cn"] = np.select(
        [
            out["response_task_class"] == "instant2s_core_clean",
            out["response_task_class"] == "instant2s_ok_but_long_event_context",
            out["response_task_class"].isin(
                ["switch_to_3s_late_peak_core", "switch_to_3s_continuing_response_core"]
            ),
            out["response_task_class"] == "manual_2s_tail_or_anchor_review",
        ],
        [
            "可作为2秒即时响应核心候选，也可进入简单3秒响应候选。",
            "可作为2秒即时响应候选，但完整响应需要长事件复核。",
            "优先转入3秒响应覆盖候选，重跑车辆-only基线后再决定是否继续。",
            "先复核锚点或尾段，不进入核心训练结论。",
        ],
        default="先回到阶段2复核或拆分长事件，不进入核心训练结论。",
    )
    return out


def sample_role(row: pd.Series) -> str:
    window_id = str(row["window_config_id"])
    if window_id == PRE2 and bool(row["include_instant2s_core_candidate"]):
        return "instant2s_core_candidate"
    if window_id == PRE3 and bool(row["include_response3s_strict_core"]):
        return "response3s_strict_core_candidate"
    if window_id == PRE3 and bool(row["include_response3s_candidate"]):
        return "response3s_review_candidate"
    if window_id == PRE1 and bool(row["include_instant2s_core_candidate"]):
        return "early1s_control_for_instant2s"
    if bool(row["requires_long_event_split_review"]):
        return "long_event_review_holdout"
    if bool(row["requires_manual_window_or_anchor_review"]):
        return "manual_window_anchor_review_holdout"
    return "not_primary_for_next_stage"


def build_sample_decision(samples: pd.DataFrame, event_decision: pd.DataFrame) -> pd.DataFrame:
    decision_cols = [
        "event_uid",
        "response_task_class",
        "response_task_class_cn",
        "response_task_track",
        "include_instant2s_core_candidate",
        "include_response3s_candidate",
        "include_response3s_strict_core",
        "requires_manual_window_or_anchor_review",
        "requires_long_event_split_review",
        "do_not_claim_complete_response_without_review",
        "recommended_training_action",
        "stage3_main_recommendation_cn",
        "label_window_2s_needs_review",
        "label3_still_needs_review",
        "old_2s_may_miss_future_peak",
        "old_2s_may_miss_future_change",
        "recommended_window_policy",
    ]
    out = samples.merge(event_decision[decision_cols], on="event_uid", how="left")
    if out["response_task_class"].isna().any():
        missing = int(out["response_task_class"].isna().sum())
        raise RuntimeError(f"Missing event task decision for {missing} sample rows")
    out["task_sample_role"] = out.apply(sample_role, axis=1)
    out["recommended_for_next_vehicle_baseline"] = out["task_sample_role"].isin(
        [
            "instant2s_core_candidate",
            "response3s_strict_core_candidate",
            "response3s_review_candidate",
            "early1s_control_for_instant2s",
        ]
    )
    return out


def summarize(event_decision: pd.DataFrame, sample_decision: pd.DataFrame) -> dict[str, pd.DataFrame]:
    task_counts = (
        event_decision.groupby(["response_task_class", "response_task_track", "response_task_class_cn"], dropna=False)
        .size()
        .reset_index(name="n_events")
        .sort_values("n_events", ascending=False)
    )
    task_counts["rate"] = task_counts["n_events"] / len(event_decision)

    track_counts = (
        event_decision.groupby("response_task_track", dropna=False)
        .size()
        .reset_index(name="n_events")
        .sort_values("n_events", ascending=False)
    )
    track_counts["rate"] = track_counts["n_events"] / len(event_decision)

    sample_role_counts = (
        sample_decision.groupby(["window_config_id", "task_sample_role"], dropna=False)
        .size()
        .reset_index(name="n_samples")
        .sort_values(["window_config_id", "n_samples"], ascending=[True, False])
    )

    split_summary_rows = []
    for split_col in ["session_level_split", "subject_level_split", "random_event_split", "default_split"]:
        if split_col not in event_decision.columns:
            continue
        for split_name, part in event_decision.groupby(split_col, dropna=False):
            split_summary_rows.append(
                {
                    "split_strategy": split_col,
                    "split_name": split_name,
                    "n_events": int(len(part)),
                    "instant2s_core_candidate_n": int(part["include_instant2s_core_candidate"].sum()),
                    "response3s_strict_core_n": int(part["include_response3s_strict_core"].sum()),
                    "long_event_review_n": int(part["requires_long_event_split_review"].sum()),
                    "manual_review_n": int(part["requires_manual_window_or_anchor_review"].sum()),
                }
            )
    split_summary = pd.DataFrame(split_summary_rows)

    subject_summary = (
        event_decision.groupby("subject", dropna=False)
        .agg(
            n_events=("event_uid", "count"),
            instant2s_core_candidate_n=("include_instant2s_core_candidate", "sum"),
            response3s_strict_core_n=("include_response3s_strict_core", "sum"),
            long_event_review_n=("requires_long_event_split_review", "sum"),
            manual_review_n=("requires_manual_window_or_anchor_review", "sum"),
        )
        .reset_index()
    )
    subject_summary["long_event_review_rate"] = subject_summary["long_event_review_n"] / subject_summary["n_events"]
    subject_summary = subject_summary.sort_values(["long_event_review_rate", "n_events"], ascending=[False, False])
    return {
        "task_counts": task_counts,
        "track_counts": track_counts,
        "sample_role_counts": sample_role_counts,
        "split_summary": split_summary,
        "subject_summary": subject_summary,
    }


def plot_task_counts(task_counts: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(11, 5.8))
    part = task_counts.sort_values("n_events", ascending=True)
    ax.barh(part["response_task_class"], part["n_events"], color="#4c78a8")
    ax.set_xlabel("Events")
    ax.set_title("Response task decision counts")
    for i, (_, row) in enumerate(part.iterrows()):
        ax.text(float(row["n_events"]) + 3, i, f"{int(row['n_events'])} ({row['rate']:.1%})", va="center")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    out = FIG_DIR / "response_task_decision_counts.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_sample_roles(sample_role_counts: pd.DataFrame) -> Path:
    pivot = sample_role_counts.pivot(index="window_config_id", columns="task_sample_role", values="n_samples").fillna(0)
    fig, ax = plt.subplots(figsize=(12, 7.2))
    bottom = np.zeros(len(pivot), dtype=float)
    colors = plt.get_cmap("tab20").colors
    for j, col in enumerate(pivot.columns):
        vals = pivot[col].to_numpy(dtype=float)
        ax.bar(pivot.index, vals, bottom=bottom, label=col, color=colors[j % len(colors)])
        bottom += vals
    ax.set_ylabel("Samples")
    ax.set_title("Sample roles by existing window config")
    ax.tick_params(axis="x", rotation=18)
    ax.legend(fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=True)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    out = FIG_DIR / "response_task_sample_roles_by_window.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def code_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    part = df if max_rows is None else df.head(max_rows)
    return "```text\n" + part.to_string(index=False) + "\n```"


def write_reports(
    event_decision: pd.DataFrame,
    sample_decision: pd.DataFrame,
    summaries: dict[str, pd.DataFrame],
    figures: dict[str, Path],
) -> None:
    n_events = len(event_decision)
    instant2_n = int(event_decision["include_instant2s_core_candidate"].sum())
    resp3_strict_n = int(event_decision["include_response3s_strict_core"].sum())
    resp3_candidate_n = int(event_decision["include_response3s_candidate"].sum())
    long_review_n = int(event_decision["requires_long_event_split_review"].sum())
    manual_review_n = int(event_decision["requires_manual_window_or_anchor_review"].sum())
    sample_next_n = int(sample_decision["recommended_for_next_vehicle_baseline"].sum())

    user = f"""# 阶段 3 用户查看版：失稳样本响应任务定义决策 v0.1

## 为什么做

标签窗口覆盖审计显示，当前 2 秒标签经常没有覆盖完整方向盘响应，3 秒标签也有不少长事件仍未稳定。如果不先把任务定义拆清楚，继续训练模型会把“标签问题”和“模型能力问题”混在一起。

## 这个阶段做了什么

这一步没有改原始数据，也没有训练模型。它把 906 个高置信失稳事件分成几类：2 秒即时响应可用、应转 3 秒响应覆盖、2 秒尾段/锚点需复核、长事件或持续控制需回到阶段 2 复核。

## 当前决策数字

- 可作为 2 秒即时响应核心候选：{instant2_n}/{n_events}。
- 可作为 3 秒响应覆盖候选：{resp3_candidate_n}/{n_events}，其中严格核心候选 {resp3_strict_n}/{n_events}。
- 需要长事件/持续控制复核：{long_review_n}/{n_events}。
- 需要人工窗口或锚点复核：{manual_review_n}/{n_events}。
- 现有 2718 个窗口样本中，下一轮车辆-only 基线可优先使用的候选窗口样本为 {sample_next_n} 个。

## 现在应该怎么理解

2 秒标签不适合再被说成“完整响应预测”。它可以作为“事件触发后的即时响应”任务。3 秒标签更接近完整响应，但仍有大量长事件需要拆分或标记为持续控制。后续如果要训练强车辆模型，应至少并行保留 2 秒即时响应和 3 秒响应覆盖两个任务定义。

## 哪些还不能下结论

这个决策表只是规则覆盖层，不等于人工最终真值；长事件不一定是坏样本，可能是驾驶员真实持续控制。它也不能说明连续风格、生理或 EEG 有效。

## 推荐优先查看

1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_response_task_decision_v0_1/tables/event_response_task_decision_table.csv`
2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_response_task_decision_v0_1/tables/sample_response_task_manifest.csv`
3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_response_task_decision_v0_1/figures/response_task_decision_counts.png`
4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_response_task_decision_v0_1/figures/response_task_sample_roles_by_window.png`
"""
    (REPORT_ROOT / "stage03_vehicle_instability_response_task_decision_user_summary_cn.md").write_text(
        user, encoding="utf-8"
    )

    technical = f"""# 阶段 3：失稳样本响应任务定义决策 v0.1

## 输入

- 原始样本清单：`{SAMPLES_PATH.as_posix()}`
- 标签窗口覆盖审计表：`{EVENT_POLICY_PATH.as_posix()}`

## 决策原则

1. 不把 2 秒标签继续称为完整响应；它只能作为即时响应任务。
2. 2 秒后存在后续峰值或明显方向盘变化且 3 秒标签稳定的事件，转为 3 秒响应覆盖候选。
3. 3 秒仍未稳定的事件，不直接进入完整响应核心训练；优先回到阶段 2 复核或拆成启动响应/持续控制。
4. 所有这些决策只基于车辆标签窗口和样本规则，不涉及生理、脑电、连续风格或驾驶员 ID。

## 事件任务类别计数

{code_table(summaries['task_counts'])}

## 任务轨道计数

{code_table(summaries['track_counts'])}

## 样本窗口角色计数

{code_table(summaries['sample_role_counts'])}

## 核心数字

- 事件总数：{n_events}
- 2 秒即时响应核心候选：{instant2_n}
- 3 秒响应覆盖候选：{resp3_candidate_n}
- 3 秒严格核心候选：{resp3_strict_n}
- 长事件/持续控制复核：{long_review_n}
- 手动窗口/锚点复核：{manual_review_n}
- 下一轮车辆-only 基线优先候选窗口样本：{sample_next_n}

## 输出

- 事件级决策表：`{(TABLE_DIR / 'event_response_task_decision_table.csv').as_posix()}`
- 样本级任务 manifest：`{(TABLE_DIR / 'sample_response_task_manifest.csv').as_posix()}`
- 任务类别计数：`{(TABLE_DIR / 'response_task_decision_counts.csv').as_posix()}`
- split 汇总：`{(TABLE_DIR / 'response_task_split_summary.csv').as_posix()}`
- subject 汇总：`{(TABLE_DIR / 'response_task_subject_summary.csv').as_posix()}`
- 图：`{figures['task_counts'].as_posix()}`
- 图：`{figures['sample_roles'].as_posix()}`

## 下一步建议

先基于这个覆盖层重跑两个车辆-only 对照：A 轨道的 2 秒即时响应核心候选，以及 B 轨道的 3 秒响应覆盖核心候选。D 轨道长事件暂不进入最终主线训练，先做复核或拆分。
"""
    (REPORT_ROOT / "stage03_vehicle_instability_response_task_decision_v0_1_cn.md").write_text(
        technical, encoding="utf-8"
    )


def main() -> None:
    ensure_dirs()
    samples = pd.read_csv(SAMPLES_PATH)
    policy = pd.read_csv(EVENT_POLICY_PATH)
    event_decision = build_event_decision(policy)
    sample_decision = build_sample_decision(samples, event_decision)
    summaries = summarize(event_decision, sample_decision)

    event_decision.to_csv(TABLE_DIR / "event_response_task_decision_table.csv", index=False, encoding="utf-8-sig")
    sample_decision.to_csv(TABLE_DIR / "sample_response_task_manifest.csv", index=False, encoding="utf-8-sig")
    summaries["task_counts"].to_csv(TABLE_DIR / "response_task_decision_counts.csv", index=False, encoding="utf-8-sig")
    summaries["track_counts"].to_csv(TABLE_DIR / "response_task_track_counts.csv", index=False, encoding="utf-8-sig")
    summaries["sample_role_counts"].to_csv(TABLE_DIR / "response_task_sample_role_counts.csv", index=False, encoding="utf-8-sig")
    summaries["split_summary"].to_csv(TABLE_DIR / "response_task_split_summary.csv", index=False, encoding="utf-8-sig")
    summaries["subject_summary"].to_csv(TABLE_DIR / "response_task_subject_summary.csv", index=False, encoding="utf-8-sig")

    figures = {
        "task_counts": plot_task_counts(summaries["task_counts"]),
        "sample_roles": plot_sample_roles(summaries["sample_role_counts"]),
    }
    write_reports(event_decision, sample_decision, summaries, figures)

    summary = {
        "n_events": int(len(event_decision)),
        "n_sample_rows": int(len(sample_decision)),
        "instant2s_core_candidate_n": int(event_decision["include_instant2s_core_candidate"].sum()),
        "response3s_candidate_n": int(event_decision["include_response3s_candidate"].sum()),
        "response3s_strict_core_n": int(event_decision["include_response3s_strict_core"].sum()),
        "long_event_review_n": int(event_decision["requires_long_event_split_review"].sum()),
        "manual_window_or_anchor_review_n": int(event_decision["requires_manual_window_or_anchor_review"].sum()),
        "next_vehicle_baseline_window_sample_n": int(sample_decision["recommended_for_next_vehicle_baseline"].sum()),
        "event_decision_path": str(TABLE_DIR / "event_response_task_decision_table.csv").replace("\\", "/"),
        "sample_manifest_path": str(TABLE_DIR / "sample_response_task_manifest.csv").replace("\\", "/"),
        "figures": {k: str(v).replace("\\", "/") for k, v in figures.items()},
        "source_policy_path": str(EVENT_POLICY_PATH).replace("\\", "/"),
        "server_used": False,
        "credential_file_read": False,
        "uses_subject_id_as_model_input": False,
        "uses_physio": False,
        "uses_eeg": False,
        "uses_continuous_style": False,
        "model_training_performed": False,
        "raw_files_modified": False,
    }
    (LOG_DIR / "response_task_decision_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
