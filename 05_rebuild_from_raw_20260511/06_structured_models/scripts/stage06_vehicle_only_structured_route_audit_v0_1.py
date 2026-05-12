from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path("F:/data_set_process/data_process/05_rebuild_from_raw_20260511")
OUT_ROOT = PROJECT_ROOT / "06_structured_models" / "stage06_vehicle_only_structured_route_audit_v0_1"
TABLE_DIR = OUT_ROOT / "tables"
FIG_DIR = OUT_ROOT / "figures"
LOG_DIR = OUT_ROOT / "logs"
REPORT_DIR = PROJECT_ROOT / "09_reports"

TRACK_ID = "B_response3s_strict_core"
BASELINE_MODEL = "rbf_kernel_ridge_context_no_subject"

METRIC_SOURCES = [
    (
        "structured_vehicle_transformer",
        PROJECT_ROOT
        / "03_baselines"
        / "stage03_vehicle_instability_structured_vehicle_transformer_v0_1"
        / "tables"
        / "structured_vehicle_transformer_metrics.csv",
        10,
    ),
    (
        "rbf_keypoint_selector",
        PROJECT_ROOT
        / "03_baselines"
        / "stage03_vehicle_instability_rbf_keypoint_selector_v0_1"
        / "tables"
        / "rbf_keypoint_selector_metrics.csv",
        20,
    ),
    (
        "topk_vehicle_transformer",
        PROJECT_ROOT
        / "03_baselines"
        / "stage03_vehicle_instability_topk_vehicle_transformer_v0_1"
        / "tables"
        / "topk_vehicle_transformer_metrics.csv",
        30,
    ),
    (
        "topk_reliability_selector",
        PROJECT_ROOT
        / "03_baselines"
        / "stage03_vehicle_instability_topk_reliability_selector_v0_1"
        / "tables"
        / "topk_reliability_selector_metrics.csv",
        40,
    ),
    (
        "rbf_keypoint_multihypothesis_review",
        PROJECT_ROOT
        / "03_baselines"
        / "stage03_vehicle_instability_rbf_keypoint_multihypothesis_review_v0_1"
        / "tables"
        / "multihypothesis_metrics.csv",
        50,
    ),
]

LOWER_IS_BETTER = [
    "rmse_steer",
    "wrong_side_rate",
    "peak_amp_mae",
    "severe_amp_under_rate",
    "peak_time_mae_s",
    "onset_delay_mae_s",
    "tail_abs_error_mean",
    "tail_drift_risk_rate",
    "zero_crossing_mismatch_rate",
    "difficult_top20_rmse",
]
HIGHER_IS_BETTER = [
    "peak_direction_accuracy",
    "large_response_recall",
    "reversal_count_exact_match_rate",
]
KEY_METRICS = [
    "rmse_steer",
    "wrong_side_rate",
    "large_response_recall",
    "peak_amp_mae",
    "severe_amp_under_rate",
    "peak_time_mae_s",
    "tail_abs_error_mean",
    "tail_drift_risk_rate",
    "reversal_count_exact_match_rate",
    "difficult_top20_rmse",
]


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIG_DIR, LOG_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def read_metric_sources() -> pd.DataFrame:
    frames = []
    missing = []
    for source_name, path, priority in METRIC_SOURCES:
        if not path.exists():
            missing.append(str(path))
            continue
        df = pd.read_csv(path)
        if "split" not in df.columns or "model_name" not in df.columns:
            missing.append(f"{path} (missing split/model_name)")
            continue
        df = df[(df["split"] == "test") & (df.get("track_id", TRACK_ID) == TRACK_ID)].copy()
        if df.empty:
            continue
        df["source_table"] = source_name
        df["source_priority"] = priority
        frames.append(df)
    if not frames:
        raise RuntimeError("No usable metric source was found.")
    out = pd.concat(frames, ignore_index=True, sort=False)
    out = out.sort_values(["model_name", "source_priority"])
    out = out.drop_duplicates(subset=["model_name"], keep="first").reset_index(drop=True)
    if missing:
        print("Missing or skipped sources:")
        for item in missing:
            print(f"  - {item}")
    return out


def family_for_model(name: str) -> str:
    name_l = name.lower()
    if name_l == BASELINE_MODEL:
        return "primary_rbf_reference"
    if "oracle" in name_l or "best_of" in name_l:
        return "upper_bound_oracle"
    if "topk" in name_l:
        return "multi_hypothesis_or_selector"
    if "keypoint" in name_l:
        return "keypoint_residual_or_selector"
    if "structured" in name_l:
        return "response_decomposition_transformer"
    if "transformer" in name_l:
        return "direct_transformer"
    if "knn" in name_l:
        return "template_reference"
    if "ridge" in name_l:
        return "ridge_reference"
    if "mean" in name_l or "zero" in name_l or "trend" in name_l:
        return "nonlearning_reference"
    return "other_vehicle_reference"


def add_deltas(df: pd.DataFrame) -> pd.DataFrame:
    base_rows = df[df["model_name"] == BASELINE_MODEL]
    if base_rows.empty:
        raise RuntimeError(f"Missing baseline model: {BASELINE_MODEL}")
    base = base_rows.iloc[0]
    out = df.copy()
    for metric in KEY_METRICS:
        if metric in out.columns:
            out[f"delta_vs_rbf__{metric}"] = pd.to_numeric(out[metric], errors="coerce") - float(base[metric])
    gains = []
    regressions = []
    for _, row in out.iterrows():
        gain = 0
        regression = 0
        for metric in LOWER_IS_BETTER:
            if metric in row and f"delta_vs_rbf__{metric}" in out.columns:
                delta = row[f"delta_vs_rbf__{metric}"]
                if pd.isna(delta) or abs(delta) < 1e-12:
                    continue
                gain += int(delta < 0)
                regression += int(delta > 0)
        for metric in HIGHER_IS_BETTER:
            if metric in row and f"delta_vs_rbf__{metric}" in out.columns:
                delta = row[f"delta_vs_rbf__{metric}"]
                if pd.isna(delta) or abs(delta) < 1e-12:
                    continue
                gain += int(delta > 0)
                regression += int(delta < 0)
        gains.append(gain)
        regressions.append(regression)
    out["physical_metric_gain_count_vs_rbf"] = gains
    out["physical_metric_regression_count_vs_rbf"] = regressions
    out["candidate_family"] = out["model_name"].map(family_for_model)
    return out


def status_for_row(row: pd.Series) -> str:
    name = row["model_name"]
    family = row["candidate_family"]
    rmse_delta = row.get("delta_vs_rbf__rmse_steer", np.nan)
    gains = int(row.get("physical_metric_gain_count_vs_rbf", 0))
    regressions = int(row.get("physical_metric_regression_count_vs_rbf", 0))
    if name == BASELINE_MODEL:
        return "keep_limited_primary_reference"
    if family == "upper_bound_oracle":
        return "research_signal_not_deployable"
    if pd.notna(rmse_delta) and rmse_delta <= 0 and gains >= 2:
        return "candidate_continue"
    if pd.notna(rmse_delta) and rmse_delta <= 0.01 and gains >= 4 and regressions <= 5:
        return "weak_candidate_continue"
    if family == "keypoint_residual_or_selector" and pd.notna(rmse_delta) and rmse_delta <= 0.02 and gains >= 2:
        return "weak_no_go_current_form"
    if pd.notna(rmse_delta) and rmse_delta > 0.02:
        return "no_go_current_form"
    if gains >= 5 and regressions <= 5:
        return "diagnostic_candidate"
    return "reference_or_no_go"


def build_scorecard(df: pd.DataFrame) -> pd.DataFrame:
    out = add_deltas(df)
    out["route_status"] = out.apply(status_for_row, axis=1)
    columns = [
        "track_id",
        "model_name",
        "candidate_family",
        "route_status",
        "source_table",
        "n_samples",
        "rmse_steer",
        "delta_vs_rbf__rmse_steer",
        "wrong_side_rate",
        "delta_vs_rbf__wrong_side_rate",
        "large_response_recall",
        "delta_vs_rbf__large_response_recall",
        "peak_amp_mae",
        "delta_vs_rbf__peak_amp_mae",
        "severe_amp_under_rate",
        "delta_vs_rbf__severe_amp_under_rate",
        "peak_time_mae_s",
        "delta_vs_rbf__peak_time_mae_s",
        "tail_abs_error_mean",
        "delta_vs_rbf__tail_abs_error_mean",
        "tail_drift_risk_rate",
        "delta_vs_rbf__tail_drift_risk_rate",
        "reversal_count_exact_match_rate",
        "delta_vs_rbf__reversal_count_exact_match_rate",
        "difficult_top20_rmse",
        "delta_vs_rbf__difficult_top20_rmse",
        "physical_metric_gain_count_vs_rbf",
        "physical_metric_regression_count_vs_rbf",
    ]
    columns = [c for c in columns if c in out.columns]
    out = out[columns].sort_values(
        ["route_status", "rmse_steer", "physical_metric_gain_count_vs_rbf"],
        ascending=[True, True, False],
    )
    return out


def build_gate_table(scorecard: pd.DataFrame) -> pd.DataFrame:
    def model_row(name: str) -> pd.Series | None:
        rows = scorecard[scorecard["model_name"] == name]
        if rows.empty:
            return None
        return rows.iloc[0]

    base = model_row(BASELINE_MODEL)
    direct = model_row("vehicle_transformer_context_no_subject")
    structured = model_row("structured_vehicle_transformer_aux_no_subject")
    keypoint = model_row("keypoint_residual_vehicle_transformer_no_subject")
    selector = model_row("selector_logreg_rbf_keypoint_no_subject")
    topk_oracle = scorecard[scorecard["model_name"].str.contains("oracle|best_of", case=False, regex=True)]
    if not topk_oracle.empty:
        topk_oracle_best = topk_oracle.sort_values("rmse_steer").iloc[0]
    else:
        topk_oracle_best = None

    rows = []
    rows.append(
        {
            "gate": "rbf_primary_reference",
            "status": "pass_limited",
            "evidence": f"RBF test RMSE={base['rmse_steer']:.6f}, wrong_side={base['wrong_side_rate']:.3f}, large_recall={base['large_response_recall']:.3f}.",
            "decision": "保留为当前车辆-only主参照，但不是问题已解决。",
        }
    )
    if direct is not None:
        rows.append(
            {
                "gate": "direct_transformer_upgrade",
                "status": "fail",
                "evidence": f"direct Transformer RMSE delta={direct['delta_vs_rbf__rmse_steer']:+.6f}，大幅响应召回 delta={direct['delta_vs_rbf__large_response_recall']:+.3f}。",
                "decision": "不升级为主线。",
            }
        )
    if structured is not None:
        rows.append(
            {
                "gate": "response_decomposition_transformer_upgrade",
                "status": "fail",
                "evidence": f"结构化 Transformer RMSE delta={structured['delta_vs_rbf__rmse_steer']:+.6f}，尾段漂移 delta={structured['delta_vs_rbf__tail_drift_risk_rate']:+.3f}，困难样本 RMSE delta={structured['delta_vs_rbf__difficult_top20_rmse']:+.6f}。",
                "decision": "当前版本 no-go，只保留为失败样本和辅助头诊断。",
            }
        )
    if keypoint is not None:
        rows.append(
            {
                "gate": "keypoint_residual_upgrade",
                "status": "weak_no_go",
                "evidence": f"keypoint residual RMSE delta={keypoint['delta_vs_rbf__rmse_steer']:+.6f}，wrong-side delta={keypoint['delta_vs_rbf__wrong_side_rate']:+.3f}，large recall delta={keypoint['delta_vs_rbf__large_response_recall']:+.3f}。",
                "decision": "有方向/大幅响应信号，但整体和若干物理指标不足，不能单独升级。",
            }
        )
    if selector is not None:
        rows.append(
            {
                "gate": "rbf_keypoint_selector",
                "status": "weak_candidate_continue",
                "evidence": f"selector RMSE delta={selector['delta_vs_rbf__rmse_steer']:+.6f}，物理指标改善数={int(selector['physical_metric_gain_count_vs_rbf'])}，退化数={int(selector['physical_metric_regression_count_vs_rbf'])}。",
                "decision": "作为下一版选择器/可靠性候选继续，但不能作为已完成强基线。",
            }
        )
    if topk_oracle_best is not None:
        rows.append(
            {
                "gate": "multi_hypothesis_oracle_bound",
                "status": "research_signal_not_deployable",
                "evidence": f"最优 oracle/best-of-K RMSE={topk_oracle_best['rmse_steer']:.6f}，RBF RMSE={base['rmse_steer']:.6f}。",
                "decision": "说明多候选空间有潜力；必须做可部署选择策略，不能用 oracle 当结论。",
            }
        )
    rows.extend(
        [
            {
                "gate": "stage05_physio_eeg_allowed",
                "status": "blocked",
                "evidence": "车辆-only结构化主参照尚未形成稳定可解释升级，风格路线也刚被 no-go。",
                "decision": "继续阻塞生理/EEG有效性结论。",
            },
            {
                "gate": "stage06_next_route",
                "status": "go_stage06b",
                "evidence": "RBF仍为主参照；keypoint selector和oracle多候选给出有限但真实的研究信号。",
                "decision": "下一步优先做RBF+关键点/多假设的可部署选择器、可靠性门控和坏样本复盘。",
            },
        ]
    )
    return pd.DataFrame(rows)


def build_next_actions() -> pd.DataFrame:
    rows = [
        {
            "priority": 1,
            "action": "固定B轨道RBF为limited primary reference",
            "why": "RBF在当前test RMSE和尾段/困难样本上仍最稳，但错侧和反向修正没解决。",
            "server_needed": "no",
        },
        {
            "priority": 2,
            "action": "复盘selector_logreg_rbf_keypoint样本级选择错误",
            "why": "该路线RMSE几乎持平，同时改善方向、大幅响应和困难样本，最适合继续做可部署选择器。",
            "server_needed": "no",
        },
        {
            "priority": 3,
            "action": "把best-of-K/oracle收益转化为非oracle选择策略",
            "why": "oracle RMSE明显低于RBF，说明候选空间有潜力，但当前top1/selector不能稳定选中。",
            "server_needed": "local_first",
        },
        {
            "priority": 4,
            "action": "做坏样本分桶：错侧、幅值不足、峰时、尾段、反向修正、多段修正",
            "why": "下一版结构模型必须证明改善物理错误，而不是只改善RMSE。",
            "server_needed": "no",
        },
        {
            "priority": 5,
            "action": "暂不进入生理/EEG有效性验证",
            "why": "车辆主参照仍未结构化冻结，直接加新模态会混淆归因。",
            "server_needed": "no",
        },
    ]
    return pd.DataFrame(rows)


def plot_summary(scorecard: pd.DataFrame) -> tuple[Path, Path]:
    selected_names = [
        BASELINE_MODEL,
        "vehicle_transformer_context_no_subject",
        "structured_vehicle_transformer_aux_no_subject",
        "keypoint_residual_vehicle_transformer_no_subject",
        "selector_logreg_rbf_keypoint_no_subject",
        "topk_top1_rbf_fallback_logreg_no_subject",
        "topk_vehicle_transformer_best_of_3_oracle",
        "oracle_best_of_rbf_plus_topk_upper_bound",
    ]
    plot_df = scorecard[scorecard["model_name"].isin(selected_names)].copy()
    plot_df = plot_df.sort_values("rmse_steer")
    labels = [short_name(x) for x in plot_df["model_name"]]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ["#4b5563" if "oracle" not in n.lower() and "best_of" not in n.lower() else "#9ca3af" for n in plot_df["model_name"]]
    ax.barh(labels, plot_df["rmse_steer"], color=colors)
    ax.axvline(float(scorecard.loc[scorecard["model_name"] == BASELINE_MODEL, "rmse_steer"].iloc[0]), color="#2563eb", linewidth=1.5, linestyle="--", label="RBF")
    ax.set_xlabel("test RMSE lower is better")
    ax.set_title("Stage 6 vehicle-only structured route: RMSE")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    rmse_path = FIG_DIR / "vehicle_structured_route_rmse_summary.png"
    fig.savefig(rmse_path, dpi=180)
    plt.close(fig)

    delta_metrics = [
        ("delta_vs_rbf__rmse_steer", "RMSE", -1),
        ("delta_vs_rbf__wrong_side_rate", "Wrong-side", -1),
        ("delta_vs_rbf__large_response_recall", "Large recall", 1),
        ("delta_vs_rbf__difficult_top20_rmse", "Difficult RMSE", -1),
    ]
    delta_df = plot_df[~plot_df["model_name"].str.contains("oracle|best_of", case=False, regex=True)].copy()
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.8), sharey=True)
    y = np.arange(len(delta_df))
    for ax, (metric, title, direction) in zip(axes, delta_metrics):
        values = delta_df[metric].astype(float).to_numpy()
        colors = ["#16a34a" if v * direction > 0 else "#dc2626" if v * direction < 0 else "#6b7280" for v in values]
        ax.barh(y, values, color=colors)
        ax.axvline(0, color="#111827", linewidth=1)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([short_name(x) for x in delta_df["model_name"]])
    fig.suptitle("Delta vs RBF: green is better")
    fig.tight_layout()
    delta_path = FIG_DIR / "vehicle_structured_route_delta_vs_rbf.png"
    fig.savefig(delta_path, dpi=180)
    plt.close(fig)
    return rmse_path, delta_path


def short_name(name: str) -> str:
    mapping = {
        BASELINE_MODEL: "RBF",
        "vehicle_transformer_context_no_subject": "Direct TX",
        "structured_vehicle_transformer_aux_no_subject": "Structured TX",
        "keypoint_residual_vehicle_transformer_no_subject": "Keypoint",
        "selector_logreg_rbf_keypoint_no_subject": "RBF+Keypoint selector",
        "topk_top1_rbf_fallback_logreg_no_subject": "TopK fallback",
        "topk_vehicle_transformer_best_of_3_oracle": "TopK oracle",
        "oracle_best_of_rbf_plus_topk_upper_bound": "RBF+TopK oracle",
    }
    return mapping.get(name, name.replace("_no_subject", "").replace("_context", "").replace("_", " ")[:28])


def md_table(df: pd.DataFrame, cols: list[str], max_rows: int = 12) -> str:
    if df.empty:
        return "_无数据_"
    view = df[cols].head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_numeric_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.6f}")
    widths = {col: max(len(str(col)), *(len(str(v)) for v in view[col])) for col in view.columns}
    header = "| " + " | ".join(str(col).ljust(widths[col]) for col in view.columns) + " |"
    sep = "| " + " | ".join("-" * widths[col] for col in view.columns) + " |"
    rows = [
        "| " + " | ".join(str(row[col]).ljust(widths[col]) for col in view.columns) + " |"
        for _, row in view.iterrows()
    ]
    return "\n".join([header, sep, *rows])


def write_reports(scorecard: pd.DataFrame, gate: pd.DataFrame, next_actions: pd.DataFrame, rmse_fig: Path, delta_fig: Path) -> tuple[Path, Path, Path]:
    base = scorecard[scorecard["model_name"] == BASELINE_MODEL].iloc[0]
    structured = scorecard[scorecard["model_name"] == "structured_vehicle_transformer_aux_no_subject"].iloc[0]
    selector = scorecard[scorecard["model_name"] == "selector_logreg_rbf_keypoint_no_subject"].iloc[0]
    oracle = scorecard[scorecard["candidate_family"] == "upper_bound_oracle"].sort_values("rmse_steer").iloc[0]
    rmse_fig_s = str(rmse_fig).replace("\\", "/")
    delta_fig_s = str(delta_fig).replace("\\", "/")

    user_summary = f"""# 阶段 6 用户查看版：车辆-only结构化路线审计 v0.1

## 这个阶段为什么做

阶段 4 已经说明当前连续风格路线不能升级主线，生理/EEG 也还不能进入有效性结论。因此现在必须先把车辆-only结构化路线重新收口：哪些车辆模型能作为主参照，哪些只是失败/诊断候选，下一步应该继续哪条结构化路线。

## 这个阶段检查了什么

- 只检查 B 轨道 270 个严格核心 3 秒响应样本上的已有车辆-only结果。
- 汇总 RBF、direct Transformer、响应分解 Transformer、关键点+残差、多假设/top-K、选择器/可靠性候选。
- 所有比较都以当前 RBF KRR 为参照。
- 没有使用生理、脑电、连续风格、驾驶员 ID、服务器或服务器密码文件。

## 目前发现了什么

- RBF 仍是当前最稳的车辆-only主参照：test RMSE={base['rmse_steer']:.6f}，错侧率={base['wrong_side_rate']:.3f}，大幅响应召回={base['large_response_recall']:.3f}。
- 响应分解 Transformer v0.1 不能升级：test RMSE={structured['rmse_steer']:.6f}，比 RBF 差 {structured['delta_vs_rbf__rmse_steer']:+.6f}，大幅响应召回和尾段也更差。
- `selector_logreg_rbf_keypoint_no_subject` 是弱候选：RMSE 基本持平，方向/大幅响应/困难样本有一些改善，但尾段和零线穿越等指标仍退化，不能直接定为主线。
- oracle/best-of-K 上限很强：最佳 oracle RMSE={oracle['rmse_steer']:.6f}，说明多候选空间有潜力，但 oracle 不是可部署方法，不能当最终结论。

## 哪些结果可信

可信的是已有车辆-only候选在同一 B 轨道 test 集上的相对表现，以及“当前结构化 Transformer v0.1 不该升级主线”这个判断。

## 哪些还不能下结论

不能说车辆-only问题已经解决；也不能说生理/EEG有效。当前只能说，多假设/关键点选择器方向有研究信号，但还没有形成稳定可部署选择策略。

## 下一阶段是否可以继续

可以继续，但下一步应是 Stage 6b：RBF + 关键点/多假设候选的可部署选择器、可靠性门控和坏样本复盘，而不是直接加入生理/EEG。

## 推荐优先查看

1. `{gate_path()}`
2. `{scorecard_path()}`
3. `{rmse_fig_s}`
4. `{delta_fig_s}`
"""

    tech_report = f"""# 阶段 6：车辆-only结构化路线审计 v0.1

生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}

## 输入

- structured Transformer 指标表
- RBF/keypoint selector 指标表
- top-K Transformer 指标表
- top-K reliability selector 指标表
- RBF/keypoint multihypothesis review 指标表

本轮只读已有指标表，不重新训练模型，不读取原始数据，不使用服务器。

## Gate 结论

{md_table(gate, ['gate', 'status', 'decision'], max_rows=20)}

## 关键候选 scorecard

{md_table(scorecard.sort_values('rmse_steer'), ['model_name', 'route_status', 'rmse_steer', 'delta_vs_rbf__rmse_steer', 'wrong_side_rate', 'large_response_recall', 'difficult_top20_rmse'], max_rows=20)}

## 解释边界

- RBF 继续作为 limited primary reference，不代表车辆-only任务已解决。
- 响应分解 Transformer v0.1 是 no-go 当前形式，不代表结构化建模方向被否定。
- keypoint selector 是弱候选，需要样本级选择错误复盘和可靠性门控。
- oracle/best-of-K 是研究上限，不可作为可部署结果或论文主结论。
- 生理/EEG 仍 blocked。

## 图

- RMSE 汇总图：`{rmse_fig_s}`
- 相对 RBF delta 图：`{delta_fig_s}`
"""

    user_path = REPORT_DIR / "stage06_vehicle_only_structured_route_audit_user_summary_cn.md"
    stage_path = REPORT_DIR / "stage06_user_summary_cn.md"
    tech_path = REPORT_DIR / "stage06_vehicle_only_structured_route_audit_v0_1_cn.md"
    user_path.write_text(user_summary, encoding="utf-8")
    stage_path.write_text(user_summary, encoding="utf-8")
    tech_path.write_text(tech_report, encoding="utf-8")
    return user_path, stage_path, tech_path


def scorecard_path() -> str:
    return str(TABLE_DIR / "vehicle_structured_candidate_scorecard.csv").replace("\\", "/")


def gate_path() -> str:
    return str(TABLE_DIR / "vehicle_structured_route_gate_table.csv").replace("\\", "/")


def main() -> None:
    ensure_dirs()
    metrics = read_metric_sources()
    scorecard = build_scorecard(metrics)
    gate = build_gate_table(scorecard)
    next_actions = build_next_actions()

    scorecard_out = TABLE_DIR / "vehicle_structured_candidate_scorecard.csv"
    delta_out = TABLE_DIR / "vehicle_structured_metric_delta_vs_rbf.csv"
    gate_out = TABLE_DIR / "vehicle_structured_route_gate_table.csv"
    next_out = TABLE_DIR / "vehicle_structured_next_actions.csv"

    scorecard.to_csv(scorecard_out, index=False, encoding="utf-8-sig")
    delta_cols = ["model_name", "candidate_family"] + [c for c in scorecard.columns if c.startswith("delta_vs_rbf__")]
    scorecard[delta_cols].to_csv(delta_out, index=False, encoding="utf-8-sig")
    gate.to_csv(gate_out, index=False, encoding="utf-8-sig")
    next_actions.to_csv(next_out, index=False, encoding="utf-8-sig")

    rmse_fig, delta_fig = plot_summary(scorecard)
    user_path, stage_path, tech_path = write_reports(scorecard, gate, next_actions, rmse_fig, delta_fig)

    summary = {
        "output_version": "stage06_vehicle_only_structured_route_audit_v0_1",
        "run_time_local": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "track_id": TRACK_ID,
        "n_candidates": int(scorecard.shape[0]),
        "primary_reference": BASELINE_MODEL,
        "rbf_rmse": float(scorecard.loc[scorecard["model_name"] == BASELINE_MODEL, "rmse_steer"].iloc[0]),
        "structured_transformer_status": scorecard.loc[
            scorecard["model_name"] == "structured_vehicle_transformer_aux_no_subject", "route_status"
        ].iloc[0],
        "selector_status": scorecard.loc[
            scorecard["model_name"] == "selector_logreg_rbf_keypoint_no_subject", "route_status"
        ].iloc[0],
        "stage05_physio_eeg_allowed": "blocked",
        "next_route": "stage06b_keypoint_multihypothesis_selector_reliability",
        "scorecard_path": str(scorecard_out).replace("\\", "/"),
        "gate_path": str(gate_out).replace("\\", "/"),
        "next_actions_path": str(next_out).replace("\\", "/"),
        "user_summary_path": str(user_path).replace("\\", "/"),
        "stage_summary_path": str(stage_path).replace("\\", "/"),
        "technical_report_path": str(tech_path).replace("\\", "/"),
        "server_used": False,
        "server_credential_file_read": False,
        "uses_physio": False,
        "uses_eeg": False,
        "uses_continuous_style": False,
        "raw_files_modified": False,
    }
    summary_path = LOG_DIR / "vehicle_structured_route_audit_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
