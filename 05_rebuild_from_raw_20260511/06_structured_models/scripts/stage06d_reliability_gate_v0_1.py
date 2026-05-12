from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("F:/data_set_process/data_process/05_rebuild_from_raw_20260511")
SCRIPT_DIR = ROOT / "03_baselines" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_stage3_vehicle_baselines as eval_utils  # noqa: E402


SOURCE_ROWS = ROOT / "03_baselines" / "stage03_vehicle_instability_keypoint_residual_vehicle_transformer_v0_1" / "tables" / "keypoint_residual_vehicle_transformer_per_sample_metrics.csv"
DETAIL_PATH = ROOT / "06_structured_models" / "stage06c_selector_feature_revision_v0_1" / "tables" / "selector_revision_candidate_details.csv"
OUT_ROOT = ROOT / "06_structured_models" / "stage06d_reliability_gate_v0_1"
TABLE_DIR = OUT_ROOT / "tables"
FIG_DIR = OUT_ROOT / "figures"
LOG_DIR = OUT_ROOT / "logs"
REPORT_DIR = ROOT / "09_reports"

TRACK_ID = "B_response3s_strict_core"
RBF_MODEL = "rbf_kernel_ridge_context_no_subject"
KEYPOINT_MODEL = "keypoint_residual_vehicle_transformer_no_subject"
ORACLE_MODEL = "oracle_best_of_rbf_keypoint_upper_bound"


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIG_DIR, LOG_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = pd.read_csv(SOURCE_ROWS)
    rows = rows[(rows["track_id"] == TRACK_ID) & (rows["model_name"].isin([RBF_MODEL, KEYPOINT_MODEL]))].copy()
    detail = pd.read_csv(DETAIL_PATH)
    if rows.empty or detail.empty:
        raise RuntimeError("Missing source rows or stage06c candidate details.")
    return rows, detail


def choose_rows(rows: pd.DataFrame, sample_decision: pd.DataFrame, model_name: str) -> pd.DataFrame:
    pair = rows.merge(sample_decision[["sample_id", "selected_model"]], on="sample_id", how="inner")
    selected = pair[pair["model_name"] == pair["selected_model"]].copy()
    selected["model_name"] = model_name
    return selected.drop(columns=["selected_model"])


def aggregate_for_policy(rows: pd.DataFrame, detail: pd.DataFrame, candidate_model: str, threshold: float, policy_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = detail[detail["candidate_model"] == candidate_model].copy()
    sub["selected_model"] = np.where(sub["selector_prob_keypoint"].astype(float) >= threshold, KEYPOINT_MODEL, RBF_MODEL)
    selected_rows = choose_rows(rows, sub[["sample_id", "selected_model"]], policy_name)
    metrics = eval_utils.aggregate_metrics(selected_rows)
    metrics["track_id"] = TRACK_ID
    metrics["candidate_model"] = candidate_model
    metrics["policy_name"] = policy_name
    metrics["threshold"] = threshold
    metrics["keypoint_selected_rate"] = [
        float(sub[sub["split"] == split]["selected_model"].eq(KEYPOINT_MODEL).mean()) for split in metrics["split"]
    ]
    return metrics, sub


def reference_metrics(rows: pd.DataFrame) -> pd.DataFrame:
    refs = rows.copy()
    oracle = refs.sort_values("sample_rmse").groupby("sample_id", as_index=False).head(1).copy()
    oracle["model_name"] = ORACLE_MODEL
    metrics = eval_utils.aggregate_metrics(pd.concat([refs, oracle], ignore_index=True, sort=False))
    metrics["track_id"] = TRACK_ID
    metrics["candidate_model"] = metrics["model_name"]
    metrics["policy_name"] = metrics["model_name"]
    metrics["threshold"] = np.nan
    metrics["keypoint_selected_rate"] = np.nan
    return metrics


def scan_policies(rows: pd.DataFrame, detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    thresholds = np.round(np.linspace(0.05, 0.95, 19), 2)
    metric_frames = []
    decision_frames = []
    for candidate_model in sorted(detail["candidate_model"].unique()):
        for threshold in thresholds:
            policy_name = f"{candidate_model}__thr_{threshold:.2f}"
            metrics, decisions = aggregate_for_policy(rows, detail, candidate_model, float(threshold), policy_name)
            metric_frames.append(metrics)
            decisions = decisions.copy()
            decisions["policy_name"] = policy_name
            decisions["threshold"] = float(threshold)
            decision_frames.append(decisions)
    all_metrics = pd.concat(metric_frames, ignore_index=True, sort=False)
    decisions = pd.concat(decision_frames, ignore_index=True, sort=False)
    refs = reference_metrics(rows)
    all_with_refs = pd.concat([refs, all_metrics], ignore_index=True, sort=False)
    return all_with_refs, all_metrics, decisions


def select_reliability_policies(all_metrics: pd.DataFrame) -> pd.DataFrame:
    val = all_metrics[all_metrics["split"] == "val"].copy()
    rbf_val = val[val["model_name"] == RBF_MODEL].iloc[0]
    candidates = val[~val["model_name"].isin([RBF_MODEL, KEYPOINT_MODEL, ORACLE_MODEL])].copy()
    candidates["rmse_margin_vs_rbf"] = candidates["rmse_steer"].astype(float) - float(rbf_val["rmse_steer"])
    candidates["wrong_margin_vs_rbf"] = candidates["wrong_side_rate"].astype(float) - float(rbf_val["wrong_side_rate"])
    candidates["large_margin_vs_rbf"] = candidates["large_response_recall"].astype(float) - float(rbf_val["large_response_recall"])

    policies = []
    best_rmse = candidates.sort_values(["rmse_steer", "keypoint_selected_rate"], ascending=[True, True]).iloc[0]
    policies.append({"policy_label": "val_best_rmse", "policy_name": best_rmse["policy_name"], "selection_rule": "min val RMSE"})

    noninferior = candidates[candidates["rmse_margin_vs_rbf"] <= 0.0].copy()
    if not noninferior.empty:
        conservative = noninferior.sort_values(["keypoint_selected_rate", "rmse_steer"], ascending=[True, True]).iloc[0]
        policies.append(
            {
                "policy_label": "val_rmse_noninferior_conservative",
                "policy_name": conservative["policy_name"],
                "selection_rule": "val RMSE <= RBF, then lowest keypoint selected rate",
            }
        )

    phys = candidates[
        (candidates["rmse_margin_vs_rbf"] <= 0.0)
        & (candidates["wrong_margin_vs_rbf"] <= 0.0)
        & (candidates["large_margin_vs_rbf"] >= 0.0)
    ].copy()
    if not phys.empty:
        phys_pick = phys.sort_values(["keypoint_selected_rate", "rmse_steer"], ascending=[True, True]).iloc[0]
        policies.append(
            {
                "policy_label": "val_rmse_physical_noninferior_conservative",
                "policy_name": phys_pick["policy_name"],
                "selection_rule": "val RMSE/wrong-side/large-recall noninferior to RBF, then lowest keypoint selected rate",
            }
        )

    out = pd.DataFrame(policies).drop_duplicates("policy_label")
    return out


def materialize_policy_metrics(all_metrics: pd.DataFrame, policy_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, policy in policy_table.iterrows():
        subset = all_metrics[all_metrics["policy_name"] == policy["policy_name"]].copy()
        subset["policy_label"] = policy["policy_label"]
        subset["selection_rule"] = policy["selection_rule"]
        rows.append(subset)
    refs = all_metrics[all_metrics["model_name"].isin([RBF_MODEL, KEYPOINT_MODEL, ORACLE_MODEL])].copy()
    refs["policy_label"] = refs["model_name"]
    refs["selection_rule"] = "reference"
    rows.append(refs)
    return pd.concat(rows, ignore_index=True, sort=False)


def confusion_for_policy(decisions: pd.DataFrame, policy_name: str) -> pd.DataFrame:
    test = decisions[(decisions["policy_name"] == policy_name) & (decisions["split"] == "test")].copy()
    test["selected_keypoint"] = test["selected_model"].eq(KEYPOINT_MODEL).astype(int)
    test["oracle_keypoint"] = test["keypoint_better_rmse"].astype(int)
    test["selection_outcome"] = np.select(
        [
            (test["selected_keypoint"] == 1) & (test["oracle_keypoint"] == 1),
            (test["selected_keypoint"] == 1) & (test["oracle_keypoint"] == 0),
            (test["selected_keypoint"] == 0) & (test["oracle_keypoint"] == 1),
            (test["selected_keypoint"] == 0) & (test["oracle_keypoint"] == 0),
        ],
        ["TP_select_keypoint_correct", "FP_select_keypoint_hurts", "FN_missed_keypoint_gain", "TN_keep_rbf_correct"],
        default="unknown",
    )
    out = (
        test.groupby("selection_outcome", dropna=False)
        .agg(
            n_samples=("sample_id", "count"),
            mean_selector_prob_keypoint=("selector_prob_keypoint", "mean"),
            mean_keypoint_delta_vs_rbf=("rmse_delta_keypoint_minus_rbf", "mean"),
        )
        .reset_index()
    )
    return out


def build_gate(policy_metrics: pd.DataFrame, selected_policies: pd.DataFrame) -> pd.DataFrame:
    test = policy_metrics[policy_metrics["split"] == "test"].copy()
    rbf = test[test["model_name"] == RBF_MODEL].iloc[0]
    rows = []
    for _, policy in selected_policies.iterrows():
        row = test[test["policy_label"] == policy["policy_label"]].iloc[0]
        rmse_delta = float(row["rmse_steer"] - rbf["rmse_steer"])
        wrong_delta = float(row["wrong_side_rate"] - rbf["wrong_side_rate"])
        large_delta = float(row["large_response_recall"] - rbf["large_response_recall"])
        difficult_delta = float(row["difficult_top20_rmse"] - rbf["difficult_top20_rmse"])
        status = "continue_candidate" if (rmse_delta <= 0.0 and wrong_delta <= 0.0 and large_delta >= 0.0) else "no_upgrade"
        rows.append(
            {
                "gate": policy["policy_label"],
                "status": status,
                "evidence": f"test RMSE delta={rmse_delta:+.6f}, wrong-side delta={wrong_delta:+.3f}, large recall delta={large_delta:+.3f}, difficult RMSE delta={difficult_delta:+.6f}.",
                "decision": "若RMSE仍退化则不能升级；若仅物理指标改善则保留为诊断候选。",
            }
        )
    rows.append(
        {
            "gate": "stage05_physio_eeg_allowed",
            "status": "blocked",
            "evidence": "reliability gate 尚未形成稳定车辆-only主线升级。",
            "decision": "继续阻塞生理/EEG有效性结论。",
        }
    )
    return pd.DataFrame(rows)


def plot_policy_metrics(policy_metrics: pd.DataFrame) -> tuple[Path, Path]:
    test = policy_metrics[policy_metrics["split"] == "test"].copy()
    show = test[
        test["policy_label"].isin(
            [
                RBF_MODEL,
                KEYPOINT_MODEL,
                ORACLE_MODEL,
                "val_best_rmse",
                "val_rmse_noninferior_conservative",
                "val_rmse_physical_noninferior_conservative",
            ]
        )
    ].copy()
    show = show.drop_duplicates("policy_label")
    show = show.sort_values("rmse_steer")
    labels = [short_label(x) for x in show["policy_label"]]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.barh(labels, show["rmse_steer"], color="#4b5563")
    rbf = float(show[show["policy_label"] == RBF_MODEL]["rmse_steer"].iloc[0])
    ax.axvline(rbf, color="#2563eb", linestyle="--", linewidth=1.5)
    ax.set_xlabel("test RMSE lower is better")
    ax.set_title("Reliability gate policies: test RMSE")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    rmse_path = FIG_DIR / "reliability_gate_test_rmse.png"
    fig.savefig(rmse_path, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    for ax, metric, title in [
        (axes[0], "wrong_side_rate", "wrong-side lower"),
        (axes[1], "large_response_recall", "large recall higher"),
        (axes[2], "difficult_top20_rmse", "difficult RMSE lower"),
    ]:
        ax.barh(labels, show[metric].astype(float), color="#6b7280")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    phys_path = FIG_DIR / "reliability_gate_physical_metrics.png"
    fig.savefig(phys_path, dpi=180)
    plt.close(fig)
    return rmse_path, phys_path


def short_label(label: str) -> str:
    mapping = {
        RBF_MODEL: "RBF",
        KEYPOINT_MODEL: "Keypoint",
        ORACLE_MODEL: "Oracle",
        "val_best_rmse": "Val-best",
        "val_rmse_noninferior_conservative": "Conservative RMSE",
        "val_rmse_physical_noninferior_conservative": "Conservative physical",
    }
    return mapping.get(label, label)


def md_table(df: pd.DataFrame, cols: list[str], max_rows: int = 12) -> str:
    view = df[cols].head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_numeric_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.6f}")
    widths = {col: max(len(str(col)), *(len(str(v)) for v in view[col])) for col in view.columns}
    header = "| " + " | ".join(str(col).ljust(widths[col]) for col in view.columns) + " |"
    sep = "| " + " | ".join("-" * widths[col] for col in view.columns) + " |"
    rows = ["| " + " | ".join(str(row[col]).ljust(widths[col]) for col in view.columns) + " |" for _, row in view.iterrows()]
    return "\n".join([header, sep, *rows])


def write_reports(policy_metrics: pd.DataFrame, gate: pd.DataFrame, confusion: pd.DataFrame, rmse_fig: Path, phys_fig: Path) -> tuple[Path, Path]:
    test = policy_metrics[policy_metrics["split"] == "test"].copy()
    rbf = test[test["model_name"] == RBF_MODEL].iloc[0]
    best_policy = test[~test["policy_label"].isin([RBF_MODEL, KEYPOINT_MODEL, ORACLE_MODEL])].sort_values("rmse_steer").iloc[0]
    user = f"""# Stage 6d 用户查看版：RBF/KNN 可靠性门控 v0.1

## 为什么做

这里不是在把 Transformer 当主线继续训练。当前可用的车辆-only 主参照仍然是 RBF/KNN 类强基线；Stage 6c 的 RF selector 只是尝试在“保留 RBF/KNN 预测”和“切换到 keypoint 候选预测”之间做选择。Stage 6c 虽然改善了错侧率和大幅响应召回，但 RMSE 退化，所以这一阶段用更保守的 reliability gate 控制错选 keypoint 的风险。规则只用 val 选择，test 只做最终评估。

## 目前发现

- RBF/KNN 主参照 test RMSE={rbf['rmse_steer']:.6f}。
- 当前最好的 reliability policy：`{best_policy['policy_label']}`，test RMSE={best_policy['rmse_steer']:.6f}，相对 RBF/KNN delta={best_policy['rmse_steer'] - rbf['rmse_steer']:+.6f}。
- 该 policy wrong-side={best_policy['wrong_side_rate']:.3f}，large recall={best_policy['large_response_recall']:.3f}。
- gate 若显示 `no_upgrade`，说明 reliability gate 仍不能升级为主车辆路线，只能作为诊断候选。

## 当前判断

如果保守门控仍不能同时改善 RMSE 和物理指标，Stage 6 的 selector 路线需要暂时降级。下一步应考虑更直接的多假设候选生成/选择，或回到车辆-only 表示和样本规则复查；生理/EEG 继续阻塞。

## 推荐优先查看

1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06d_reliability_gate_v0_1/tables/reliability_gate_gate_table.csv`
2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06d_reliability_gate_v0_1/tables/reliability_gate_policy_metrics.csv`
3. `{str(rmse_fig).replace(chr(92), '/')}`
4. `{str(phys_fig).replace(chr(92), '/')}`
"""
    tech = f"""# Stage 6d：RBF/KNN reliability gate v0.1

生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}

说明：当前主参照是 RBF/KNN 类车辆-only 强基线，不是 Transformer。本文的 gate 只是在 RBF/KNN 与 keypoint 候选之间做保守切换，不使用生理、EEG、连续风格或被试 ID。

## Gate

{md_table(gate, ['gate', 'status', 'decision'], max_rows=20)}

## Test metrics

{md_table(test.sort_values('rmse_steer'), ['policy_label', 'model_name', 'rmse_steer', 'wrong_side_rate', 'large_response_recall', 'difficult_top20_rmse'], max_rows=20)}

## Best policy confusion

{md_table(confusion, ['selection_outcome', 'n_samples', 'mean_selector_prob_keypoint', 'mean_keypoint_delta_vs_rbf'], max_rows=10)}
"""
    user_path = REPORT_DIR / "stage06d_reliability_gate_user_summary_cn.md"
    tech_path = REPORT_DIR / "stage06d_reliability_gate_v0_1_cn.md"
    user_path.write_text(user, encoding="utf-8")
    tech_path.write_text(tech, encoding="utf-8")
    return user_path, tech_path


def main() -> None:
    ensure_dirs()
    rows, detail = load_inputs()
    all_with_refs, all_policy_metrics, decisions = scan_policies(rows, detail)
    selected_policies = select_reliability_policies(all_with_refs)
    policy_metrics = materialize_policy_metrics(all_with_refs, selected_policies)
    gate = build_gate(policy_metrics, selected_policies)
    best_policy_name = policy_metrics[
        (policy_metrics["split"] == "test")
        & (~policy_metrics["policy_label"].isin([RBF_MODEL, KEYPOINT_MODEL, ORACLE_MODEL]))
    ].sort_values("rmse_steer").iloc[0]["policy_name"]
    confusion = confusion_for_policy(decisions, best_policy_name)
    rmse_fig, phys_fig = plot_policy_metrics(policy_metrics)
    user_path, tech_path = write_reports(policy_metrics, gate, confusion, rmse_fig, phys_fig)

    all_policy_metrics.to_csv(TABLE_DIR / "reliability_gate_all_threshold_metrics.csv", index=False, encoding="utf-8-sig")
    selected_policies.to_csv(TABLE_DIR / "reliability_gate_selected_policies.csv", index=False, encoding="utf-8-sig")
    policy_metrics.to_csv(TABLE_DIR / "reliability_gate_policy_metrics.csv", index=False, encoding="utf-8-sig")
    gate.to_csv(TABLE_DIR / "reliability_gate_gate_table.csv", index=False, encoding="utf-8-sig")
    confusion.to_csv(TABLE_DIR / "reliability_gate_best_confusion.csv", index=False, encoding="utf-8-sig")

    test = policy_metrics[policy_metrics["split"] == "test"].copy()
    rbf = test[test["model_name"] == RBF_MODEL].iloc[0]
    best = test[~test["policy_label"].isin([RBF_MODEL, KEYPOINT_MODEL, ORACLE_MODEL])].sort_values("rmse_steer").iloc[0]
    summary = {
        "output_version": "stage06d_reliability_gate_v0_1",
        "run_time_local": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "best_policy_label": best["policy_label"],
        "best_policy_name": best["policy_name"],
        "rbf_test_rmse": float(rbf["rmse_steer"]),
        "best_policy_test_rmse": float(best["rmse_steer"]),
        "best_policy_delta_vs_rbf_rmse": float(best["rmse_steer"] - rbf["rmse_steer"]),
        "best_policy_wrong_side_rate": float(best["wrong_side_rate"]),
        "rbf_wrong_side_rate": float(rbf["wrong_side_rate"]),
        "best_policy_large_recall": float(best["large_response_recall"]),
        "rbf_large_recall": float(rbf["large_response_recall"]),
        "stage05_physio_eeg_allowed": "blocked",
        "server_used": False,
        "server_credential_file_read": False,
        "uses_physio": False,
        "uses_eeg": False,
        "uses_continuous_style": False,
        "uses_subject_id": False,
        "raw_files_modified": False,
        "gate_path": str(TABLE_DIR / "reliability_gate_gate_table.csv").replace("\\", "/"),
        "metrics_path": str(TABLE_DIR / "reliability_gate_policy_metrics.csv").replace("\\", "/"),
        "user_summary_path": str(user_path).replace("\\", "/"),
        "technical_report_path": str(tech_path).replace("\\", "/"),
    }
    (LOG_DIR / "reliability_gate_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
