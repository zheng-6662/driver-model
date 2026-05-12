from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path("F:/data_set_process/data_process/05_rebuild_from_raw_20260511")
IN_DIR = PROJECT_ROOT / "03_baselines" / "stage03_vehicle_instability_rbf_keypoint_selector_v0_1" / "tables"
OUT_ROOT = PROJECT_ROOT / "06_structured_models" / "stage06b_keypoint_selector_error_review_v0_1"
TABLE_DIR = OUT_ROOT / "tables"
FIG_DIR = OUT_ROOT / "figures"
LOG_DIR = OUT_ROOT / "logs"
REPORT_DIR = PROJECT_ROOT / "09_reports"

RBF_MODEL = "rbf_kernel_ridge_context_no_subject"
KEYPOINT_MODEL = "keypoint_residual_vehicle_transformer_no_subject"
SELECTOR_MODEL = "selector_logreg_rbf_keypoint_no_subject"


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIG_DIR, LOG_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_detail() -> pd.DataFrame:
    training = pd.read_csv(IN_DIR / "rbf_keypoint_selector_training_table.csv")
    decisions = pd.read_csv(IN_DIR / "rbf_keypoint_selector_decisions.csv")
    keep_decision_cols = [
        "sample_id",
        "selector_prob_keypoint",
        "selector_threshold",
        "selected_model",
    ]
    df = training.merge(decisions[keep_decision_cols], on="sample_id", how="left", validate="one_to_one")
    df["selected_keypoint"] = df["selected_model"].eq(KEYPOINT_MODEL).astype(int)
    df["oracle_keypoint"] = pd.to_numeric(df["keypoint_better_rmse"], errors="coerce").fillna(0).astype(int)
    df["selection_outcome"] = np.select(
        [
            (df["selected_keypoint"] == 1) & (df["oracle_keypoint"] == 1),
            (df["selected_keypoint"] == 1) & (df["oracle_keypoint"] == 0),
            (df["selected_keypoint"] == 0) & (df["oracle_keypoint"] == 1),
            (df["selected_keypoint"] == 0) & (df["oracle_keypoint"] == 0),
        ],
        ["TP_select_keypoint_correct", "FP_select_keypoint_hurts", "FN_missed_keypoint_gain", "TN_keep_rbf_correct"],
        default="unknown",
    )
    rbf_rmse = df[f"sample_rmse__{RBF_MODEL}"].astype(float)
    key_rmse = df[f"sample_rmse__{KEYPOINT_MODEL}"].astype(float)
    df["sample_rmse_rbf"] = rbf_rmse
    df["sample_rmse_keypoint"] = key_rmse
    df["sample_rmse_selected"] = np.where(df["selected_keypoint"] == 1, key_rmse, rbf_rmse)
    df["sample_rmse_oracle"] = np.minimum(rbf_rmse, key_rmse)
    df["selector_regret_vs_oracle"] = df["sample_rmse_selected"] - df["sample_rmse_oracle"]
    df["selector_delta_vs_rbf"] = df["sample_rmse_selected"] - rbf_rmse
    df["keypoint_delta_vs_rbf"] = key_rmse - rbf_rmse
    df["selector_helped_rbf"] = (df["selector_delta_vs_rbf"] < -1e-12).astype(int)
    df["selector_hurt_rbf"] = (df["selector_delta_vs_rbf"] > 1e-12).astype(int)
    for metric in [
        "wrong_side",
        "large_response_recalled",
        "severe_amp_under",
        "tail_drift_risk",
        "zero_crossing_mismatch",
        "reversal_count_exact",
    ]:
        rbf_col = f"{metric}__{RBF_MODEL}"
        key_col = f"{metric}__{KEYPOINT_MODEL}"
        if rbf_col in df.columns and key_col in df.columns:
            df[f"{metric}__selected"] = np.where(df["selected_keypoint"] == 1, df[key_col], df[rbf_col])
            df[f"{metric}__delta_selected_minus_rbf"] = df[f"{metric}__selected"] - df[rbf_col]
    return df


def confusion_table(df: pd.DataFrame) -> pd.DataFrame:
    test = df[df["split"] == "test"].copy()
    rows = []
    for selected in [0, 1]:
        for oracle in [0, 1]:
            subset = test[(test["selected_keypoint"] == selected) & (test["oracle_keypoint"] == oracle)]
            rows.append(
                {
                    "selected_keypoint": selected,
                    "oracle_keypoint_better": oracle,
                    "n_samples": int(len(subset)),
                    "mean_regret_vs_oracle": float(subset["selector_regret_vs_oracle"].mean()) if len(subset) else 0.0,
                    "mean_delta_vs_rbf": float(subset["selector_delta_vs_rbf"].mean()) if len(subset) else 0.0,
                }
            )
    return pd.DataFrame(rows)


def summary_by_group(df: pd.DataFrame) -> pd.DataFrame:
    test = df[df["split"] == "test"].copy()
    group_cols = ["subject", "road_design_module_name", "event_level", "is_large_response", "is_difficult_peak_top20"]
    rows = []
    for col in group_cols:
        if col not in test.columns:
            continue
        grouped = test.groupby(col, dropna=False)
        for key, g in grouped:
            rows.append(
                {
                    "group_type": col,
                    "group_value": key,
                    "n_samples": int(len(g)),
                    "selected_keypoint_rate": float(g["selected_keypoint"].mean()),
                    "oracle_keypoint_better_rate": float(g["oracle_keypoint"].mean()),
                    "selector_helped_count": int(g["selector_helped_rbf"].sum()),
                    "selector_hurt_count": int(g["selector_hurt_rbf"].sum()),
                    "mean_delta_vs_rbf": float(g["selector_delta_vs_rbf"].mean()),
                    "mean_regret_vs_oracle": float(g["selector_regret_vs_oracle"].mean()),
                    "mean_prob_keypoint": float(g["selector_prob_keypoint"].mean()),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["group_type", "mean_regret_vs_oracle", "n_samples"], ascending=[True, False, False])
    return out


def build_next_actions(test: pd.DataFrame, confusion: pd.DataFrame) -> pd.DataFrame:
    fn = int(confusion[(confusion["selected_keypoint"] == 0) & (confusion["oracle_keypoint_better"] == 1)]["n_samples"].sum())
    fp = int(confusion[(confusion["selected_keypoint"] == 1) & (confusion["oracle_keypoint_better"] == 0)]["n_samples"].sum())
    rows = [
        {
            "priority": 1,
            "action": "先复盘FN_missed_keypoint_gain样本",
            "why": f"当前 test 中漏选 keypoint 的样本数为 {fn}，这是 selector 没吃到 oracle/keypoint 上限的主要来源。",
        },
        {
            "priority": 2,
            "action": "控制FP_select_keypoint_hurts样本",
            "why": f"当前 test 中错选 keypoint 的样本数为 {fp}，这类样本直接拉高 selector RMSE。",
        },
        {
            "priority": 3,
            "action": "加入可靠性/不确定性特征",
            "why": "当前 selector probability 与 keypoint 是否真的更好仍有重叠，需要引入候选间差异、历史稳定性或响应形态风险特征。",
        },
        {
            "priority": 4,
            "action": "把选择目标从纯RMSE改成物理错误多目标",
            "why": "selector 虽然 RMSE 基本持平，但方向、大幅响应和困难样本有信号；下一版应显式惩罚错侧、严重幅值不足和尾段漂移。",
        },
        {
            "priority": 5,
            "action": "继续阻塞生理/EEG",
            "why": "selector 还未形成稳定可部署车辆-only提升，不能把剩余错误归因给新模态。",
        },
    ]
    return pd.DataFrame(rows)


def plot_confusion(confusion: pd.DataFrame) -> Path:
    matrix = np.zeros((2, 2), dtype=float)
    for _, row in confusion.iterrows():
        matrix[int(row["selected_keypoint"]), int(row["oracle_keypoint_better"])] = row["n_samples"]
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Oracle RBF", "Oracle keypoint"])
    ax.set_yticklabels(["Selected RBF", "Selected keypoint"])
    ax.set_xlabel("Which model actually lower RMSE")
    ax.set_ylabel("Selector choice")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(matrix[i, j])}", ha="center", va="center", color="#111827")
    ax.set_title("RBF vs keypoint selector confusion, test")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = FIG_DIR / "keypoint_selector_confusion_matrix.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_regret(test: pd.DataFrame) -> Path:
    view = test.sort_values("selector_regret_vs_oracle", ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    labels = [
        f"{row.subject}/{row.road_design_module_name}/{row.anchor_time_rel_s:.1f}s"
        for row in view.itertuples(index=False)
    ]
    colors = ["#dc2626" if "FP" in x else "#f97316" if "FN" in x else "#6b7280" for x in view["selection_outcome"]]
    ax.barh(labels, view["selector_regret_vs_oracle"], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("RMSE regret vs oracle lower is better")
    ax.set_title("Top selector regret samples")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    out = FIG_DIR / "keypoint_selector_top_regret_samples.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_probability(test: pd.DataFrame) -> Path:
    color_map = {
        "TP_select_keypoint_correct": "#16a34a",
        "FP_select_keypoint_hurts": "#dc2626",
        "FN_missed_keypoint_gain": "#f97316",
        "TN_keep_rbf_correct": "#2563eb",
    }
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for outcome, g in test.groupby("selection_outcome"):
        ax.scatter(
            g["selector_prob_keypoint"],
            g["keypoint_delta_vs_rbf"],
            label=outcome,
            s=55,
            alpha=0.8,
            color=color_map.get(outcome, "#6b7280"),
        )
    threshold = float(test["selector_threshold"].dropna().iloc[0])
    ax.axvline(threshold, color="#111827", linestyle="--", linewidth=1, label=f"threshold={threshold:.2f}")
    ax.axhline(0, color="#111827", linewidth=1)
    ax.set_xlabel("selector probability of keypoint")
    ax.set_ylabel("keypoint RMSE minus RBF; below 0 means keypoint better")
    ax.set_title("Selector probability vs actual keypoint gain, test")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out = FIG_DIR / "keypoint_selector_probability_vs_gain.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


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


def write_reports(test: pd.DataFrame, confusion: pd.DataFrame, group_summary: pd.DataFrame, next_actions: pd.DataFrame, figs: dict[str, Path]) -> tuple[Path, Path]:
    n_test = int(len(test))
    selected_rate = float(test["selected_keypoint"].mean())
    oracle_rate = float(test["oracle_keypoint"].mean())
    helped = int(test["selector_helped_rbf"].sum())
    hurt = int(test["selector_hurt_rbf"].sum())
    mean_delta = float(test["selector_delta_vs_rbf"].mean())
    mean_regret = float(test["selector_regret_vs_oracle"].mean())
    fn = int(confusion[(confusion["selected_keypoint"] == 0) & (confusion["oracle_keypoint_better"] == 1)]["n_samples"].sum())
    fp = int(confusion[(confusion["selected_keypoint"] == 1) & (confusion["oracle_keypoint_better"] == 0)]["n_samples"].sum())
    tp = int(confusion[(confusion["selected_keypoint"] == 1) & (confusion["oracle_keypoint_better"] == 1)]["n_samples"].sum())
    tn = int(confusion[(confusion["selected_keypoint"] == 0) & (confusion["oracle_keypoint_better"] == 0)]["n_samples"].sum())
    confusion_md = md_table(confusion, ["selected_keypoint", "oracle_keypoint_better", "n_samples", "mean_regret_vs_oracle", "mean_delta_vs_rbf"], max_rows=4)
    group_md = md_table(
        group_summary.sort_values("mean_regret_vs_oracle", ascending=False),
        ["group_type", "group_value", "n_samples", "selected_keypoint_rate", "oracle_keypoint_better_rate", "mean_regret_vs_oracle"],
        max_rows=12,
    )
    user = f"""# Stage 6b 用户查看版：RBF/keypoint 选择器错误复盘 v0.1

## 为什么做

阶段 6 审计发现，`selector_logreg_rbf_keypoint_no_subject` 没有明显超过 RBF，但它在方向、大幅响应和困难样本上有一些信号。这个阶段要回答：选择器是因为错选 keypoint 变差，还是因为漏掉了 keypoint 本来能改善的样本。

## 检查了什么

- 只看 B 轨道 test 40 个样本。
- 比较每个样本中 RBF 和 keypoint 谁的 RMSE 更低。
- 检查 selector 实际选了谁。
- 统计 TP/FP/FN/TN、oracle regret、相对 RBF 是帮了还是害了。
- 没有使用生理、脑电、连续风格、驾驶员 ID、服务器或服务器密码文件。

## 目前发现

- test 样本数：{n_test}
- selector 选择 keypoint 比例：{selected_rate:.3f}
- oracle 中 keypoint 更优比例：{oracle_rate:.3f}
- TP 选对 keypoint：{tp}；FP 错选 keypoint：{fp}；FN 漏选 keypoint：{fn}；TN 保持 RBF 正确：{tn}
- selector 相对 RBF 帮助样本：{helped}；伤害样本：{hurt}
- selector 平均 RMSE delta vs RBF：{mean_delta:+.6f}
- selector 平均 oracle regret：{mean_regret:.6f}

## 当前判断

选择器不是完全没信号，但当前概率阈值和特征还不能稳定识别“keypoint 真正更好”的样本。下一步应优先复盘 FN 和 FP，而不是直接加入生理/EEG。

## 推荐优先查看

1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06b_keypoint_selector_error_review_v0_1/tables/keypoint_selector_confusion_table.csv`
2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06b_keypoint_selector_error_review_v0_1/tables/keypoint_selector_top_regret_samples.csv`
3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06b_keypoint_selector_error_review_v0_1/figures/keypoint_selector_probability_vs_gain.png`
4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06b_keypoint_selector_error_review_v0_1/figures/keypoint_selector_top_regret_samples.png`
"""
    tech = f"""# Stage 6b：RBF/keypoint 选择器错误复盘 v0.1

生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}

## 输入和边界

- 输入：`rbf_keypoint_selector_training_table.csv`、`rbf_keypoint_selector_decisions.csv`、`rbf_keypoint_selector_selected_per_sample_metrics.csv`
- 只读已有车辆-only结果，不训练新模型。
- 不使用生理、脑电、连续风格、驾驶员 ID 或服务器。

## Test 混淆表

{confusion_md}

## 高 regret 分组

{group_md}

## 下一步动作

{md_table(next_actions, ['priority', 'action', 'why'], max_rows=10)}

## 图

- 混淆矩阵：`{str(figs['confusion']).replace(chr(92), '/')}`
- top regret 样本：`{str(figs['regret']).replace(chr(92), '/')}`
- probability vs actual gain：`{str(figs['probability']).replace(chr(92), '/')}`
"""
    user_path = REPORT_DIR / "stage06b_keypoint_selector_error_review_user_summary_cn.md"
    tech_path = REPORT_DIR / "stage06b_keypoint_selector_error_review_v0_1_cn.md"
    user_path.write_text(user, encoding="utf-8")
    tech_path.write_text(tech, encoding="utf-8")
    return user_path, tech_path


def main() -> None:
    ensure_dirs()
    detail = load_detail()
    test = detail[detail["split"] == "test"].copy()
    confusion = confusion_table(detail)
    group_summary = summary_by_group(detail)
    next_actions = build_next_actions(test, confusion)

    detail_out = TABLE_DIR / "keypoint_selector_sample_detail.csv"
    confusion_out = TABLE_DIR / "keypoint_selector_confusion_table.csv"
    group_out = TABLE_DIR / "keypoint_selector_group_summary.csv"
    top_regret_out = TABLE_DIR / "keypoint_selector_top_regret_samples.csv"
    missed_out = TABLE_DIR / "keypoint_selector_missed_keypoint_gain_samples.csv"
    false_positive_out = TABLE_DIR / "keypoint_selector_false_keypoint_samples.csv"
    next_out = TABLE_DIR / "keypoint_selector_next_actions.csv"

    detail.to_csv(detail_out, index=False, encoding="utf-8-sig")
    confusion.to_csv(confusion_out, index=False, encoding="utf-8-sig")
    group_summary.to_csv(group_out, index=False, encoding="utf-8-sig")
    test.sort_values("selector_regret_vs_oracle", ascending=False).head(20).to_csv(top_regret_out, index=False, encoding="utf-8-sig")
    test[test["selection_outcome"] == "FN_missed_keypoint_gain"].sort_values("selector_regret_vs_oracle", ascending=False).to_csv(missed_out, index=False, encoding="utf-8-sig")
    test[test["selection_outcome"] == "FP_select_keypoint_hurts"].sort_values("selector_regret_vs_oracle", ascending=False).to_csv(false_positive_out, index=False, encoding="utf-8-sig")
    next_actions.to_csv(next_out, index=False, encoding="utf-8-sig")

    figs = {
        "confusion": plot_confusion(confusion),
        "regret": plot_regret(test),
        "probability": plot_probability(test),
    }
    user_path, tech_path = write_reports(test, confusion, group_summary, next_actions, figs)
    summary = {
        "output_version": "stage06b_keypoint_selector_error_review_v0_1",
        "run_time_local": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "test_n": int(len(test)),
        "selected_keypoint_rate": float(test["selected_keypoint"].mean()),
        "oracle_keypoint_better_rate": float(test["oracle_keypoint"].mean()),
        "selector_helped_rbf_count": int(test["selector_helped_rbf"].sum()),
        "selector_hurt_rbf_count": int(test["selector_hurt_rbf"].sum()),
        "mean_selector_delta_vs_rbf": float(test["selector_delta_vs_rbf"].mean()),
        "mean_selector_regret_vs_oracle": float(test["selector_regret_vs_oracle"].mean()),
        "false_positive_keypoint_count": int((test["selection_outcome"] == "FP_select_keypoint_hurts").sum()),
        "false_negative_missed_keypoint_count": int((test["selection_outcome"] == "FN_missed_keypoint_gain").sum()),
        "stage05_physio_eeg_allowed": "blocked",
        "next_route": "selector_feature_revision_and_reliability_gate",
        "detail_path": str(detail_out).replace("\\", "/"),
        "confusion_path": str(confusion_out).replace("\\", "/"),
        "top_regret_path": str(top_regret_out).replace("\\", "/"),
        "user_summary_path": str(user_path).replace("\\", "/"),
        "technical_report_path": str(tech_path).replace("\\", "/"),
        "server_used": False,
        "server_credential_file_read": False,
        "uses_physio": False,
        "uses_eeg": False,
        "uses_continuous_style": False,
        "raw_files_modified": False,
    }
    summary_path = LOG_DIR / "keypoint_selector_error_review_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
