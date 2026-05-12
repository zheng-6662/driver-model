# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd


ROOT = Path("F:/data_set_process/data_process/05_rebuild_from_raw_20260511")
OUTPUT_VERSION = "stage07i_stability_calibrated_selection_v0_1"
TRACK_ID = "B_response3s_strict_core"

STAGE7G_ROOT = ROOT / "07_multihypothesis" / "stage07g_keypoint_segment_candidates_v0_1"
STAGE7H_ROOT = ROOT / "07_multihypothesis" / "stage07h_val_test_selection_diagnostics_v0_1"
STAGE7G_SUMMARY = STAGE7G_ROOT / "logs" / "stage07g_keypoint_segment_candidates_summary.json"
STAGE7G_METRICS = STAGE7G_ROOT / "tables" / "stage07g_candidate_metrics.csv"
STAGE7G_PER_SAMPLE = STAGE7G_ROOT / "tables" / "stage07g_candidate_per_sample_metrics.csv"
STAGE7H_STABILITY = STAGE7H_ROOT / "tables" / "stage07h_candidate_split_stability.csv"

OUT_ROOT = ROOT / "07_multihypothesis" / OUTPUT_VERSION
TABLE_DIR = OUT_ROOT / "tables"
FIG_DIR = OUT_ROOT / "figures"
LOG_DIR = OUT_ROOT / "logs"
REPORT_DIR = ROOT / "09_reports"

RBF_MODEL = "rbf_kernel_ridge_context_no_subject"
ORACLE_MODEL = "keypoint_segment_oracle"


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIG_DIR, LOG_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def path_str(path: Path) -> str:
    return str(path).replace("\\", "/")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    for path in [STAGE7G_SUMMARY, STAGE7G_METRICS, STAGE7G_PER_SAMPLE, STAGE7H_STABILITY]:
        if not path.exists():
            raise FileNotFoundError(path)
    summary = json.loads(STAGE7G_SUMMARY.read_text(encoding="utf-8"))
    metrics = pd.read_csv(STAGE7G_METRICS)
    per_sample = pd.read_csv(STAGE7G_PER_SAMPLE)
    stability = pd.read_csv(STAGE7H_STABILITY)
    return metrics, per_sample, stability, summary


def to_float(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def add_score_columns(stability: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [c for c in stability.columns if c not in {"model_name", "diagnostic_status"}]
    out = to_float(stability, numeric_cols)
    rbf_rows = out[out["model_name"].eq(RBF_MODEL)]
    if rbf_rows.empty:
        raise ValueError(f"Missing reference model in stability table: {RBF_MODEL}")
    rbf = rbf_rows.iloc[0]
    for split in ["train", "val", "test"]:
        out[f"difficult_rmse_delta_vs_rbf_{split}"] = out[f"difficult_top20_rmse_{split}"] - float(rbf[f"difficult_top20_rmse_{split}"])
        out[f"wrong_side_delta_vs_rbf_{split}"] = out[f"wrong_side_rate_{split}"] - float(rbf[f"wrong_side_rate_{split}"])
        out[f"large_recall_delta_vs_rbf_{split}"] = out[f"large_response_recall_{split}"] - float(rbf[f"large_response_recall_{split}"])
    out = out[~out["model_name"].astype(str).str.contains("oracle", case=False, na=False)].copy()
    out = out[out["model_name"].ne(RBF_MODEL)].copy()
    out["abs_train_val_delta_gap"] = (out["rmse_delta_vs_rbf_val"] - out["rmse_delta_vs_rbf_train"]).abs()
    out["score_stability_l05"] = out["rmse_delta_vs_rbf_val"] + 0.5 * out["abs_train_val_delta_gap"]
    out["score_stability_l10"] = out["rmse_delta_vs_rbf_val"] + 1.0 * out["abs_train_val_delta_gap"]
    out["score_val_plus_difficult"] = out["rmse_delta_vs_rbf_val"] + 0.35 * out["difficult_rmse_delta_vs_rbf_val"]
    out["score_val_plus_physical"] = (
        out["rmse_delta_vs_rbf_val"]
        + 0.04 * out["wrong_side_delta_vs_rbf_val"].fillna(0.0)
        - 0.02 * out["large_recall_delta_vs_rbf_val"].fillna(0.0)
        + 0.20 * out["difficult_rmse_delta_vs_rbf_val"].fillna(0.0)
    )
    return out


def metric_lookup(metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = to_float(
        metrics,
        [
            "rmse_steer",
            "rmse_delta_vs_rbf",
            "wrong_side_rate",
            "large_response_recall",
            "difficult_top20_rmse",
            "difficult_rmse_delta_vs_rbf",
            "peak_amp_mae",
            "tail_abs_error_mean",
        ],
    )
    keep = [
        "model_name",
        "split",
        "rmse_steer",
        "rmse_delta_vs_rbf",
        "wrong_side_rate",
        "large_response_recall",
        "difficult_top20_rmse",
        "difficult_rmse_delta_vs_rbf",
        "peak_amp_mae",
        "tail_abs_error_mean",
    ]
    return metrics[keep].copy()


def choose_by_score(score_table: pd.DataFrame, score_col: str, require_val_rmse_nonworse: bool = False) -> str:
    candidates = score_table.copy()
    if require_val_rmse_nonworse:
        candidates = candidates[candidates["rmse_delta_vs_rbf_val"] <= 0.002]
    if candidates.empty:
        return RBF_MODEL
    return str(candidates.sort_values([score_col, "rmse_delta_vs_rbf_val"]).iloc[0]["model_name"])


def build_policy_table(score_table: pd.DataFrame, metrics: pd.DataFrame, stage7g_summary: dict[str, Any]) -> pd.DataFrame:
    original_selected = str(stage7g_summary["selected_policy"])
    policy_defs = [
        ("stage7g_original_val_best", original_selected, "original Stage 7g validation RMSE gate"),
        ("stability_penalty_l05", choose_by_score(score_table, "score_stability_l05"), "score=val_delta+0.5*abs(train_delta-val_delta)"),
        ("stability_penalty_l10", choose_by_score(score_table, "score_stability_l10"), "score=val_delta+1.0*abs(train_delta-val_delta)"),
        ("val_plus_difficult", choose_by_score(score_table, "score_val_plus_difficult", require_val_rmse_nonworse=True), "score=val_delta+0.35*difficult_delta; require val nonworse"),
        ("val_plus_physical", choose_by_score(score_table, "score_val_plus_physical", require_val_rmse_nonworse=True), "score combines val RMSE, wrong-side, large recall and difficult RMSE"),
    ]
    metric = metric_lookup(metrics)
    rows: list[dict[str, Any]] = []
    for policy_name, selected_model, rule in policy_defs:
        for split in ["train", "val", "test"]:
            row = metric[(metric["model_name"].eq(selected_model)) & (metric["split"].eq(split))]
            if row.empty:
                continue
            rec = row.iloc[0].to_dict()
            rec["policy_name"] = policy_name
            rec["selected_model"] = selected_model
            rec["selection_rule"] = rule
            rows.append(rec)
    return pd.DataFrame(rows)


def selected_policy_summary(policy_table: pd.DataFrame) -> pd.DataFrame:
    test = policy_table[policy_table["split"].eq("test")].copy()
    val = policy_table[policy_table["split"].eq("val")][["policy_name", "rmse_delta_vs_rbf", "wrong_side_rate", "large_response_recall", "difficult_rmse_delta_vs_rbf"]].rename(
        columns={
            "rmse_delta_vs_rbf": "val_rmse_delta_vs_rbf",
            "wrong_side_rate": "val_wrong_side_rate",
            "large_response_recall": "val_large_response_recall",
            "difficult_rmse_delta_vs_rbf": "val_difficult_delta_vs_rbf",
        }
    )
    out = test.merge(val, on="policy_name", how="left")
    out["test_rmse_improved"] = out["rmse_delta_vs_rbf"] < -1e-6
    out["test_physical_or_difficult_gain"] = (
        (out["difficult_rmse_delta_vs_rbf"] < -1e-6)
        | (out["wrong_side_rate"] < out["val_wrong_side_rate"])
        | (out["large_response_recall"] > out["val_large_response_recall"])
    )
    return out.sort_values(["rmse_delta_vs_rbf", "difficult_rmse_delta_vs_rbf"])


def per_sample_policy_gain(per_sample: pd.DataFrame, selected_model: str) -> pd.DataFrame:
    per_sample = to_float(per_sample, ["sample_rmse", "wrong_side", "is_large_response", "is_difficult_peak_top20"])
    keep = per_sample[per_sample["model_name"].isin([RBF_MODEL, selected_model])].copy()
    piv = keep.pivot_table(index="sample_id", columns="model_name", values="sample_rmse", aggfunc="first")
    meta_cols = [
        "sample_id",
        "event_uid",
        "subject",
        "session_stamp",
        "split",
        "gt_peak_abs",
        "is_large_response",
        "is_difficult_peak_top20",
    ]
    meta = per_sample[per_sample["model_name"].eq(RBF_MODEL)][meta_cols].drop_duplicates("sample_id")
    out = meta.copy()
    out["rbf_sample_rmse"] = piv.loc[out["sample_id"], RBF_MODEL].to_numpy(dtype=float)
    out["selected_sample_rmse"] = piv.loc[out["sample_id"], selected_model].to_numpy(dtype=float)
    out["gain_over_rbf"] = out["rbf_sample_rmse"] - out["selected_sample_rmse"]
    out["selected_better_than_rbf"] = out["gain_over_rbf"] > 0
    return out.sort_values(["split", "gain_over_rbf"], ascending=[True, False])


def gate_table(selected_policy: str, selected_model: str, summary: pd.DataFrame) -> pd.DataFrame:
    row = summary[summary["policy_name"].eq(selected_policy)].iloc[0]
    test_delta = float(row["rmse_delta_vs_rbf"])
    difficult_delta = float(row["difficult_rmse_delta_vs_rbf"])
    weak_continue = selected_model != RBF_MODEL and test_delta < 0 and difficult_delta < 0
    return pd.DataFrame(
        [
            {"gate_item": "selected_calibration_policy", "status": selected_policy, "evidence": f"selected_model={selected_model}"},
            {"gate_item": "deployable_upgrade", "status": "weak_candidate_continue" if weak_continue else "no_upgrade", "evidence": f"test RMSE delta {test_delta:+.6f}; difficult RMSE delta {difficult_delta:+.6f}; needs repeat validation"},
            {"gate_item": "mainline_upgrade", "status": "not_final", "evidence": "single split evidence only; no repeated validation or held-out confirmation beyond current test"},
            {"gate_item": "stage08_physio_eeg_allowed", "status": "blocked", "evidence": "vehicle-only selection calibration still needs robustness check"},
            {"gate_item": "server_used", "status": "no", "evidence": "local diagnostic/selection run only; credential file not read"},
        ]
    )


def plot_policy_summary(policy_summary: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.3))
    labels = policy_summary["policy_name"].astype(str).tolist()
    x = np.arange(len(labels))
    for ax, col, title in [
        (axes[0], "rmse_delta_vs_rbf", "Test RMSE delta"),
        (axes[1], "difficult_rmse_delta_vs_rbf", "Test difficult RMSE delta"),
        (axes[2], "val_rmse_delta_vs_rbf", "Val RMSE delta"),
    ]:
        ax.bar(x, policy_summary[col].astype(float), color="#4c78a8")
        ax.axhline(0, color="#999999", linewidth=0.9)
        ax.set_title(title)
        ax.set_xticks(x, labels, rotation=35, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Stage 7i policy comparison", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_score_components(score_table: pd.DataFrame, path: Path) -> None:
    cols = ["model_name", "rmse_delta_vs_rbf_train", "rmse_delta_vs_rbf_val", "abs_train_val_delta_gap", "score_stability_l05"]
    top = score_table.sort_values("score_stability_l05")[cols].head(8).copy()
    x = np.arange(len(top))
    width = 0.24
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.bar(x - width, top["rmse_delta_vs_rbf_train"], width, label="train delta")
    ax.bar(x, top["rmse_delta_vs_rbf_val"], width, label="val delta")
    ax.bar(x + width, top["score_stability_l05"], width, label="stability score")
    ax.axhline(0, color="#999999", linewidth=0.9)
    ax.set_xticks(x, [m.replace("_", "\n") for m in top["model_name"]], rotation=0, fontsize=7)
    ax.set_title("Top stability-calibrated candidate scores")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_gain_distribution(gain: pd.DataFrame, selected_model: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    data = [gain[gain["split"].eq(split)]["gain_over_rbf"].dropna().to_numpy() for split in ["train", "val", "test"]]
    ax.boxplot(data, tick_labels=["train", "val", "test"], showfliers=False)
    ax.axhline(0, color="#999999", linewidth=0.9)
    ax.set_ylabel("sample RMSE gain over RBF")
    ax.set_title(f"Per-sample gain for {selected_model}")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_reports(
    policy_summary: pd.DataFrame,
    score_table: pd.DataFrame,
    gain: pd.DataFrame,
    gate: pd.DataFrame,
    selected_policy: str,
    selected_model: str,
    figures: dict[str, str],
) -> None:
    row = policy_summary[policy_summary["policy_name"].eq(selected_policy)].iloc[0]
    test_delta = float(row["rmse_delta_vs_rbf"])
    test_rmse = float(row["rmse_steer"])
    difficult_delta = float(row["difficult_rmse_delta_vs_rbf"])
    wrong_side = float(row["wrong_side_rate"])
    large_recall = float(row["large_response_recall"])
    policy_text = policy_summary[
        [
            "policy_name",
            "selected_model",
            "val_rmse_delta_vs_rbf",
            "rmse_delta_vs_rbf",
            "wrong_side_rate",
            "large_response_recall",
            "difficult_rmse_delta_vs_rbf",
        ]
    ].to_string(index=False)
    score_text = score_table.sort_values("score_stability_l05")[
        [
            "model_name",
            "rmse_delta_vs_rbf_train",
            "rmse_delta_vs_rbf_val",
            "score_stability_l05",
            "rmse_delta_vs_rbf_test",
            "difficult_rmse_delta_vs_rbf_test",
        ]
    ].head(10).to_string(index=False)
    gain_text = gain.groupby("split").agg(
        n_samples=("gain_over_rbf", "size"),
        mean_gain=("gain_over_rbf", "mean"),
        median_gain=("gain_over_rbf", "median"),
        positive_gain_rate=("selected_better_than_rbf", "mean"),
    ).reset_index().to_string(index=False)
    user = f"""# Stage 7i 用户查看版：稳定性校准候选选择 v0.1

## 这个阶段为什么做

Stage 7h 发现 Stage 7g 的问题不是候选完全没用，而是 validation 按最小 RMSE 选到了 test 上退化的候选。Stage 7i 不训练新模型，只用 train/val 重新设计更保守的选择规则，检查能否选出更稳定的车辆-only 候选。

## 这个阶段检查了什么

- 原始 Stage 7g val-best 规则。
- train/val 稳定性惩罚规则：`val_delta + 0.5 * abs(train_delta - val_delta)`。
- 更强惩罚、困难样本和物理指标加权规则。
- 每个规则只能用 train/val 信息选候选，test 只做最终报告。

## 目前发现了什么

- 当前推荐继续观察的规则：`{selected_policy}`。
- 该规则选中的候选：`{selected_model}`。
- test RMSE={test_rmse:.6f}，相对 RBF/KNN delta={test_delta:+.6f}。
- difficult RMSE delta={difficult_delta:+.6f}，wrong-side={wrong_side:.3f}，large recall={large_recall:.3f}。
- gate={gate.set_index("gate_item").loc["deployable_upgrade", "status"]}。

## 规则对照

```text
{policy_text}
```

## 稳定性分数前十

```text
{score_text}
```

## 逐样本收益

```text
{gain_text}
```

## 哪些结果可信

可信的是：这个规则没有看 test 标签来选候选，选择只来自 train/val 的稳定性分数；它比原始 val-best 规则更符合 Stage 7h 暴露出的风险。

## 哪些结果还不能下结论

这还不能直接升为最终主线。原因是目前只有一个固定 session-level split，没有多折验证；收益主要是 RMSE 和困难样本 RMSE，错侧率和大幅响应召回没有进一步改善。它只能作为“弱候选继续验证”。

## 下一阶段是否可以继续

可以继续做 Stage 7j：对该稳定性校准规则做多折 session validation 或重新构建分层 validation，再决定是否把它冻结为车辆-only 主候选。生理/EEG 仍不进入。

## 推荐优先查看

1. `{figures["policy_summary"]}`
2. `{figures["score_components"]}`
3. `{figures["gain_distribution"]}`
4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07i_stability_calibrated_selection_v0_1/tables/stage07i_policy_test_summary.csv`
5. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07i_stability_calibrated_selection_v0_1/tables/stage07i_gate_table.csv`
"""
    (REPORT_DIR / "stage07i_stability_calibrated_selection_user_summary_cn.md").write_text(user, encoding="utf-8-sig")

    tech = f"""# Stage 7i 技术报告：stability-calibrated non-oracle selection v0.1

## Scope

- Source candidates: Stage 7g.
- No new trajectory model training.
- Selection uses train/val only; test is final reporting.
- No server used. Credential file not read.

## Selected Rule

- policy=`{selected_policy}`
- selected_model=`{selected_model}`
- test_delta_vs_rbf={test_delta:+.6f}
- difficult_delta_vs_rbf={difficult_delta:+.6f}

## Gate

```text
{gate.to_string(index=False)}
```
"""
    (REPORT_DIR / "stage07i_stability_calibrated_selection_v0_1_cn.md").write_text(tech, encoding="utf-8-sig")


def main() -> None:
    ensure_dirs()
    metrics, per_sample, stability, stage7g_summary = load_inputs()
    score_table = add_score_columns(stability)
    policy_table = build_policy_table(score_table, metrics, stage7g_summary)
    policy_summary = selected_policy_summary(policy_table)
    selected_policy = "stability_penalty_l05"
    selected_model = str(policy_summary[policy_summary["policy_name"].eq(selected_policy)].iloc[0]["selected_model"])
    gain = per_sample_policy_gain(per_sample, selected_model)
    gate = gate_table(selected_policy, selected_model, policy_summary)

    figures = {
        "policy_summary": path_str(FIG_DIR / "stage07i_policy_summary.png"),
        "score_components": path_str(FIG_DIR / "stage07i_stability_score_components.png"),
        "gain_distribution": path_str(FIG_DIR / "stage07i_selected_gain_distribution.png"),
    }
    plot_policy_summary(policy_summary, Path(figures["policy_summary"]))
    plot_score_components(score_table, Path(figures["score_components"]))
    plot_gain_distribution(gain, selected_model, Path(figures["gain_distribution"]))

    score_table.to_csv(TABLE_DIR / "stage07i_candidate_score_table.csv", index=False, encoding="utf-8-sig")
    policy_table.to_csv(TABLE_DIR / "stage07i_policy_split_metrics.csv", index=False, encoding="utf-8-sig")
    policy_summary.to_csv(TABLE_DIR / "stage07i_policy_test_summary.csv", index=False, encoding="utf-8-sig")
    gain.to_csv(TABLE_DIR / "stage07i_selected_policy_gain_samples.csv", index=False, encoding="utf-8-sig")
    gate.to_csv(TABLE_DIR / "stage07i_gate_table.csv", index=False, encoding="utf-8-sig")

    write_reports(policy_summary, score_table, gain, gate, selected_policy, selected_model, figures)
    row = policy_summary[policy_summary["policy_name"].eq(selected_policy)].iloc[0]
    output_summary = {
        "output_version": OUTPUT_VERSION,
        "track_id": TRACK_ID,
        "source_stage": "stage07g_keypoint_segment_candidates_v0_1",
        "selected_policy": selected_policy,
        "selected_model": selected_model,
        "test_rmse": float(row["rmse_steer"]),
        "test_delta_vs_rbf": float(row["rmse_delta_vs_rbf"]),
        "test_difficult_delta_vs_rbf": float(row["difficult_rmse_delta_vs_rbf"]),
        "gate_status": str(gate.set_index("gate_item").loc["deployable_upgrade", "status"]),
        "mainline_upgrade": "not_final",
        "stage08_physio_eeg_allowed": False,
        "server_used": False,
        "credential_file_read": False,
        "raw_files_modified": False,
        "uses_physio": False,
        "uses_eeg": False,
        "uses_continuous_style": False,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "figures": figures,
    }
    (LOG_DIR / "stage07i_stability_calibrated_selection_summary.json").write_text(json.dumps(output_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
