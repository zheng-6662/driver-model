from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("F:/data_set_process/data_process/05_rebuild_from_raw_20260511")
BASELINE_SCRIPT_DIR = ROOT / "03_baselines" / "scripts"
if str(BASELINE_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_SCRIPT_DIR))

import evaluate_stage3_vehicle_baselines as eval_utils  # noqa: E402


TRACK_ID = "B_response3s_strict_core"
OUT_ROOT = ROOT / "06_structured_models" / "stage06e_multicandidate_oracle_gap_v0_1"
TABLE_DIR = OUT_ROOT / "tables"
FIG_DIR = OUT_ROOT / "figures"
LOG_DIR = OUT_ROOT / "logs"
REPORT_DIR = ROOT / "09_reports"

SOURCE_TABLES = [
    ROOT / "03_baselines" / "stage03_vehicle_instability_clean_task_vehicle_baselines_v0_1" / "tables" / "clean_task_vehicle_per_sample_metrics.csv",
    ROOT / "03_baselines" / "stage03_vehicle_instability_clean_task_vehicle_transformer_v0_1" / "tables" / "clean_task_vehicle_transformer_per_sample_metrics.csv",
    ROOT / "03_baselines" / "stage03_vehicle_instability_structured_vehicle_transformer_v0_1" / "tables" / "structured_vehicle_transformer_per_sample_metrics.csv",
    ROOT / "03_baselines" / "stage03_vehicle_instability_keypoint_residual_vehicle_transformer_v0_1" / "tables" / "keypoint_residual_vehicle_transformer_per_sample_metrics.csv",
    ROOT / "03_baselines" / "stage03_vehicle_instability_topk_vehicle_transformer_v0_1" / "tables" / "topk_vehicle_transformer_per_sample_metrics.csv",
    ROOT / "03_baselines" / "stage03_vehicle_instability_topk_reliability_selector_v0_1" / "tables" / "topk_reliability_selector_per_sample_metrics.csv",
    ROOT / "03_baselines" / "stage03_vehicle_instability_rbf_keypoint_multihypothesis_review_v0_1" / "tables" / "multihypothesis_per_sample_metrics.csv",
]

STAGE6D_METRICS = ROOT / "06_structured_models" / "stage06d_reliability_gate_v0_1" / "tables" / "reliability_gate_policy_metrics.csv"

RBF_MODEL = "rbf_kernel_ridge_context_no_subject"
KEYPOINT_MODEL = "keypoint_residual_vehicle_transformer_no_subject"
DEPLOYABLE_SELECTOR_MODELS = {
    "selector_logreg_rbf_keypoint_no_subject",
    "topk_rbf_branch_logreg_selector_no_subject",
    "topk_top1_rbf_fallback_logreg_no_subject",
    "topk_branch_logreg_selector_no_subject",
}
RAW_BRANCH_MODELS = [
    "rbf_kernel_ridge_context_no_subject",
    "ridge_rich_context_no_subject",
    "ridge_rich_history_no_subject",
    "knn_template_context_no_subject",
    "direction_gated_knn_template_no_subject",
    "peak_scaled_template_context_no_subject",
    "vehicle_transformer_context_no_subject",
    "structured_vehicle_transformer_aux_no_subject",
    "keypoint_residual_vehicle_transformer_no_subject",
    "topk_vehicle_transformer_branch0_no_subject",
    "topk_vehicle_transformer_branch1_no_subject",
    "topk_vehicle_transformer_branch2_no_subject",
]

POOL_DEFINITIONS = {
    "oracle_rbf_plus_keypoint": [RBF_MODEL, KEYPOINT_MODEL],
    "oracle_rbf_plus_topk3": [
        RBF_MODEL,
        "topk_vehicle_transformer_branch0_no_subject",
        "topk_vehicle_transformer_branch1_no_subject",
        "topk_vehicle_transformer_branch2_no_subject",
    ],
    "oracle_topk3_only": [
        "topk_vehicle_transformer_branch0_no_subject",
        "topk_vehicle_transformer_branch1_no_subject",
        "topk_vehicle_transformer_branch2_no_subject",
    ],
    "oracle_broad_vehicle_pool": RAW_BRANCH_MODELS,
}


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIG_DIR, LOG_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_candidate_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    manifest = []
    for source_idx, path in enumerate(SOURCE_TABLES):
        if not path.exists():
            manifest.append({"source_path": str(path).replace("\\", "/"), "exists": False, "rows_loaded": 0})
            continue
        df = pd.read_csv(path)
        df = df[df["track_id"].eq(TRACK_ID)].copy()
        df["source_priority"] = source_idx
        df["source_table"] = path.name
        frames.append(df)
        manifest.append({"source_path": str(path).replace("\\", "/"), "exists": True, "rows_loaded": int(len(df))})
    if not frames:
        raise RuntimeError("No source tables loaded.")
    rows = pd.concat(frames, ignore_index=True, sort=False)
    rows = rows.sort_values(["source_priority", "model_name", "sample_id"]).drop_duplicates(["split", "sample_id", "model_name"], keep="first")
    return rows, pd.DataFrame(manifest)


def model_availability(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.groupby(["model_name", "split"], as_index=False)
        .agg(n_samples=("sample_id", "nunique"), mean_sample_rmse=("sample_rmse", "mean"))
        .sort_values(["split", "mean_sample_rmse", "model_name"])
    )


def make_oracle_rows(rows: pd.DataFrame, pool_name: str, model_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = rows[rows["model_name"].isin(model_names)].copy()
    if sub.empty:
        raise RuntimeError(f"No rows for pool {pool_name}.")
    winner = sub.sort_values(["sample_id", "split", "sample_rmse"]).groupby(["split", "sample_id"], as_index=False).head(1).copy()
    winner_detail = winner[["split", "sample_id", "model_name", "sample_rmse"]].rename(columns={"model_name": "oracle_winner_model", "sample_rmse": "oracle_sample_rmse"})
    winner["model_name"] = pool_name
    return winner, winner_detail


def aggregate_rows(rows: pd.DataFrame) -> pd.DataFrame:
    metrics = eval_utils.aggregate_metrics(rows)
    metrics["track_id"] = TRACK_ID
    return metrics


def build_summary(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidate_models = sorted(set(RAW_BRANCH_MODELS) | DEPLOYABLE_SELECTOR_MODELS | {RBF_MODEL, KEYPOINT_MODEL})
    base_rows = rows[rows["model_name"].isin(candidate_models)].copy()
    oracle_frames = []
    winner_frames = []
    for pool_name, models in POOL_DEFINITIONS.items():
        oracle, winners = make_oracle_rows(rows, pool_name, models)
        oracle_frames.append(oracle)
        winners["pool_name"] = pool_name
        winner_frames.append(winners)
    all_eval_rows = pd.concat([base_rows, *oracle_frames], ignore_index=True, sort=False)
    metrics = aggregate_rows(all_eval_rows)
    winners = pd.concat(winner_frames, ignore_index=True, sort=False)
    return metrics, winners, base_rows


def winner_summary(winners: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    rbf = rows[rows["model_name"].eq(RBF_MODEL)][["split", "sample_id", "sample_rmse"]].rename(columns={"sample_rmse": "rbf_sample_rmse"})
    detail = winners.merge(rbf, on=["split", "sample_id"], how="left")
    detail["oracle_gain_vs_rbf"] = detail["rbf_sample_rmse"] - detail["oracle_sample_rmse"]
    return (
        detail.groupby(["pool_name", "split", "oracle_winner_model"], as_index=False)
        .agg(
            n_wins=("sample_id", "nunique"),
            mean_oracle_gain_vs_rbf=("oracle_gain_vs_rbf", "mean"),
            median_oracle_gain_vs_rbf=("oracle_gain_vs_rbf", "median"),
            mean_oracle_sample_rmse=("oracle_sample_rmse", "mean"),
        )
        .sort_values(["pool_name", "split", "n_wins"], ascending=[True, True, False])
    )


def oracle_gap_table(metrics: pd.DataFrame) -> pd.DataFrame:
    test = metrics[metrics["split"].eq("test")].copy()
    rbf = test[test["model_name"].eq(RBF_MODEL)].iloc[0]
    rows = []
    interesting = [
        RBF_MODEL,
        "ridge_rich_context_no_subject",
        "keypoint_residual_vehicle_transformer_no_subject",
        "selector_logreg_rbf_keypoint_no_subject",
        "topk_rbf_branch_logreg_selector_no_subject",
        "topk_top1_rbf_fallback_logreg_no_subject",
        "oracle_rbf_plus_keypoint",
        "oracle_rbf_plus_topk3",
        "oracle_topk3_only",
        "oracle_broad_vehicle_pool",
    ]
    for name in interesting:
        sub = test[test["model_name"].eq(name)]
        if sub.empty:
            continue
        row = sub.iloc[0].to_dict()
        row["delta_vs_rbf_rmse"] = float(row["rmse_steer"]) - float(rbf["rmse_steer"])
        row["delta_vs_rbf_wrong_side"] = float(row["wrong_side_rate"]) - float(rbf["wrong_side_rate"])
        row["delta_vs_rbf_large_recall"] = float(row["large_response_recall"]) - float(rbf["large_response_recall"])
        row["delta_vs_rbf_difficult_rmse"] = float(row["difficult_top20_rmse"]) - float(rbf["difficult_top20_rmse"])
        if name.startswith("oracle"):
            row["role"] = "oracle_upper_bound_not_deployable"
        elif name in DEPLOYABLE_SELECTOR_MODELS:
            row["role"] = "deployable_selector_attempt"
        elif name == RBF_MODEL:
            row["role"] = "current_rbf_knn_reference"
        else:
            row["role"] = "single_or_branch_candidate"
        rows.append(row)
    return pd.DataFrame(rows).sort_values("rmse_steer")


def merge_stage6d_metrics(gap: pd.DataFrame) -> pd.DataFrame:
    if not STAGE6D_METRICS.exists():
        return gap
    d = pd.read_csv(STAGE6D_METRICS)
    d = d[d["split"].eq("test") & d["policy_label"].isin(["val_best_rmse", "val_rmse_noninferior_conservative"])].copy()
    if d.empty:
        return gap
    rbf_rmse = float(gap[gap["model_name"].eq(RBF_MODEL)]["rmse_steer"].iloc[0])
    add_rows = []
    for _, row in d.iterrows():
        out = row.to_dict()
        out["model_name"] = f"stage06d_{row['policy_label']}"
        out["role"] = "deployable_selector_attempt"
        out["delta_vs_rbf_rmse"] = float(row["rmse_steer"]) - rbf_rmse
        out["delta_vs_rbf_wrong_side"] = np.nan
        out["delta_vs_rbf_large_recall"] = np.nan
        out["delta_vs_rbf_difficult_rmse"] = np.nan
        add_rows.append(out)
    return pd.concat([gap, pd.DataFrame(add_rows)], ignore_index=True, sort=False).sort_values("rmse_steer")


def sample_gain_detail(winners: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    rbf = rows[rows["model_name"].eq(RBF_MODEL)][["split", "sample_id", "sample_rmse", "wrong_side", "is_large_response", "severe_amp_under", "tail_drift_risk"]].rename(
        columns={"sample_rmse": "rbf_sample_rmse"}
    )
    detail = winners.merge(rbf, on=["split", "sample_id"], how="left")
    detail["oracle_gain_vs_rbf"] = detail["rbf_sample_rmse"] - detail["oracle_sample_rmse"]
    return detail.sort_values(["split", "pool_name", "oracle_gain_vs_rbf"], ascending=[True, True, False])


def build_gate(gap: pd.DataFrame) -> pd.DataFrame:
    test = gap.copy()
    rbf = test[test["model_name"].eq(RBF_MODEL)].iloc[0]
    broad = test[test["model_name"].eq("oracle_broad_vehicle_pool")].iloc[0]
    selectors = test[test["role"].eq("deployable_selector_attempt")].copy()
    best_selector = selectors.sort_values("rmse_steer").iloc[0] if not selectors.empty else None
    rows = [
        {
            "gate": "oracle_pool_signal",
            "status": "research_signal_not_deployable" if broad["rmse_steer"] < rbf["rmse_steer"] else "no_signal",
            "evidence": f"broad oracle test RMSE={broad['rmse_steer']:.6f}, delta={broad['delta_vs_rbf_rmse']:+.6f}; this uses labels for selection.",
            "decision": "只说明候选池存在上限，不可作为部署模型或生理有效性证据。",
        }
    ]
    if best_selector is not None:
        rows.append(
            {
                "gate": "deployable_selection_gap",
                "status": "blocked",
                "evidence": f"best deployable selector test RMSE={best_selector['rmse_steer']:.6f}, delta={best_selector['delta_vs_rbf_rmse']:+.6f}.",
                "decision": "Stage 7 若继续，必须优先解决非 oracle 选择策略；不能只报告 best-of-K。",
            }
        )
    rows.append(
        {
            "gate": "stage05_physio_eeg_allowed",
            "status": "blocked",
            "evidence": "车辆-only 多候选选择仍未闭环。",
            "decision": "继续阻塞生理/EEG增量结论。",
        }
    )
    return pd.DataFrame(rows)


def plot_oracle_gap(gap: pd.DataFrame, winners: pd.DataFrame) -> tuple[Path, Path]:
    test_gap = gap[gap["split"].eq("test") if "split" in gap.columns else [True] * len(gap)].copy()
    if "split" in test_gap.columns:
        test_gap = test_gap[test_gap["split"].eq("test") | test_gap["split"].isna()].copy()
    show = test_gap[test_gap["model_name"].isin([
        RBF_MODEL,
        "stage06d_val_rmse_noninferior_conservative",
        "stage06d_val_best_rmse",
        "selector_logreg_rbf_keypoint_no_subject",
        "topk_rbf_branch_logreg_selector_no_subject",
        "oracle_rbf_plus_keypoint",
        "oracle_rbf_plus_topk3",
        "oracle_broad_vehicle_pool",
    ])].copy()
    show["label"] = show["model_name"].map(
        {
            RBF_MODEL: "RBF/KNN ref",
            "stage06d_val_rmse_noninferior_conservative": "Stage6d conservative",
            "stage06d_val_best_rmse": "Stage6d aggressive",
            "selector_logreg_rbf_keypoint_no_subject": "RBF/keypoint selector",
            "topk_rbf_branch_logreg_selector_no_subject": "TopK selector",
            "oracle_rbf_plus_keypoint": "Oracle RBF+keypoint",
            "oracle_rbf_plus_topk3": "Oracle RBF+topK",
            "oracle_broad_vehicle_pool": "Oracle broad pool",
        }
    ).fillna(show["model_name"])
    show = show.sort_values("rmse_steer", ascending=False)

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ["#9ca3af" if not str(name).startswith("oracle") else "#2563eb" for name in show["model_name"]]
    ax.barh(show["label"], show["rmse_steer"], color=colors)
    ax.axvline(float(show[show["model_name"].eq(RBF_MODEL)]["rmse_steer"].iloc[0]), color="#111827", linestyle="--", linewidth=1)
    ax.set_xlabel("test RMSE")
    ax.set_title("Deployable selectors vs oracle upper bounds")
    fig.tight_layout()
    rmse_path = FIG_DIR / "multicandidate_oracle_gap_rmse.png"
    fig.savefig(rmse_path, dpi=180)
    plt.close(fig)

    test_winners = winners[(winners["split"].eq("test")) & (winners["pool_name"].eq("oracle_broad_vehicle_pool"))].copy()
    counts = test_winners["oracle_winner_model"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.barh(counts.index, counts.values, color="#0f766e")
    ax.set_xlabel("test samples won under broad oracle")
    ax.set_title("Which candidate wins only when oracle can choose?")
    fig.tight_layout()
    winner_path = FIG_DIR / "multicandidate_oracle_winner_counts.png"
    fig.savefig(winner_path, dpi=180)
    plt.close(fig)
    return rmse_path, winner_path


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


def write_reports(gap: pd.DataFrame, gate: pd.DataFrame, winner_counts: pd.DataFrame, rmse_fig: Path, winner_fig: Path) -> tuple[Path, Path]:
    rbf = gap[gap["model_name"].eq(RBF_MODEL)].iloc[0]
    broad = gap[gap["model_name"].eq("oracle_broad_vehicle_pool")].iloc[0]
    selectors = gap[gap["role"].eq("deployable_selector_attempt")].copy()
    best_selector = selectors.sort_values("rmse_steer").iloc[0]
    user = f"""# Stage 6e 用户查看版：多候选 oracle gap 复核 v0.1

## 为什么做

Stage 6d 说明当前 RBF/keypoint reliability gate 不能升级。这个阶段不训练新模型，只把已有车辆-only候选放到同一个候选池里，检查“如果有 oracle 按真实标签挑候选，上限有多高”和“实际可部署 selector 离上限差多远”。

## 目前发现

- 当前 RBF/KNN 主参照 test RMSE={rbf['rmse_steer']:.6f}。
- broad oracle pool test RMSE={broad['rmse_steer']:.6f}，相对 RBF/KNN delta={broad['delta_vs_rbf_rmse']:+.6f}；这个结果不可部署，因为它用真实标签挑选最佳候选。
- 当前最好的可部署 selector test RMSE={best_selector['rmse_steer']:.6f}，相对 RBF/KNN delta={best_selector['delta_vs_rbf_rmse']:+.6f}，没有把 oracle 上限稳定转成实际增益。
- 结论不是“Transformer 更好”，也不是“生理该进来”；结论是车辆-only 多候选路线存在上限，但选择策略还没解决。

## 当前判断

可以进入 Stage 7 的前提不是继续报告 best-of-K，而是建立不用真实标签的候选选择策略、概率校准和坏样本可靠性判断。生理/EEG 仍阻塞。

## 推荐优先查看

1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06e_multicandidate_oracle_gap_v0_1/tables/multicandidate_oracle_gap_table.csv`
2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06e_multicandidate_oracle_gap_v0_1/tables/multicandidate_oracle_winner_summary.csv`
3. `{str(rmse_fig).replace(chr(92), "/")}`
4. `{str(winner_fig).replace(chr(92), "/")}`
"""
    tech = f"""# Stage 6e：多候选 oracle gap 复核 v0.1

生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}

说明：本轮不训练新模型，不使用生理、EEG、连续风格或被试 ID。oracle 结果只作为研究上限，不可部署。

## Gate

{md_table(gate, ['gate', 'status', 'evidence', 'decision'], max_rows=10)}

## Test gap table

{md_table(gap[['model_name', 'role', 'rmse_steer', 'delta_vs_rbf_rmse', 'wrong_side_rate', 'large_response_recall', 'difficult_top20_rmse']], ['model_name', 'role', 'rmse_steer', 'delta_vs_rbf_rmse', 'wrong_side_rate', 'large_response_recall', 'difficult_top20_rmse'], max_rows=20)}

## Broad oracle winner summary

{md_table(winner_counts[(winner_counts['split'].eq('test')) & (winner_counts['pool_name'].eq('oracle_broad_vehicle_pool'))], ['oracle_winner_model', 'n_wins', 'mean_oracle_gain_vs_rbf'], max_rows=20)}
"""
    user_path = REPORT_DIR / "stage06e_multicandidate_oracle_gap_user_summary_cn.md"
    tech_path = REPORT_DIR / "stage06e_multicandidate_oracle_gap_v0_1_cn.md"
    user_path.write_text(user, encoding="utf-8")
    tech_path.write_text(tech, encoding="utf-8")
    return user_path, tech_path


def main() -> None:
    ensure_dirs()
    rows, source_manifest = load_candidate_rows()
    availability = model_availability(rows)
    metrics, winners, base_rows = build_summary(rows)
    winner_counts = winner_summary(winners, base_rows)
    sample_detail = sample_gain_detail(winners, base_rows)
    gap = oracle_gap_table(metrics)
    gap = merge_stage6d_metrics(gap)
    gate = build_gate(gap)
    rmse_fig, winner_fig = plot_oracle_gap(gap, winners)
    user_path, tech_path = write_reports(gap, gate, winner_counts, rmse_fig, winner_fig)

    source_manifest.to_csv(TABLE_DIR / "multicandidate_source_manifest.csv", index=False, encoding="utf-8-sig")
    availability.to_csv(TABLE_DIR / "multicandidate_model_availability.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(TABLE_DIR / "multicandidate_all_metrics.csv", index=False, encoding="utf-8-sig")
    gap.to_csv(TABLE_DIR / "multicandidate_oracle_gap_table.csv", index=False, encoding="utf-8-sig")
    winners.to_csv(TABLE_DIR / "multicandidate_oracle_winner_detail.csv", index=False, encoding="utf-8-sig")
    winner_counts.to_csv(TABLE_DIR / "multicandidate_oracle_winner_summary.csv", index=False, encoding="utf-8-sig")
    sample_detail.to_csv(TABLE_DIR / "multicandidate_oracle_gain_sample_detail.csv", index=False, encoding="utf-8-sig")
    gate.to_csv(TABLE_DIR / "multicandidate_oracle_gap_gate_table.csv", index=False, encoding="utf-8-sig")

    rbf = gap[gap["model_name"].eq(RBF_MODEL)].iloc[0]
    broad = gap[gap["model_name"].eq("oracle_broad_vehicle_pool")].iloc[0]
    selectors = gap[gap["role"].eq("deployable_selector_attempt")].copy()
    best_selector = selectors.sort_values("rmse_steer").iloc[0]
    summary = {
        "output_version": "stage06e_multicandidate_oracle_gap_v0_1",
        "run_time_local": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "rbf_test_rmse": float(rbf["rmse_steer"]),
        "broad_oracle_test_rmse": float(broad["rmse_steer"]),
        "broad_oracle_delta_vs_rbf_rmse": float(broad["delta_vs_rbf_rmse"]),
        "best_deployable_selector": best_selector["model_name"],
        "best_deployable_selector_test_rmse": float(best_selector["rmse_steer"]),
        "best_deployable_selector_delta_vs_rbf_rmse": float(best_selector["delta_vs_rbf_rmse"]),
        "gate_status": "oracle_signal_but_deployable_selection_blocked",
        "server_used": False,
        "server_credential_file_read": False,
        "uses_physio": False,
        "uses_eeg": False,
        "uses_continuous_style": False,
        "uses_subject_id": False,
        "raw_files_modified": False,
        "user_summary_path": str(user_path).replace("\\", "/"),
        "technical_report_path": str(tech_path).replace("\\", "/"),
        "gap_table_path": str(TABLE_DIR / "multicandidate_oracle_gap_table.csv").replace("\\", "/"),
    }
    (LOG_DIR / "multicandidate_oracle_gap_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
