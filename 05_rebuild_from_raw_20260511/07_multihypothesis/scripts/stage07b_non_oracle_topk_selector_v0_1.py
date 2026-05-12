from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path("F:/data_set_process/data_process/05_rebuild_from_raw_20260511")
BASELINE_SCRIPT_DIR = ROOT / "03_baselines" / "scripts"
if str(BASELINE_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_SCRIPT_DIR))

import evaluate_stage3_vehicle_baselines as eval_utils  # noqa: E402


FEATURE_TABLE = ROOT / "03_baselines" / "stage03_vehicle_instability_topk_reliability_selector_v0_1" / "tables" / "topk_reliability_selector_feature_table.csv"
PER_SAMPLE = ROOT / "03_baselines" / "stage03_vehicle_instability_topk_reliability_selector_v0_1" / "tables" / "topk_reliability_selector_per_sample_metrics.csv"
OUT_ROOT = ROOT / "07_multihypothesis" / "stage07b_non_oracle_topk_selector_v0_1"
TABLE_DIR = OUT_ROOT / "tables"
FIG_DIR = OUT_ROOT / "figures"
LOG_DIR = OUT_ROOT / "logs"
REPORT_DIR = ROOT / "09_reports"

TRACK_ID = "B_response3s_strict_core"
RBF_MODEL = "rbf_kernel_ridge_context_no_subject"
CHOICE_TO_MODEL = {
    "rbf": RBF_MODEL,
    "branch0": "topk_vehicle_transformer_branch0_no_subject",
    "branch1": "topk_vehicle_transformer_branch1_no_subject",
    "branch2": "topk_vehicle_transformer_branch2_no_subject",
}
LABEL_ORDER = ["rbf", "branch0", "branch1", "branch2"]

FORBIDDEN_INPUT_SUBSTRINGS = [
    "sample_rmse",
    "wrong_side",
    "large_response_recalled",
    "severe_amp_under",
    "peak_amp_abs_error",
    "peak_time_abs_error",
    "onset_delay_abs_error",
    "tail_abs_error",
    "tail_drift",
    "zero_crossing",
    "reversal_count_exact",
    "gt_peak_abs",
    "is_large_response",
    "is_difficult",
    "best_candidate",
    "best_branch",
    "matches_best",
    "worse_than_rbf",
    "minus_rbf",
    "gain_over_rbf",
]

BLOCKED_COLUMNS = {"sample_id", "event_uid", "subject", "session_stamp", "split"}
ALLOWED_EXPLICIT = {
    "anchor_time_rel_s",
    "curvature_anchor",
    "input_valid_ratio",
    "median_speed_kmh_window",
    "event_type",
    "event_level",
    "road_type_anchor",
    "old_v400_road_type_mode",
    "old_v400_phase_mode",
    "road_design_module_name",
    "road_design_instance_name",
    "road_design_risk_class",
    "road_design_mapping_reliability",
    "top1_branch",
    "top1_prob",
    "prob_margin",
    "branch_spread_mean",
    "branch_spread_peak",
    "branch0_prob",
    "branch1_prob",
    "branch2_prob",
    "prob_entropy",
    "rbf_pred_peak_abs",
    "rbf_pred_reversal_count",
    "rbf_pred_multi_segment",
    "top1_pred_peak_abs",
    "top1_pred_reversal_count",
    "top1_pred_multi_segment",
    "branch0_pred_peak_abs",
    "branch0_pred_reversal_count",
    "branch0_pred_multi_segment",
    "branch1_pred_peak_abs",
    "branch1_pred_reversal_count",
    "branch1_pred_multi_segment",
    "branch2_pred_peak_abs",
    "branch2_pred_reversal_count",
    "branch2_pred_multi_segment",
}


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIG_DIR, LOG_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    features = pd.read_csv(FEATURE_TABLE)
    rows = pd.read_csv(PER_SAMPLE)
    rows = rows[rows["model_name"].isin(CHOICE_TO_MODEL.values())].copy()
    if features.empty or rows.empty:
        raise RuntimeError("Missing Stage 7b inputs.")
    return features, rows


def select_allowed_features(features: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    feature_rows = []
    allowed = []
    for col in features.columns:
        reason = ""
        status = "excluded"
        if col in BLOCKED_COLUMNS:
            reason = "identifier_or_split_not_model_input"
        elif col == "best_candidate_oracle":
            reason = "training_label_only_not_input"
        elif any(token in col for token in FORBIDDEN_INPUT_SUBSTRINGS):
            reason = "label_derived_forbidden_input"
        elif col in ALLOWED_EXPLICIT:
            status = "allowed"
            reason = "pre-event_context_or_candidate_prediction_only"
            allowed.append(col)
        else:
            reason = "not_in_allowlist"
        feature_rows.append({"feature": col, "input_status": status, "reason": reason})
    return allowed, pd.DataFrame(feature_rows)


def preprocess_for_features(features: pd.DataFrame, allowed_features: list[str]) -> tuple[ColumnTransformer, list[str], list[str]]:
    use = features[allowed_features]
    categorical = [c for c in use.columns if use[c].dtype == object]
    numeric = [c for c in use.columns if c not in categorical]
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ],
        remainder="drop",
    )
    return pre, numeric, categorical


def model_candidates(pre: ColumnTransformer) -> dict[str, Pipeline]:
    return {
        "logreg_balanced_c0_2": Pipeline(
            [
                ("pre", pre),
                (
                    "clf",
                    LogisticRegression(
                        C=0.2,
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "logreg_balanced_c1": Pipeline(
            [
                ("pre", pre),
                (
                    "clf",
                    LogisticRegression(
                        C=1.0,
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "rf_shallow_balanced": Pipeline(
            [
                ("pre", pre),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=3,
                        min_samples_leaf=8,
                        class_weight="balanced_subsample",
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def choose_rows(rows: pd.DataFrame, decisions: pd.DataFrame, policy_name: str) -> pd.DataFrame:
    mapped = decisions[["split", "sample_id", "selected_choice"]].copy()
    mapped["selected_model"] = mapped["selected_choice"].map(CHOICE_TO_MODEL)
    pair = rows.merge(mapped[["split", "sample_id", "selected_model"]], on=["split", "sample_id"], how="inner")
    out = pair[pair["model_name"].eq(pair["selected_model"])].copy()
    out["model_name"] = policy_name
    return out.drop(columns=["selected_model"])


def aggregate_policy(rows: pd.DataFrame, decisions: pd.DataFrame, policy_name: str) -> pd.DataFrame:
    selected = choose_rows(rows, decisions, policy_name)
    metrics = eval_utils.aggregate_metrics(selected)
    metrics["track_id"] = TRACK_ID
    metrics["policy_name"] = policy_name
    return metrics


def reference_decisions(features: pd.DataFrame) -> pd.DataFrame:
    out = []
    base_cols = ["split", "sample_id", "best_candidate_oracle", "top1_branch"]
    sub = features[base_cols].copy()
    rbf = sub[["split", "sample_id", "best_candidate_oracle"]].copy()
    rbf["selected_choice"] = "rbf"
    rbf["policy_name"] = "always_rbf_reference"
    out.append(rbf)

    top1 = sub[["split", "sample_id", "best_candidate_oracle", "top1_branch"]].copy()
    top1["selected_choice"] = "branch" + top1["top1_branch"].astype(int).astype(str)
    top1["policy_name"] = "topk_top1_non_oracle"
    out.append(top1.drop(columns=["top1_branch"]))

    oracle = sub[["split", "sample_id", "best_candidate_oracle"]].copy()
    oracle["selected_choice"] = oracle["best_candidate_oracle"]
    oracle["policy_name"] = "oracle_best_of_rbf_topk_not_deployable"
    out.append(oracle)
    return pd.concat(out, ignore_index=True, sort=False)


def predict_with_confidence(model: Pipeline, x: pd.DataFrame, sample_meta: pd.DataFrame, policy_name: str) -> pd.DataFrame:
    pred = model.predict(x)
    prob = model.predict_proba(x)
    classes = list(model.named_steps["clf"].classes_)
    max_prob = prob.max(axis=1)
    out = sample_meta[["split", "sample_id", "best_candidate_oracle"]].copy()
    out["selected_choice"] = pred
    out["selected_confidence"] = max_prob
    out["policy_name"] = policy_name
    for i, cls in enumerate(classes):
        out[f"prob_{cls}"] = prob[:, i]
    return out


def apply_confidence_fallback(decisions: pd.DataFrame, threshold: float, policy_name: str) -> pd.DataFrame:
    out = decisions.copy()
    out["selected_choice"] = np.where(out["selected_confidence"].astype(float) >= threshold, out["selected_choice"], "rbf")
    out["policy_name"] = policy_name
    out["fallback_threshold"] = threshold
    return out


def brier_multiclass(decisions: pd.DataFrame) -> float:
    prob_cols = [c for c in decisions.columns if c.startswith("prob_")]
    if not prob_cols:
        return np.nan
    y = decisions["best_candidate_oracle"].astype(str).to_numpy()
    score = 0.0
    for col in prob_cols:
        cls = col.replace("prob_", "")
        score += ((y == cls).astype(float) - decisions[col].astype(float).to_numpy()) ** 2
    return float(np.mean(score))


def confidence_ece(decisions: pd.DataFrame, n_bins: int = 5) -> float:
    if "selected_confidence" not in decisions.columns:
        return np.nan
    d = decisions.copy()
    d["correct"] = d["selected_choice"].eq(d["best_candidate_oracle"]).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        sub = d[(d["selected_confidence"] >= lo) & (d["selected_confidence"] < hi if hi < 1.0 else d["selected_confidence"] <= hi)]
        if sub.empty:
            continue
        ece += len(sub) / len(d) * abs(float(sub["correct"].mean()) - float(sub["selected_confidence"].mean()))
    return float(ece)


def decision_diagnostics(decisions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (policy, split), sub in decisions.groupby(["policy_name", "split"]):
        rows.append(
            {
                "policy_name": policy,
                "split": split,
                "n_samples": int(len(sub)),
                "oracle_choice_accuracy": float(accuracy_score(sub["best_candidate_oracle"], sub["selected_choice"])),
                "mean_confidence": float(sub["selected_confidence"].mean()) if "selected_confidence" in sub else np.nan,
                "brier_multiclass": brier_multiclass(sub),
                "ece_5bin": confidence_ece(sub),
                "rbf_selected_rate": float(sub["selected_choice"].eq("rbf").mean()),
                "branch0_selected_rate": float(sub["selected_choice"].eq("branch0").mean()),
                "branch1_selected_rate": float(sub["selected_choice"].eq("branch1").mean()),
                "branch2_selected_rate": float(sub["selected_choice"].eq("branch2").mean()),
            }
        )
    return pd.DataFrame(rows)


def run_selectors(features: pd.DataFrame, rows: pd.DataFrame, allowed_features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pre, numeric, categorical = preprocess_for_features(features, allowed_features)
    models = model_candidates(pre)
    train = features["split"].eq("train")
    val = features["split"].eq("val")
    test = features["split"].eq("test")
    x_train = features.loc[train, allowed_features]
    y_train = features.loc[train, "best_candidate_oracle"]

    reference = reference_decisions(features)
    all_decisions = [reference]
    model_rows = []
    for name, model in models.items():
        model.fit(x_train, y_train)
        base_dec = predict_with_confidence(model, features[allowed_features], features, name)
        all_decisions.append(base_dec)
        for threshold in [0.0, 0.35, 0.5, 0.65, 0.8]:
            policy_name = f"{name}__fallback_rbf_conf_lt_{threshold:.2f}"
            all_decisions.append(apply_confidence_fallback(base_dec, threshold, policy_name))
        val_pred = base_dec[val]
        test_pred = base_dec[test]
        model_rows.append(
            {
                "candidate_model": name,
                "numeric_feature_count": len(numeric),
                "categorical_feature_count": len(categorical),
                "val_oracle_choice_accuracy": float(accuracy_score(features.loc[val, "best_candidate_oracle"], val_pred["selected_choice"])),
                "test_oracle_choice_accuracy": float(accuracy_score(features.loc[test, "best_candidate_oracle"], test_pred["selected_choice"])),
                "val_brier": brier_multiclass(val_pred),
                "test_brier": brier_multiclass(test_pred),
                "val_ece_5bin": confidence_ece(val_pred),
                "test_ece_5bin": confidence_ece(test_pred),
            }
        )
    decisions = pd.concat(all_decisions, ignore_index=True, sort=False)

    metric_frames = []
    for policy, sub in decisions.groupby("policy_name"):
        metric_frames.append(aggregate_policy(rows, sub, policy))
    metrics = pd.concat(metric_frames, ignore_index=True, sort=False)

    val_metrics = metrics[metrics["split"].eq("val")].copy()
    deployable = val_metrics[
        (~val_metrics["policy_name"].str.contains("oracle", regex=False))
        & (~val_metrics["policy_name"].eq("always_rbf_reference"))
    ].copy()
    best_policy = deployable.sort_values(["rmse_steer", "wrong_side_rate", "large_response_recall"], ascending=[True, True, False]).iloc[0]
    selected_policy = str(best_policy["policy_name"])
    selected_metrics = metrics[metrics["policy_name"].isin(["always_rbf_reference", "topk_top1_non_oracle", "oracle_best_of_rbf_topk_not_deployable", selected_policy])].copy()
    selected_metrics["selected_for_report"] = selected_metrics["policy_name"].eq(selected_policy)

    diagnostics = decision_diagnostics(decisions)
    model_info = pd.DataFrame(model_rows)
    return decisions, metrics, selected_metrics, pd.concat([model_info], ignore_index=True)


def coverage_risk(decisions: pd.DataFrame, rows: pd.DataFrame, selected_policy: str) -> pd.DataFrame:
    d = decisions[decisions["policy_name"].eq(selected_policy) & decisions["split"].isin(["val", "test"])].copy()
    selected_rows = choose_rows(rows, d, selected_policy)
    joined = d[["split", "sample_id", "selected_confidence"]].merge(
        selected_rows[["split", "sample_id", "sample_rmse", "wrong_side", "large_response_recalled"]],
        on=["split", "sample_id"],
        how="left",
    )
    out = []
    for split, sub in joined.groupby("split"):
        sub = sub.sort_values("selected_confidence", ascending=False).reset_index(drop=True)
        for coverage in [0.5, 0.75, 1.0]:
            n = max(1, int(np.ceil(len(sub) * coverage)))
            keep = sub.head(n)
            out.append(
                {
                    "policy_name": selected_policy,
                    "split": split,
                    "coverage": coverage,
                    "n_kept": int(n),
                    "mean_sample_rmse": float(keep["sample_rmse"].mean()),
                    "wrong_side_rate": float(keep["wrong_side"].mean()),
                    "large_response_recall_mean": float(keep["large_response_recalled"].mean()),
                    "mean_confidence": float(keep["selected_confidence"].mean()),
                }
            )
    return pd.DataFrame(out)


def build_gate(selected_metrics: pd.DataFrame) -> pd.DataFrame:
    test = selected_metrics[selected_metrics["split"].eq("test")].copy()
    rbf = test[test["policy_name"].eq("always_rbf_reference")].iloc[0]
    selected = test[test["selected_for_report"].eq(True)].iloc[0]
    oracle = test[test["policy_name"].eq("oracle_best_of_rbf_topk_not_deployable")].iloc[0]
    rmse_delta = float(selected["rmse_steer"]) - float(rbf["rmse_steer"])
    physical_gain = (
        float(selected["wrong_side_rate"]) < float(rbf["wrong_side_rate"])
        or float(selected["large_response_recall"]) > float(rbf["large_response_recall"])
        or float(selected["difficult_top20_rmse"]) < float(rbf["difficult_top20_rmse"])
    )
    promote = rmse_delta <= 0.0 and physical_gain
    rows = [
        {
            "gate": "stage07b_deployable_selector_upgrade",
            "status": "pass" if promote else "no_upgrade",
            "evidence": f"selected test RMSE delta={rmse_delta:+.6f}; wrong-side={selected['wrong_side_rate']:.3f} vs RBF {rbf['wrong_side_rate']:.3f}; large recall={selected['large_response_recall']:.3f} vs RBF {rbf['large_response_recall']:.3f}.",
            "decision": "可升级为 Stage 7 候选。" if promote else "当前轻量 selector 不升级主线。",
        },
        {
            "gate": "oracle_gap_remaining",
            "status": "still_large",
            "evidence": f"oracle RMSE={oracle['rmse_steer']:.6f}; selected RMSE={selected['rmse_steer']:.6f}.",
            "decision": "仍需改进非 oracle 选择策略或候选表示。",
        },
        {
            "gate": "stage05_physio_eeg_allowed",
            "status": "blocked",
            "evidence": "车辆-only Stage 7b 轻量 selector 未形成可升级结果。",
            "decision": "继续阻塞生理/EEG有效性结论。",
        },
    ]
    return pd.DataFrame(rows)


def plot_results(selected_metrics: pd.DataFrame, diagnostics: pd.DataFrame, coverage: pd.DataFrame) -> tuple[Path, Path, Path]:
    test = selected_metrics[selected_metrics["split"].eq("test")].copy().sort_values("rmse_steer", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(test["policy_name"], test["rmse_steer"], color=["#9ca3af" if "oracle" not in p else "#2563eb" for p in test["policy_name"]])
    ax.set_xlabel("test RMSE")
    ax.set_title("Stage 7b non-oracle selector vs references")
    fig.tight_layout()
    rmse_path = FIG_DIR / "stage07b_selector_test_rmse.png"
    fig.savefig(rmse_path, dpi=180)
    plt.close(fig)

    test_diag = diagnostics[diagnostics["split"].eq("test")].copy().sort_values("oracle_choice_accuracy")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(test_diag["policy_name"], test_diag["oracle_choice_accuracy"], color="#0f766e")
    ax.set_xlabel("oracle choice accuracy")
    ax.set_title("Candidate choice accuracy, diagnostic only")
    fig.tight_layout()
    acc_path = FIG_DIR / "stage07b_selector_choice_accuracy.png"
    fig.savefig(acc_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for split, sub in coverage.groupby("split"):
        ax.plot(sub["coverage"], sub["mean_sample_rmse"], marker="o", label=split)
    ax.set_xlabel("coverage")
    ax.set_ylabel("mean sample RMSE")
    ax.set_title("Coverage-risk for selected policy")
    ax.legend()
    fig.tight_layout()
    cov_path = FIG_DIR / "stage07b_coverage_risk.png"
    fig.savefig(cov_path, dpi=180)
    plt.close(fig)
    return rmse_path, acc_path, cov_path


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


def write_reports(selected_metrics: pd.DataFrame, gate: pd.DataFrame, diagnostics: pd.DataFrame, coverage: pd.DataFrame, rmse_fig: Path, acc_fig: Path, cov_fig: Path) -> tuple[Path, Path]:
    test = selected_metrics[selected_metrics["split"].eq("test")].copy()
    rbf = test[test["policy_name"].eq("always_rbf_reference")].iloc[0]
    selected = test[test["selected_for_report"].eq(True)].iloc[0]
    selected_diag = diagnostics[(diagnostics["split"].eq("test")) & (diagnostics["policy_name"].eq(selected["policy_name"]))].iloc[0]
    user = f"""# Stage 7b 用户查看版：非 oracle top-K selector 轻量实验 v0.1

## 为什么做

Stage 7a 固定了不能用 test 标签选候选的规则。本轮用已有 top-K/RBF 特征做一个轻量 selector，检查非 oracle 选择器是否能把 Stage 6e 的 oracle 上限转成实际可部署收益。

## 目前发现

- RBF/KNN 主参照 test RMSE={rbf['rmse_steer']:.6f}。
- val 选中的非 oracle policy：`{selected['policy_name']}`，test RMSE={selected['rmse_steer']:.6f}，相对 RBF delta={selected['rmse_steer'] - rbf['rmse_steer']:+.6f}。
- 该 policy wrong-side={selected['wrong_side_rate']:.3f}，large recall={selected['large_response_recall']:.3f}，difficult RMSE={selected['difficult_top20_rmse']:.6f}。
- 该 policy 在 test 上选择 RBF 的比例为 {selected_diag['rbf_selected_rate']:.3f}；如果比例接近 1，说明当前 selector 实际只是退回主参照，没有带来新选择能力。
- 如果 gate 为 `no_upgrade`，说明当前轻量 selector 还不能升级主线。

## 可信边界

本轮 selector 输入只使用事件/道路上下文、候选概率、候选分歧和候选预测自身形态。`test_sample_rmse`、`wrong_side`、`best_candidate_oracle` 等 label-derived 字段没有进入输入。test 只用于最终评估。

## 下一步

如果当前轻量 selector 不升级，应继续改候选表示、导出完整预测轨迹差异特征，或考虑更明确的置信度 fallback；生理/EEG 继续阻塞。

## 推荐优先查看

1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07b_non_oracle_topk_selector_v0_1/tables/stage07b_selected_policy_metrics.csv`
2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07b_non_oracle_topk_selector_v0_1/tables/stage07b_gate_table.csv`
3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07b_non_oracle_topk_selector_v0_1/tables/stage07b_feature_audit.csv`
4. `{str(rmse_fig).replace(chr(92), "/")}`
5. `{str(cov_fig).replace(chr(92), "/")}`
"""
    tech = f"""# Stage 7b：非 oracle top-K selector 轻量实验 v0.1

生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}

本轮不使用生理、EEG、连续风格或被试 ID。selector 输入显式剔除 label-derived 字段。

## Gate

{md_table(gate, ['gate', 'status', 'evidence', 'decision'], max_rows=10)}

## Selected Test Metrics

{md_table(test.sort_values('rmse_steer'), ['policy_name', 'rmse_steer', 'wrong_side_rate', 'large_response_recall', 'difficult_top20_rmse', 'selected_for_report'], max_rows=20)}

## Diagnostics

{md_table(diagnostics[diagnostics['split'].eq('test')].sort_values('oracle_choice_accuracy', ascending=False), ['policy_name', 'oracle_choice_accuracy', 'mean_confidence', 'brier_multiclass', 'ece_5bin'], max_rows=20)}

## Coverage Risk

{md_table(coverage, ['policy_name', 'split', 'coverage', 'mean_sample_rmse', 'wrong_side_rate', 'mean_confidence'], max_rows=20)}
"""
    user_path = REPORT_DIR / "stage07b_non_oracle_topk_selector_user_summary_cn.md"
    tech_path = REPORT_DIR / "stage07b_non_oracle_topk_selector_v0_1_cn.md"
    user_path.write_text(user, encoding="utf-8")
    tech_path.write_text(tech, encoding="utf-8")
    return user_path, tech_path


def main() -> None:
    ensure_dirs()
    features, rows = load_inputs()
    allowed_features, feature_audit = select_allowed_features(features)
    decisions, metrics, selected_metrics, model_info = run_selectors(features, rows, allowed_features)
    selected_policy = selected_metrics[selected_metrics["selected_for_report"].eq(True)]["policy_name"].iloc[0]
    diagnostics = decision_diagnostics(decisions)
    coverage = coverage_risk(decisions, rows, selected_policy)
    gate = build_gate(selected_metrics)
    rmse_fig, acc_fig, cov_fig = plot_results(selected_metrics, diagnostics, coverage)
    user_path, tech_path = write_reports(selected_metrics, gate, diagnostics, coverage, rmse_fig, acc_fig, cov_fig)

    feature_audit.to_csv(TABLE_DIR / "stage07b_feature_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"allowed_feature": allowed_features}).to_csv(TABLE_DIR / "stage07b_allowed_features.csv", index=False, encoding="utf-8-sig")
    model_info.to_csv(TABLE_DIR / "stage07b_model_info.csv", index=False, encoding="utf-8-sig")
    decisions.to_csv(TABLE_DIR / "stage07b_selector_decisions.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(TABLE_DIR / "stage07b_all_policy_metrics.csv", index=False, encoding="utf-8-sig")
    selected_metrics.to_csv(TABLE_DIR / "stage07b_selected_policy_metrics.csv", index=False, encoding="utf-8-sig")
    diagnostics.to_csv(TABLE_DIR / "stage07b_decision_diagnostics.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(TABLE_DIR / "stage07b_coverage_risk.csv", index=False, encoding="utf-8-sig")
    gate.to_csv(TABLE_DIR / "stage07b_gate_table.csv", index=False, encoding="utf-8-sig")

    test = selected_metrics[selected_metrics["split"].eq("test")].copy()
    rbf = test[test["policy_name"].eq("always_rbf_reference")].iloc[0]
    selected = test[test["selected_for_report"].eq(True)].iloc[0]
    selected_diag = diagnostics[(diagnostics["split"].eq("test")) & (diagnostics["policy_name"].eq(selected_policy))].iloc[0]
    summary = {
        "output_version": "stage07b_non_oracle_topk_selector_v0_1",
        "run_time_local": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "allowed_feature_count": len(allowed_features),
        "selected_policy": selected_policy,
        "rbf_test_rmse": float(rbf["rmse_steer"]),
        "selected_test_rmse": float(selected["rmse_steer"]),
        "selected_delta_vs_rbf_rmse": float(selected["rmse_steer"] - rbf["rmse_steer"]),
        "selected_test_rbf_selected_rate": float(selected_diag["rbf_selected_rate"]),
        "gate_status": str(gate[gate["gate"].eq("stage07b_deployable_selector_upgrade")]["status"].iloc[0]),
        "server_used": False,
        "server_credential_file_read": False,
        "uses_physio": False,
        "uses_eeg": False,
        "uses_continuous_style": False,
        "uses_subject_id": False,
        "raw_files_modified": False,
        "test_used_for_model_selection": False,
        "user_summary_path": str(user_path).replace("\\", "/"),
        "technical_report_path": str(tech_path).replace("\\", "/"),
    }
    (LOG_DIR / "stage07b_selector_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
