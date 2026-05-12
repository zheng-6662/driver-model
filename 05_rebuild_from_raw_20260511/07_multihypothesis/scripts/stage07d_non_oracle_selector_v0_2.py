# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path("F:/data_set_process/data_process/05_rebuild_from_raw_20260511")
BASELINE_SCRIPT_DIR = ROOT / "03_baselines" / "scripts"
if str(BASELINE_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_SCRIPT_DIR))

import evaluate_stage3_vehicle_baselines as eval_utils  # noqa: E402


OUTPUT_VERSION = "stage07d_non_oracle_selector_v0_2"
TRACK_ID = "B_response3s_strict_core"
SPLIT_STRATEGY = "session_level_split"

STAGE7C_ROOT = ROOT / "07_multihypothesis" / "stage07c_candidate_trajectory_export_v0_1"
FEATURE_TABLE = STAGE7C_ROOT / "tables" / "candidate_feature_and_label_diagnosis.csv"
PER_SAMPLE = STAGE7C_ROOT / "tables" / "candidate_export_per_sample_metrics.csv"

OUT_ROOT = ROOT / "07_multihypothesis" / OUTPUT_VERSION
TABLE_DIR = OUT_ROOT / "tables"
FIG_DIR = OUT_ROOT / "figures"
LOG_DIR = OUT_ROOT / "logs"
REPORT_DIR = ROOT / "09_reports"

RBF_MODEL = "rbf_kernel_ridge_context_no_subject"
KEYPOINT_MODEL = "keypoint_residual_vehicle_transformer_no_subject"
BRANCH_MODELS = [f"topk_vehicle_transformer_branch{k}_no_subject" for k in range(3)]
TOP1_MODEL = "topk_vehicle_transformer_top1_no_subject"
BROAD_ORACLE_MODEL = "oracle_best_of_rbf_keypoint_topk_upper_bound"
RBF_TOPK_ORACLE_MODEL = "oracle_best_of_rbf_topk_upper_bound"
CANDIDATE_MODELS = [RBF_MODEL, KEYPOINT_MODEL, *BRANCH_MODELS]
TOP1_CHOICE_MAP = {0: BRANCH_MODELS[0], 1: BRANCH_MODELS[1], 2: BRANCH_MODELS[2]}

IDENTIFIER_COLUMNS = {"sample_id", "event_uid", "subject", "session_stamp", "split"}
FORBIDDEN_PREFIXES = ("label_diag__",)
FORBIDDEN_SUBSTRINGS = (
    "oracle",
    "sample_rmse",
    "wrong_side",
    "large_response",
    "severe_amp",
    "peak_amp_abs_error",
    "peak_time_abs_error",
    "onset_delay_abs_error",
    "tail_abs_error",
    "tail_drift",
    "zero_crossing",
    "reversal_count_exact",
    "gt_peak",
    "is_large_response",
    "is_difficult",
)
FALLBACK_THRESHOLDS = [0.35, 0.45, 0.55, 0.65, 0.75]


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIG_DIR, LOG_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def path_str(path: Path) -> str:
    return str(path).replace("\\", "/")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not FEATURE_TABLE.exists():
        raise FileNotFoundError(FEATURE_TABLE)
    if not PER_SAMPLE.exists():
        raise FileNotFoundError(PER_SAMPLE)
    features = pd.read_csv(FEATURE_TABLE)
    per_sample = pd.read_csv(PER_SAMPLE)
    per_sample = per_sample[per_sample["model_name"].isin([*CANDIDATE_MODELS, TOP1_MODEL, BROAD_ORACLE_MODEL, RBF_TOPK_ORACLE_MODEL])].copy()
    if features.empty or per_sample.empty:
        raise RuntimeError("Stage 7c inputs are empty.")
    return features, per_sample


def select_allowed_features(features: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    rows: list[dict[str, str]] = []
    allowed: list[str] = []
    for col in features.columns:
        status = "excluded"
        reason = "not_in_allowlist"
        low = col.lower()
        if col in IDENTIFIER_COLUMNS:
            reason = "identifier_or_split"
        elif col.startswith(FORBIDDEN_PREFIXES):
            reason = "label_derived_diagnostic"
        elif any(token in low for token in FORBIDDEN_SUBSTRINGS):
            reason = "label_or_outcome_derived"
        else:
            status = "allowed"
            reason = "pre_event_context_or_candidate_prediction_only"
            allowed.append(col)
        rows.append({"feature": col, "input_status": status, "reason": reason})
    return allowed, pd.DataFrame(rows)


def make_preprocessor(features: pd.DataFrame, allowed: list[str]) -> tuple[ColumnTransformer, list[str], list[str]]:
    use = features[allowed]
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


def make_models(pre: ColumnTransformer) -> dict[str, Pipeline]:
    return {
        "logreg_balanced_c0_2": Pipeline(
            [
                ("pre", pre),
                (
                    "clf",
                    LogisticRegression(
                        C=0.2,
                        max_iter=3000,
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
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "rf_depth3_balanced": Pipeline(
            [
                ("pre", pre),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=3,
                        min_samples_leaf=8,
                        class_weight="balanced_subsample",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "rf_depth4_balanced": Pipeline(
            [
                ("pre", pre),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=4,
                        min_samples_leaf=6,
                        class_weight="balanced_subsample",
                        random_state=43,
                    ),
                ),
            ]
        ),
    }


def choice_to_short(model_name: str) -> str:
    if model_name == RBF_MODEL:
        return "rbf"
    if model_name == KEYPOINT_MODEL:
        return "keypoint"
    if model_name in BRANCH_MODELS:
        return "branch" + str(BRANCH_MODELS.index(model_name))
    return model_name


def add_reference_decisions(features: pd.DataFrame) -> list[pd.DataFrame]:
    base = features[["split", "sample_id", "label_diag__broad_oracle_model", "top1_branch"]].copy()
    out: list[pd.DataFrame] = []

    rbf = base[["split", "sample_id", "label_diag__broad_oracle_model"]].copy()
    rbf["selected_model"] = RBF_MODEL
    rbf["policy_name"] = "always_rbf_reference"
    out.append(rbf)

    top1 = base[["split", "sample_id", "label_diag__broad_oracle_model", "top1_branch"]].copy()
    top1["selected_model"] = top1["top1_branch"].astype(int).map(TOP1_CHOICE_MAP)
    top1["policy_name"] = "topk_top1_non_oracle"
    out.append(top1.drop(columns=["top1_branch"]))

    oracle = base[["split", "sample_id", "label_diag__broad_oracle_model"]].copy()
    oracle["selected_model"] = oracle["label_diag__broad_oracle_model"].astype(str)
    oracle["policy_name"] = "broad_oracle_upper_bound"
    out.append(oracle)
    return out


def predict_policy_decisions(
    model_name: str,
    model: Pipeline,
    features: pd.DataFrame,
    allowed: list[str],
    train_mask: np.ndarray,
) -> list[pd.DataFrame]:
    x = features[allowed]
    y = features["label_diag__broad_oracle_model"].astype(str)
    model.fit(x.loc[train_mask], y.loc[train_mask])
    pred = model.predict(x)
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(x)
        confidence = np.max(prob, axis=1)
    else:
        confidence = np.ones(len(features), dtype=np.float32)
    rows: list[pd.DataFrame] = []
    base = features[["split", "sample_id", "label_diag__broad_oracle_model"]].copy()
    direct = base.copy()
    direct["selected_model"] = pred
    direct["selector_confidence"] = confidence
    direct["policy_name"] = model_name
    rows.append(direct)
    for thr in FALLBACK_THRESHOLDS:
        fb = direct.copy()
        fb["selected_model"] = np.where(confidence >= thr, fb["selected_model"], RBF_MODEL)
        fb["policy_name"] = f"{model_name}__fallback_rbf_conf_lt_{thr:.2f}"
        rows.append(fb)
    return rows


def choose_rows(per_sample: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    pair = per_sample.merge(decisions[["split", "sample_id", "selected_model"]], on=["split", "sample_id"], how="inner")
    chosen = pair[pair["model_name"].astype(str).eq(pair["selected_model"].astype(str))].copy()
    chosen["model_name"] = str(decisions["policy_name"].iloc[0])
    return chosen.drop(columns=["selected_model"])


def aggregate_policy(per_sample: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    chosen = choose_rows(per_sample, decisions)
    metrics = eval_utils.aggregate_metrics(chosen)
    metrics["track_id"] = TRACK_ID
    metrics["policy_name"] = str(decisions["policy_name"].iloc[0])
    return metrics


def decision_diagnostics(decisions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (policy, split), grp in decisions.groupby(["policy_name", "split"], dropna=False):
        oracle = grp["label_diag__broad_oracle_model"].astype(str)
        selected = grp["selected_model"].astype(str)
        rows.append(
            {
                "policy_name": policy,
                "split": split,
                "n_samples": int(len(grp)),
                "oracle_match_rate": float((oracle == selected).mean()),
                "rbf_selected_rate": float((selected == RBF_MODEL).mean()),
                "keypoint_selected_rate": float((selected == KEYPOINT_MODEL).mean()),
                "branch_selected_rate": float(selected.isin(BRANCH_MODELS).mean()),
                "unique_selected_models": int(selected.nunique()),
            }
        )
    return pd.DataFrame(rows)


def add_reference_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    out = metrics.copy()
    refs = out[out["model_name"] == "always_rbf_reference"][["split", "rmse_steer", "wrong_side_rate", "large_response_recall", "difficult_top20_rmse"]]
    refs = refs.rename(
        columns={
            "rmse_steer": "rbf_ref_rmse",
            "wrong_side_rate": "rbf_ref_wrong_side",
            "large_response_recall": "rbf_ref_large_recall",
            "difficult_top20_rmse": "rbf_ref_difficult_rmse",
        }
    )
    out = out.merge(refs, on="split", how="left")
    out["rmse_delta_vs_rbf"] = out["rmse_steer"] - out["rbf_ref_rmse"]
    out["wrong_side_delta_vs_rbf"] = out["wrong_side_rate"] - out["rbf_ref_wrong_side"]
    out["large_recall_delta_vs_rbf"] = out["large_response_recall"] - out["rbf_ref_large_recall"]
    out["difficult_rmse_delta_vs_rbf"] = out["difficult_top20_rmse"] - out["rbf_ref_difficult_rmse"]
    return out


def select_policy(metrics: pd.DataFrame, diag: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    val = metrics[metrics["split"] == "val"].copy()
    val = val[~val["model_name"].eq("broad_oracle_upper_bound")].copy()
    rbf = val[val["model_name"].eq("always_rbf_reference")].iloc[0]
    candidates = val[~val["model_name"].isin(["always_rbf_reference", "topk_top1_non_oracle"])].copy()
    candidates["meets_rmse_improvement"] = candidates["rmse_steer"] < float(rbf["rmse_steer"]) - 1e-6
    candidates["meets_noninferior_physical"] = (
        (candidates["rmse_steer"] <= float(rbf["rmse_steer"]) + 0.002)
        & (
            (candidates["wrong_side_rate"] < float(rbf["wrong_side_rate"]))
            | (candidates["large_response_recall"] > float(rbf["large_response_recall"]))
            | (candidates["difficult_top20_rmse"] < float(rbf["difficult_top20_rmse"]))
        )
    )
    if candidates["meets_rmse_improvement"].any():
        selected = candidates[candidates["meets_rmse_improvement"]].sort_values(["rmse_steer", "wrong_side_rate"]).iloc[0]["model_name"]
        reason = "val_rmse_improves_rbf"
    elif candidates["meets_noninferior_physical"].any():
        selected = candidates[candidates["meets_noninferior_physical"]].sort_values(["rmse_steer", "wrong_side_rate"]).iloc[0]["model_name"]
        reason = "val_noninferior_with_physical_gain"
    else:
        selected = "always_rbf_reference"
        reason = "no_candidate_passed_val_gate"
    table = candidates.sort_values(["rmse_steer", "wrong_side_rate"]).copy()
    table["selected_by_val_gate"] = table["model_name"].eq(selected).astype(int)
    table["selection_reason"] = reason
    merged = table.merge(diag[diag["split"] == "val"][["policy_name", "rbf_selected_rate", "oracle_match_rate"]], left_on="model_name", right_on="policy_name", how="left")
    return str(selected), merged.drop(columns=["policy_name"], errors="ignore")


def confusion_table(decisions: pd.DataFrame, policy_name: str) -> pd.DataFrame:
    sub = decisions[decisions["policy_name"] == policy_name].copy()
    rows: list[pd.DataFrame] = []
    for split, grp in sub.groupby("split"):
        tab = pd.crosstab(grp["label_diag__broad_oracle_model"], grp["selected_model"])
        tab = tab.reset_index().rename(columns={"label_diag__broad_oracle_model": "oracle_best_model"})
        tab.insert(0, "split", split)
        rows.append(tab)
    return pd.concat(rows, ignore_index=True, sort=False).fillna(0)


def plot_policy_metrics(metrics: pd.DataFrame, selected: str, path: Path) -> None:
    test = metrics[metrics["split"] == "test"].copy()
    keep = ["always_rbf_reference", "topk_top1_non_oracle", selected, "broad_oracle_upper_bound"]
    keep_unique = []
    for item in keep:
        if item not in keep_unique:
            keep_unique.append(item)
    test = test[test["model_name"].isin(keep_unique)].copy()
    test["order"] = test["model_name"].map({name: i for i, name in enumerate(keep_unique)})
    test = test.sort_values("order")
    labels = [x.replace("__fallback_rbf_conf_lt_", "\nfb<").replace("_", " ") for x in test["model_name"]]
    fig, axes = plt.subplots(1, 4, figsize=(16.0, 4.2))
    for ax, col, title in [
        (axes[0], "rmse_steer", "RMSE"),
        (axes[1], "wrong_side_rate", "Wrong-side"),
        (axes[2], "large_response_recall", "Large recall"),
        (axes[3], "difficult_top20_rmse", "Difficult RMSE"),
    ]:
        colors = ["#4c78a8" if name != selected else "#f58518" for name in test["model_name"]]
        ax.bar(np.arange(len(test)), test[col], color=colors)
        ax.set_title(title)
        ax.set_xticks(np.arange(len(test)), labels, rotation=35, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Stage 7d selected non-oracle policy on test", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_val_delta(metrics: pd.DataFrame, path: Path) -> None:
    val = metrics[(metrics["split"] == "val") & (~metrics["model_name"].eq("broad_oracle_upper_bound"))].copy()
    val = val.sort_values("rmse_delta_vs_rbf")
    fig, ax = plt.subplots(figsize=(9.0, max(4.0, 0.22 * len(val))))
    colors = np.where(val["rmse_delta_vs_rbf"] <= 0, "#54a24b", "#e45756")
    ax.barh(np.arange(len(val)), val["rmse_delta_vs_rbf"], color=colors)
    ax.axvline(0.0, color="#444444", linewidth=0.9)
    ax.set_yticks(np.arange(len(val)), val["model_name"], fontsize=7)
    ax.set_xlabel("Validation RMSE delta vs RBF/KNN")
    ax.set_title("Stage 7d val policy ranking")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_choice_counts(decisions: pd.DataFrame, selected: str, path: Path) -> None:
    sub = decisions[(decisions["policy_name"] == selected) & (decisions["split"].isin(["train", "val", "test"]))].copy()
    count = sub.groupby(["split", "selected_model"]).size().reset_index(name="n")
    order = ["train", "val", "test"]
    models = [m for m in CANDIDATE_MODELS if m in set(count["selected_model"])]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    bottom = np.zeros(len(order))
    colors = {
        RBF_MODEL: "#4c78a8",
        KEYPOINT_MODEL: "#b279a2",
        BRANCH_MODELS[0]: "#e45756",
        BRANCH_MODELS[1]: "#f58518",
        BRANCH_MODELS[2]: "#54a24b",
    }
    for model in models:
        vals = []
        for split in order:
            part = count[(count["split"] == split) & (count["selected_model"] == model)]
            vals.append(int(part["n"].iloc[0]) if not part.empty else 0)
        ax.bar(order, vals, bottom=bottom, color=colors.get(model, "#999999"), label=choice_to_short(model))
        bottom += np.array(vals)
    ax.set_ylabel("Selected sample count")
    ax.set_title(f"Choice counts for {selected}")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_reports(
    metrics: pd.DataFrame,
    selected: str,
    selected_gate: pd.DataFrame,
    diag: pd.DataFrame,
    figures: dict[str, str],
) -> None:
    test = metrics[metrics["split"] == "test"].set_index("model_name")
    val = metrics[metrics["split"] == "val"].set_index("model_name")

    def safe(frame: pd.DataFrame, model: str, col: str) -> float:
        if model not in frame.index or col not in frame.columns:
            return float("nan")
        return float(frame.loc[model, col])

    selected_test_rmse = safe(test, selected, "rmse_steer")
    rbf_test_rmse = safe(test, "always_rbf_reference", "rmse_steer")
    selected_test_rbf_rate = float(
        diag[(diag["policy_name"] == selected) & (diag["split"] == "test")]["rbf_selected_rate"].iloc[0]
        if not diag[(diag["policy_name"] == selected) & (diag["split"] == "test")].empty
        else np.nan
    )
    val_table_text = selected_gate[["model_name", "rmse_steer", "rmse_delta_vs_rbf", "wrong_side_rate", "large_response_recall", "rbf_selected_rate", "oracle_match_rate", "selected_by_val_gate"]].head(12).to_string(index=False)
    test_table_text = (
        metrics[
            (metrics["split"] == "test")
            & metrics["model_name"].isin(["always_rbf_reference", "topk_top1_non_oracle", selected, "broad_oracle_upper_bound"])
        ][["model_name", "rmse_steer", "rmse_delta_vs_rbf", "wrong_side_rate", "large_response_recall", "difficult_top20_rmse"]]
        .drop_duplicates()
        .to_string(index=False)
    )

    gate_status = "upgrade" if selected != "always_rbf_reference" and selected_test_rmse < rbf_test_rmse - 1e-6 else "no_upgrade"
    if selected == "always_rbf_reference":
        reason = "val gate 没有发现比 RBF/KNN 更可靠的非 oracle 策略。"
    elif selected_test_rmse >= rbf_test_rmse:
        reason = "val gate 选出了候选策略，但 test 没有超过 RBF/KNN。"
    else:
        reason = "test 上超过 RBF/KNN，但仍需多 seed/稳定性复核。"

    user = f"""# Stage 7d 用户查看版：非 oracle 候选选择器 v0.2

## 这个阶段为什么做

Stage 7c 已经证明候选池有 oracle 上限，但 oracle 不能部署。这个阶段专门检查：只用事件前信息和候选预测本身的特征，能不能在不看 test 标签的情况下，学会什么时候不要用 RBF/KNN。

## 这个阶段检查了什么

- 候选：RBF/KNN、keypoint residual、top-K branch0/1/2。
- 输入特征：道路/事件上下文、top-K 概率、候选轨迹自身的峰值/方向/反向修正/分散度等。
- 禁止输入：sample RMSE、真实标签、oracle winner、错侧率、困难标签、subject ID、session ID、生理、脑电、连续风格。
- 训练/选择：train 训练 classifier，val 选择策略，test 只报告。

## 目前发现了什么

- val 选择策略：`{selected}`。
- val 上该策略 RMSE={safe(val, selected, 'rmse_steer'):.6f}，RBF/KNN val RMSE={safe(val, 'always_rbf_reference', 'rmse_steer'):.6f}。
- test 上该策略 RMSE={selected_test_rmse:.6f}，RBF/KNN test RMSE={rbf_test_rmse:.6f}，delta={selected_test_rmse - rbf_test_rmse:+.6f}。
- test 上该策略选择 RBF/KNN 的比例={selected_test_rbf_rate:.3f}。
- gate={gate_status}。{reason}

## val 选择表

```text
{val_table_text}
```

## test 对照表

```text
{test_table_text}
```

## 哪些结果可信

可信的是：这个选择器没有使用 test 标签做选择，也没有使用 subject ID、生理、脑电或连续风格。它回答的是“候选池的 oracle 上限能否被当前非 oracle 特征转化为可部署收益”。

## 哪些结果还不能下结论

如果 gate 仍然是 no_upgrade，就不能说多假设路线已经超过 RBF/KNN；也不能据此进入生理/EEG 有效性结论。oracle 上限仍然只能说明潜力，不是部署性能。

## 下一阶段是否可以继续

可以继续，但如果本轮仍没有超过 RBF/KNN，下一步应转向候选生成方式本身：让候选显式覆盖方向、幅值、峰值时间、尾段和多段修正，而不是继续只堆选择器。

## 推荐优先查看

1. `{figures["policy_metrics"]}`
2. `{figures["val_delta"]}`
3. `{figures["choice_counts"]}`
4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07d_non_oracle_selector_v0_2/tables/stage07d_gate_table.csv`
5. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07d_non_oracle_selector_v0_2/tables/stage07d_selected_policy_decisions.csv`
"""
    (REPORT_DIR / "stage07d_non_oracle_selector_v0_2_user_summary_cn.md").write_text(user, encoding="utf-8")

    tech = f"""# Stage 7d 技术报告：non-oracle selector v0.2

## Scope

- Input table: `{path_str(FEATURE_TABLE)}`
- Per-sample metrics: `{path_str(PER_SAMPLE)}`
- Candidate pool: `{', '.join(CANDIDATE_MODELS)}`
- Target for train/val diagnostics: broad oracle winner from Stage 7c.
- No server used. Credential file not read.
- Excluded inputs: label diagnostics, subject/session identifiers, physio, EEG, continuous style.

## Selected Policy

- selected_policy=`{selected}`
- gate=`{gate_status}`
- reason={reason}

## Validation Candidate Table

```text
{val_table_text}
```

## Test Table

```text
{test_table_text}
```

## Figures

- `{figures["policy_metrics"]}`
- `{figures["val_delta"]}`
- `{figures["choice_counts"]}`

## Interpretation

The model selection gate is deliberately conservative. A policy must pass validation without test information before its test score is interpreted. If selected policy is RBF or test delta is non-negative, Stage 7 remains an oracle-gap problem rather than a deployable multi-hypothesis solution.
"""
    (REPORT_DIR / "stage07d_non_oracle_selector_v0_2_cn.md").write_text(tech, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    features, per_sample = load_inputs()
    allowed, feature_audit = select_allowed_features(features)
    pre, numeric, categorical = make_preprocessor(features, allowed)
    train_mask = features["split"].astype(str).eq("train").to_numpy()

    all_decisions = add_reference_decisions(features)
    for model_name, model in make_models(pre).items():
        all_decisions.extend(predict_policy_decisions(model_name, model, features, allowed, train_mask))
    decisions = pd.concat(all_decisions, ignore_index=True)
    decisions["oracle_short"] = decisions["label_diag__broad_oracle_model"].astype(str).map(choice_to_short)
    decisions["selected_short"] = decisions["selected_model"].astype(str).map(choice_to_short)

    metric_parts = [aggregate_policy(per_sample, part) for _, part in decisions.groupby("policy_name", sort=False)]
    metrics = pd.concat(metric_parts, ignore_index=True)
    metrics = add_reference_deltas(metrics)
    diag = decision_diagnostics(decisions)
    selected, selected_gate = select_policy(metrics, diag)
    selected_decisions = decisions[decisions["policy_name"] == selected].copy()
    selected_confusion = confusion_table(decisions, selected)

    selected_test = metrics[(metrics["split"] == "test") & (metrics["model_name"] == selected)].iloc[0]
    rbf_test = metrics[(metrics["split"] == "test") & (metrics["model_name"] == "always_rbf_reference")].iloc[0]
    selected_diag_test = diag[(diag["policy_name"] == selected) & (diag["split"] == "test")].iloc[0]
    gate_status = "upgrade" if selected != "always_rbf_reference" and float(selected_test["rmse_steer"]) < float(rbf_test["rmse_steer"]) - 1e-6 else "no_upgrade"
    gate = pd.DataFrame(
        [
            {
                "gate_item": "selected_policy",
                "status": selected,
                "evidence": "selected by validation gate only",
            },
            {
                "gate_item": "deployable_upgrade",
                "status": gate_status,
                "evidence": f"test delta vs RBF {float(selected_test['rmse_delta_vs_rbf']):+.6f}",
            },
            {
                "gate_item": "test_rbf_selected_rate",
                "status": f"{float(selected_diag_test['rbf_selected_rate']):.6f}",
                "evidence": "high rate means selector mostly falls back to RBF/KNN",
            },
            {
                "gate_item": "stage08_physio_eeg_allowed",
                "status": "blocked" if gate_status != "upgrade" else "still_requires_separate_gate",
                "evidence": "physio/EEG remains blocked until vehicle-only candidate selection is stable.",
            },
        ]
    )

    figures = {
        "policy_metrics": path_str(FIG_DIR / "stage07d_policy_metrics_test.png"),
        "val_delta": path_str(FIG_DIR / "stage07d_validation_rmse_delta.png"),
        "choice_counts": path_str(FIG_DIR / "stage07d_selected_choice_counts.png"),
    }
    plot_policy_metrics(metrics, selected, Path(figures["policy_metrics"]))
    plot_val_delta(metrics, Path(figures["val_delta"]))
    plot_choice_counts(decisions, selected, Path(figures["choice_counts"]))

    feature_audit.to_csv(TABLE_DIR / "stage07d_feature_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"feature": allowed, "feature_type": ["categorical" if f in categorical else "numeric" for f in allowed]}).to_csv(
        TABLE_DIR / "stage07d_allowed_features.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metrics.to_csv(TABLE_DIR / "stage07d_policy_metrics.csv", index=False, encoding="utf-8-sig")
    diag.to_csv(TABLE_DIR / "stage07d_decision_diagnostics.csv", index=False, encoding="utf-8-sig")
    decisions.to_csv(TABLE_DIR / "stage07d_all_policy_decisions.csv", index=False, encoding="utf-8-sig")
    selected_decisions.to_csv(TABLE_DIR / "stage07d_selected_policy_decisions.csv", index=False, encoding="utf-8-sig")
    selected_gate.to_csv(TABLE_DIR / "stage07d_validation_selection_table.csv", index=False, encoding="utf-8-sig")
    selected_confusion.to_csv(TABLE_DIR / "stage07d_selected_policy_confusion.csv", index=False, encoding="utf-8-sig")
    gate.to_csv(TABLE_DIR / "stage07d_gate_table.csv", index=False, encoding="utf-8-sig")

    write_reports(metrics, selected, selected_gate, diag, figures)
    summary = {
        "output_version": OUTPUT_VERSION,
        "track_id": TRACK_ID,
        "selected_policy": selected,
        "gate_status": gate_status,
        "selected_test_rmse": float(selected_test["rmse_steer"]),
        "rbf_test_rmse": float(rbf_test["rmse_steer"]),
        "selected_test_delta_vs_rbf": float(selected_test["rmse_delta_vs_rbf"]),
        "selected_test_rbf_selected_rate": float(selected_diag_test["rbf_selected_rate"]),
        "allowed_feature_count": int(len(allowed)),
        "numeric_feature_count": int(len(numeric)),
        "categorical_feature_count": int(len(categorical)),
        "server_used": False,
        "credential_file_read": False,
        "raw_files_modified": False,
        "uses_subject_id": False,
        "uses_physio": False,
        "uses_eeg": False,
        "uses_continuous_style": False,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "figures": figures,
    }
    (LOG_DIR / "stage07d_non_oracle_selector_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
