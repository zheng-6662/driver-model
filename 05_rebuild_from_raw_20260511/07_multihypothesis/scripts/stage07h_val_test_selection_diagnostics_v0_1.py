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
OUTPUT_VERSION = "stage07h_val_test_selection_diagnostics_v0_1"
TRACK_ID = "B_response3s_strict_core"

STAGE7C_ROOT = ROOT / "07_multihypothesis" / "stage07c_candidate_trajectory_export_v0_1"
STAGE7E_ROOT = ROOT / "07_multihypothesis" / "stage07e_candidate_generation_redesign_v0_1"
STAGE7G_ROOT = ROOT / "07_multihypothesis" / "stage07g_keypoint_segment_candidates_v0_1"

FEATURE_DIAG = STAGE7C_ROOT / "tables" / "candidate_feature_and_label_diagnosis.csv"
RESPONSE_TABLE = STAGE7E_ROOT / "tables" / "stage07e_response_label_table.csv"
STAGE7G_SUMMARY = STAGE7G_ROOT / "logs" / "stage07g_keypoint_segment_candidates_summary.json"
STAGE7G_METRICS = STAGE7G_ROOT / "tables" / "stage07g_candidate_metrics.csv"
STAGE7G_PER_SAMPLE = STAGE7G_ROOT / "tables" / "stage07g_candidate_per_sample_metrics.csv"
STAGE7G_TARGET_METRICS = STAGE7G_ROOT / "tables" / "stage07g_keypoint_target_metrics.csv"

OUT_ROOT = ROOT / "07_multihypothesis" / OUTPUT_VERSION
TABLE_DIR = OUT_ROOT / "tables"
FIG_DIR = OUT_ROOT / "figures"
LOG_DIR = OUT_ROOT / "logs"
REPORT_DIR = ROOT / "09_reports"

RBF_MODEL = "rbf_kernel_ridge_context_no_subject"
ORACLE_MODEL = "keypoint_segment_oracle"

CATEGORICAL_COLUMNS = [
    "direction_mode",
    "amplitude_mode",
    "peak_timing",
    "tail_mode",
    "correction_mode",
    "response_family",
    "event_level",
    "road_design_module_name",
    "road_design_risk_class",
    "top1_branch",
]
NUMERIC_COLUMNS = [
    "gt_peak_abs",
    "gt_peak_time_s",
    "gt_onset_time_s",
    "gt_tail_abs_ratio",
    "reversal_count",
    "zero_crossing",
    "anchor_time_rel_s",
    "top1_prob",
    "prob_margin",
    "prob_entropy",
    "topk_branch_spread_mean",
    "topk_branch_spread_peak",
    "branch_peak_abs_spread",
    "rbf_kernel_ridge_context_no_subject__pred_peak_abs",
    "keypoint_residual_vehicle_transformer_no_subject__pred_peak_abs",
]
BUCKET_COLUMNS = [
    "direction_mode",
    "amplitude_mode",
    "peak_timing",
    "tail_mode",
    "correction_mode",
    "road_design_module_name",
    "road_design_risk_class",
    "event_level",
]


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIG_DIR, LOG_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def path_str(path: Path) -> str:
    return str(path).replace("\\", "/")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    for path in [FEATURE_DIAG, RESPONSE_TABLE, STAGE7G_SUMMARY, STAGE7G_METRICS, STAGE7G_PER_SAMPLE, STAGE7G_TARGET_METRICS]:
        if not path.exists():
            raise FileNotFoundError(path)
    features = pd.read_csv(FEATURE_DIAG)
    response = pd.read_csv(RESPONSE_TABLE)
    metrics = pd.read_csv(STAGE7G_METRICS)
    per_sample = pd.read_csv(STAGE7G_PER_SAMPLE)
    target_metrics = pd.read_csv(STAGE7G_TARGET_METRICS)
    summary = json.loads(STAGE7G_SUMMARY.read_text(encoding="utf-8"))
    return features, response, metrics, per_sample, target_metrics, summary


def aligned_sample_table(features: pd.DataFrame, response: pd.DataFrame) -> pd.DataFrame:
    keep_feature_cols = [
        "sample_id",
        "event_uid",
        "subject",
        "session_stamp",
        "split",
        "anchor_time_rel_s",
        "event_level",
        "road_design_module_name",
        "road_design_risk_class",
        "top1_branch",
        "top1_prob",
        "prob_margin",
        "prob_entropy",
        "topk_branch_spread_mean",
        "topk_branch_spread_peak",
        "branch_peak_abs_spread",
        "rbf_kernel_ridge_context_no_subject__pred_peak_abs",
        "keypoint_residual_vehicle_transformer_no_subject__pred_peak_abs",
    ]
    cols = [c for c in keep_feature_cols if c in features.columns]
    feat = features[cols].copy()
    resp = response.drop(columns=["split"], errors="ignore").copy()
    sample = feat.merge(resp, on="sample_id", how="left", validate="one_to_one")
    sample["split"] = sample["split"].astype(str)
    return sample


def to_float(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / p.sum() if p.sum() > 0 else np.ones_like(p) / len(p)
    q = q / q.sum() if q.sum() > 0 else np.ones_like(q) / len(q)
    m = 0.5 * (p + q)
    eps = 1e-12
    kl_pm = np.sum(np.where(p > 0, p * np.log2((p + eps) / (m + eps)), 0.0))
    kl_qm = np.sum(np.where(q > 0, q * np.log2((q + eps) / (m + eps)), 0.0))
    return float(0.5 * (kl_pm + kl_qm))


def empirical_ks(x: pd.Series, y: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna().to_numpy()
    y = pd.to_numeric(y, errors="coerce").dropna().to_numpy()
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    values = np.sort(np.unique(np.concatenate([x, y])))
    cdf_x = np.searchsorted(np.sort(x), values, side="right") / len(x)
    cdf_y = np.searchsorted(np.sort(y), values, side="right") / len(y)
    return float(np.max(np.abs(cdf_x - cdf_y)))


def categorical_shift_tables(sample: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for col in [c for c in CATEGORICAL_COLUMNS if c in sample.columns]:
        values = sorted(sample[col].astype(str).fillna("NA").unique().tolist())
        props_by_split: dict[str, np.ndarray] = {}
        for split in ["train", "val", "test"]:
            part = sample[sample["split"].eq(split)]
            counts = part[col].astype(str).fillna("NA").value_counts().reindex(values, fill_value=0)
            total = int(counts.sum())
            props = (counts / total).to_numpy(dtype=float) if total else np.zeros(len(values), dtype=float)
            props_by_split[split] = props
            for value, count, prop in zip(values, counts.to_numpy(dtype=int), props):
                long_rows.append({"feature": col, "value": value, "split": split, "count": int(count), "proportion": float(prop)})
        val_test_diff = props_by_split["test"] - props_by_split["val"]
        max_idx = int(np.argmax(np.abs(val_test_diff))) if len(values) else 0
        summary_rows.append(
            {
                "feature": col,
                "n_values": int(len(values)),
                "js_val_test": js_divergence(props_by_split["val"], props_by_split["test"]),
                "js_train_val": js_divergence(props_by_split["train"], props_by_split["val"]),
                "js_train_test": js_divergence(props_by_split["train"], props_by_split["test"]),
                "largest_val_test_shift_value": values[max_idx] if values else "",
                "largest_test_minus_val_prop": float(val_test_diff[max_idx]) if len(values) else float("nan"),
            }
        )
    return pd.DataFrame(long_rows), pd.DataFrame(summary_rows).sort_values("js_val_test", ascending=False)


def numeric_shift_table(sample: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for col in [c for c in NUMERIC_COLUMNS if c in sample.columns]:
        row: dict[str, Any] = {"feature": col}
        for split in ["train", "val", "test"]:
            vals = pd.to_numeric(sample.loc[sample["split"].eq(split), col], errors="coerce")
            row[f"{split}_n"] = int(vals.notna().sum())
            row[f"{split}_mean"] = float(vals.mean()) if vals.notna().any() else float("nan")
            row[f"{split}_std"] = float(vals.std(ddof=0)) if vals.notna().any() else float("nan")
            row[f"{split}_median"] = float(vals.median()) if vals.notna().any() else float("nan")
        train_std = row.get("train_std", float("nan"))
        denom = train_std if np.isfinite(train_std) and train_std > 1e-12 else 1.0
        row["test_minus_val_mean"] = row.get("test_mean", float("nan")) - row.get("val_mean", float("nan"))
        row["test_minus_val_median"] = row.get("test_median", float("nan")) - row.get("val_median", float("nan"))
        row["std_mean_diff_val_test_by_train_std"] = row["test_minus_val_mean"] / denom
        row["ks_val_test"] = empirical_ks(sample.loc[sample["split"].eq("val"), col], sample.loc[sample["split"].eq("test"), col])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("ks_val_test", ascending=False)


def candidate_stability(metrics: pd.DataFrame, selected: str) -> tuple[pd.DataFrame, str, float]:
    metrics = to_float(
        metrics,
        [
            "rmse_steer",
            "rmse_delta_vs_rbf",
            "wrong_side_rate",
            "large_response_recall",
            "difficult_top20_rmse",
        ],
    )
    non_oracle = metrics[~metrics["model_name"].astype(str).str.contains("oracle", case=False, na=False)].copy()
    piv = non_oracle.pivot_table(
        index="model_name",
        columns="split",
        values=["rmse_steer", "rmse_delta_vs_rbf", "wrong_side_rate", "large_response_recall", "difficult_top20_rmse"],
        aggfunc="first",
    )
    piv.columns = [f"{metric}_{split}" for metric, split in piv.columns]
    out = piv.reset_index()
    for split in ["train", "val", "test"]:
        col = f"rmse_steer_{split}"
        if col in out.columns:
            out[f"{split}_rmse_rank"] = out[col].rank(method="min", ascending=True)
    out["val_test_delta_swing"] = out.get("rmse_delta_vs_rbf_test", np.nan) - out.get("rmse_delta_vs_rbf_val", np.nan)
    out["val_test_rank_shift"] = out.get("test_rmse_rank", np.nan) - out.get("val_rmse_rank", np.nan)
    out["is_selected_by_val_gate"] = out["model_name"].eq(selected)
    test_best = out[out["model_name"].ne(RBF_MODEL)].sort_values("rmse_steer_test").iloc[0]
    test_best_model = str(test_best["model_name"])
    test_best_delta = float(test_best["rmse_delta_vs_rbf_test"])
    out["is_test_best_non_oracle"] = out["model_name"].eq(test_best_model)
    out["diagnostic_status"] = np.select(
        [out["is_selected_by_val_gate"], out["is_test_best_non_oracle"]],
        ["selected_by_val_gate", "test_best_diagnostic_only"],
        default="other_candidate",
    )
    return out.sort_values(["test_rmse_rank", "val_rmse_rank"]), test_best_model, test_best_delta


def candidate_gain_tables(
    per_sample: pd.DataFrame,
    sample: pd.DataFrame,
    selected: str,
    test_best: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_sample = to_float(per_sample, ["sample_rmse", "wrong_side", "is_large_response", "is_difficult_peak_top20"])
    key_models = [RBF_MODEL, selected, test_best, "segment_resid_rf_blend_25", "rbf_resid_keypoint_scaled_blend_50", ORACLE_MODEL]
    key_models = list(dict.fromkeys([m for m in key_models if m in set(per_sample["model_name"])]))
    piv = per_sample[per_sample["model_name"].isin(key_models)].pivot_table(index="sample_id", columns="model_name", values="sample_rmse", aggfunc="first")
    long_rows: list[pd.DataFrame] = []
    base = sample.copy()
    for model in [m for m in key_models if m != RBF_MODEL]:
        if model not in piv.columns or RBF_MODEL not in piv.columns:
            continue
        part = base[["sample_id", "split", *[c for c in BUCKET_COLUMNS if c in base.columns]]].copy()
        part["candidate_model"] = model
        part["rbf_sample_rmse"] = piv.loc[part["sample_id"], RBF_MODEL].to_numpy(dtype=float)
        part["candidate_sample_rmse"] = piv.loc[part["sample_id"], model].to_numpy(dtype=float)
        part["gain_over_rbf"] = part["rbf_sample_rmse"] - part["candidate_sample_rmse"]
        part["candidate_better_than_rbf"] = part["gain_over_rbf"] > 0
        long_rows.append(part)
    gains = pd.concat(long_rows, ignore_index=True)
    bucket_rows: list[pd.DataFrame] = []
    for bucket in [c for c in BUCKET_COLUMNS if c in gains.columns]:
        grouped = (
            gains.groupby(["candidate_model", "split", bucket], dropna=False)
            .agg(
                n_samples=("gain_over_rbf", "size"),
                mean_gain_over_rbf=("gain_over_rbf", "mean"),
                median_gain_over_rbf=("gain_over_rbf", "median"),
                positive_gain_rate=("candidate_better_than_rbf", "mean"),
                mean_candidate_rmse=("candidate_sample_rmse", "mean"),
                mean_rbf_rmse=("rbf_sample_rmse", "mean"),
            )
            .reset_index()
            .rename(columns={bucket: "bucket_value"})
        )
        grouped["bucket_feature"] = bucket
        bucket_rows.append(grouped)
    bucket_summary = pd.concat(bucket_rows, ignore_index=True)
    return gains, bucket_summary.sort_values(["candidate_model", "bucket_feature", "split", "mean_gain_over_rbf"], ascending=[True, True, True, False])


def plot_candidate_stability(stability: pd.DataFrame, selected: str, test_best: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    candidates = stability[~stability["model_name"].eq(RBF_MODEL)].copy()
    x = candidates["rmse_delta_vs_rbf_val"].astype(float)
    y = candidates["rmse_delta_vs_rbf_test"].astype(float)
    colors = np.where(candidates["model_name"].eq(selected), "#d62728", np.where(candidates["model_name"].eq(test_best), "#2ca02c", "#4c78a8"))
    ax.scatter(x, y, s=70, c=colors, alpha=0.85)
    for _, row in candidates.iterrows():
        if row["model_name"] in {selected, test_best, "segment_resid_rf_blend_25", "rbf_resid_keypoint_scaled_blend_50"}:
            ax.annotate(str(row["model_name"]).replace("_", "\n"), (row["rmse_delta_vs_rbf_val"], row["rmse_delta_vs_rbf_test"]), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.axhline(0, color="#999999", linewidth=0.9)
    ax.axvline(0, color="#999999", linewidth=0.9)
    ax.set_xlabel("Validation RMSE delta vs RBF")
    ax.set_ylabel("Test RMSE delta vs RBF")
    ax.set_title("Stage 7h candidate val/test stability")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_categorical_shift(cat_long: pd.DataFrame, cat_summary: pd.DataFrame, path: Path) -> None:
    features = cat_summary.head(4)["feature"].tolist()
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), squeeze=False)
    for ax, feature in zip(axes.ravel(), features):
        part = cat_long[(cat_long["feature"].eq(feature)) & (cat_long["split"].isin(["val", "test"]))].copy()
        top_values = (
            part.groupby("value")["proportion"].max().sort_values(ascending=False).head(6).index.tolist()
        )
        part = part[part["value"].isin(top_values)]
        pivot = part.pivot(index="value", columns="split", values="proportion").fillna(0.0)
        pivot = pivot.reindex(top_values)
        x = np.arange(len(pivot))
        width = 0.38
        ax.bar(x - width / 2, pivot.get("val", pd.Series(0, index=pivot.index)), width, label="val", color="#ff7f0e")
        ax.bar(x + width / 2, pivot.get("test", pd.Series(0, index=pivot.index)), width, label="test", color="#d62728")
        ax.set_title(feature)
        ax.set_xticks(x, [str(v)[:20] for v in pivot.index], rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, max(0.05, float(pivot.max().max()) * 1.2))
        ax.grid(axis="y", alpha=0.25)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle("Largest categorical distribution shifts between val and test", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_gain_by_split(gains: pd.DataFrame, selected: str, test_best: str, path: Path) -> None:
    models = [selected, test_best, "segment_resid_rf_blend_25", "rbf_resid_keypoint_scaled_blend_50"]
    models = list(dict.fromkeys([m for m in models if m in set(gains["candidate_model"])]))
    fig, axes = plt.subplots(1, len(models), figsize=(4.3 * len(models), 4.8), squeeze=False)
    for ax, model in zip(axes.ravel(), models):
        data = [
            gains[(gains["candidate_model"].eq(model)) & (gains["split"].eq(split))]["gain_over_rbf"].dropna().to_numpy()
            for split in ["train", "val", "test"]
        ]
        ax.boxplot(data, tick_labels=["train", "val", "test"], showfliers=False)
        ax.axhline(0, color="#999999", linewidth=0.9)
        ax.set_title(model.replace("_", "\n"), fontsize=8)
        ax.set_ylabel("sample RMSE gain over RBF")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Candidate per-sample gain stability", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_target_metrics(target_metrics: pd.DataFrame, path: Path) -> None:
    tm = to_float(target_metrics, ["rmse", "mae", "bias", "corr"])
    tm = tm[tm["split"].isin(["val", "test"])].copy()
    targets = sorted(tm["target"].unique().tolist())
    fig, axes = plt.subplots(1, len(targets), figsize=(4.1 * len(targets), 4.3), squeeze=False)
    for ax, target in zip(axes.ravel(), targets):
        part = tm[tm["target"].eq(target)]
        labels: list[str] = []
        vals: list[float] = []
        colors: list[str] = []
        for prefix in ["abs", "resid"]:
            for split in ["val", "test"]:
                row = part[(part["model_prefix"].eq(prefix)) & (part["split"].eq(split))]
                if row.empty:
                    continue
                labels.append(f"{prefix}-{split}")
                vals.append(float(row.iloc[0]["rmse"]))
                colors.append("#ff7f0e" if split == "val" else "#d62728")
        ax.bar(np.arange(len(vals)), vals, color=colors)
        ax.set_title(target)
        ax.set_xticks(np.arange(len(vals)), labels, rotation=35, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Keypoint regression RMSE by split", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_reports(
    summary: dict[str, Any],
    stability: pd.DataFrame,
    cat_summary: pd.DataFrame,
    num_shift: pd.DataFrame,
    gains: pd.DataFrame,
    gate: pd.DataFrame,
    figures: dict[str, str],
    test_best: str,
    test_best_delta: float,
) -> None:
    selected = str(summary["selected_policy"])
    selected_delta = float(summary["selected_test_delta_vs_rbf"])
    rbf_rmse = float(summary["rbf_test_rmse"])
    selected_rmse = float(summary["selected_test_rmse"])
    oracle_rmse = float(summary["keypoint_segment_oracle_test_rmse"])
    top_stability = stability[
        stability["model_name"].isin([RBF_MODEL, selected, test_best, "segment_resid_rf_blend_25", "rbf_resid_keypoint_scaled_blend_50"])
    ][
        [
            "model_name",
            "rmse_delta_vs_rbf_val",
            "rmse_delta_vs_rbf_test",
            "val_rmse_rank",
            "test_rmse_rank",
            "val_test_delta_swing",
            "diagnostic_status",
        ]
    ].to_string(index=False)
    top_cat = cat_summary.head(8).to_string(index=False)
    top_num = num_shift.head(8).to_string(index=False)
    gain_summary = (
        gains[gains["candidate_model"].isin([selected, test_best])]
        .groupby(["candidate_model", "split"], dropna=False)
        .agg(n_samples=("gain_over_rbf", "size"), mean_gain=("gain_over_rbf", "mean"), positive_gain_rate=("candidate_better_than_rbf", "mean"))
        .reset_index()
        .to_string(index=False)
    )
    user = f"""# Stage 7h 用户查看版：val/test 选择不稳定诊断 v0.1

## 这个阶段为什么做

Stage 7g 出现了一个必须解释的现象：val gate 选择的 `{selected}` 在 test 上没有超过 RBF/KNN，但另一个 test-only 候选 `{test_best}` 明显更好。这个阶段不训练新模型，只诊断“为什么 val 选不中 test 上好的候选”。

## 这个阶段检查了什么

- 候选在 train/val/test 的 RMSE delta 和排名是否稳定。
- val 和 test 的响应类型、道路/事件上下文、候选置信度分布是否有偏移。
- selected 候选和 test-only 最好候选在逐样本、分响应类型和分道路 bucket 上的收益是否一致。
- 关键点回归的 val/test 误差是否一致。

## 目前发现了什么

- RBF/KNN test RMSE={rbf_rmse:.6f}。
- val gate selected=`{selected}`，selected test RMSE={selected_rmse:.6f}，delta={selected_delta:+.6f}，不能升级。
- test-only 最好非 oracle 候选=`{test_best}`，test delta={test_best_delta:+.6f}；该结果只能作为诊断，因为它没有被 val gate 选中。
- keypoint/segment oracle test RMSE={oracle_rmse:.6f}，说明候选空间仍有上限，但选择/校准还没解决。

## 候选稳定性摘要

```text
{top_stability}
```

## val/test 分布偏移摘要

```text
{top_cat}
```

```text
{top_num}
```

## 逐样本收益摘要

```text
{gain_summary}
```

## 哪些结果可信

可信的是：Stage 7h 没有训练新模型，也没有使用生理、脑电、连续风格、subject ID 或服务器凭据；它只复核 Stage 7g 已有候选在不同 split 上的稳定性。

## 哪些结果还不能下结论

不能把 `{test_best}` 当成新主线，因为它是按 test 表现事后发现的。只有未来用 train/val 规则稳定选中它或同类策略，并在 test 上仍超过 RBF/KNN，才能升级。

## 下一阶段是否可以继续

下一步应先做候选选择校准或验证集重构，例如多折 session validation、按 response bucket/道路模块分层的 val gate、关键点不确定性评分。仍不应进入生理/EEG。

## 推荐优先查看

1. `{figures["candidate_stability"]}`
2. `{figures["categorical_shift"]}`
3. `{figures["gain_by_split"]}`
4. `{figures["target_metrics"]}`
5. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07h_val_test_selection_diagnostics_v0_1/tables/stage07h_gate_table.csv`
"""
    (REPORT_DIR / "stage07h_val_test_selection_diagnostics_user_summary_cn.md").write_text(user, encoding="utf-8-sig")

    tech = f"""# Stage 7h 技术报告：val/test selection diagnostics v0.1

## Scope

- Source stage: Stage 7g keypoint/segment candidates.
- No model training in this stage.
- No server used. Credential file not read.
- Excluded from modeling claims: physio, EEG, continuous style, subject ID.

## Core Finding

- selected_by_val=`{selected}`
- selected_test_delta_vs_rbf={selected_delta:+.6f}
- test_best_non_oracle=`{test_best}`, test_delta_vs_rbf={test_best_delta:+.6f}
- gate remains no_upgrade because test-best is diagnostic only.

## Candidate Stability

```text
{top_stability}
```

## Gate

```text
{gate.to_string(index=False)}
```
"""
    (REPORT_DIR / "stage07h_val_test_selection_diagnostics_v0_1_cn.md").write_text(tech, encoding="utf-8-sig")


def main() -> None:
    ensure_dirs()
    features, response, metrics, per_sample, target_metrics, stage7g_summary = load_inputs()
    sample = aligned_sample_table(features, response)
    metrics = to_float(
        metrics,
        ["rmse_steer", "rmse_delta_vs_rbf", "wrong_side_rate", "large_response_recall", "difficult_top20_rmse"],
    )
    selected = str(stage7g_summary["selected_policy"])
    cat_long, cat_summary = categorical_shift_tables(sample)
    num_shift = numeric_shift_table(sample)
    stability, test_best, test_best_delta = candidate_stability(metrics, selected)
    gains, bucket_summary = candidate_gain_tables(per_sample, sample, selected, test_best)

    gate = pd.DataFrame(
        [
            {"gate_item": "diagnosis_completed", "status": "pass", "evidence": "val/test candidate stability and distribution shift tables generated"},
            {"gate_item": "deployable_upgrade", "status": "no_upgrade", "evidence": f"val-selected {selected} test delta {stage7g_summary['selected_test_delta_vs_rbf']:+.6f}; test-best {test_best} is diagnostic only"},
            {"gate_item": "test_best_candidate", "status": "diagnostic_only", "evidence": f"{test_best} test delta vs RBF {test_best_delta:+.6f}; not selected by validation"},
            {"gate_item": "stage08_physio_eeg_allowed", "status": "blocked", "evidence": "vehicle-only candidate selection is still not stable"},
            {"gate_item": "server_used", "status": "no", "evidence": "local diagnostic run only; credential file not read"},
        ]
    )

    figures = {
        "candidate_stability": path_str(FIG_DIR / "stage07h_candidate_val_test_stability.png"),
        "categorical_shift": path_str(FIG_DIR / "stage07h_val_test_categorical_shift.png"),
        "gain_by_split": path_str(FIG_DIR / "stage07h_candidate_gain_by_split.png"),
        "target_metrics": path_str(FIG_DIR / "stage07h_keypoint_target_rmse_by_split.png"),
    }
    plot_candidate_stability(stability, selected, test_best, Path(figures["candidate_stability"]))
    plot_categorical_shift(cat_long, cat_summary, Path(figures["categorical_shift"]))
    plot_gain_by_split(gains, selected, test_best, Path(figures["gain_by_split"]))
    plot_target_metrics(target_metrics, Path(figures["target_metrics"]))

    cat_long.to_csv(TABLE_DIR / "stage07h_categorical_shift_long.csv", index=False, encoding="utf-8-sig")
    cat_summary.to_csv(TABLE_DIR / "stage07h_categorical_shift_summary.csv", index=False, encoding="utf-8-sig")
    num_shift.to_csv(TABLE_DIR / "stage07h_numeric_shift_summary.csv", index=False, encoding="utf-8-sig")
    stability.to_csv(TABLE_DIR / "stage07h_candidate_split_stability.csv", index=False, encoding="utf-8-sig")
    gains.to_csv(TABLE_DIR / "stage07h_candidate_gain_samples.csv", index=False, encoding="utf-8-sig")
    bucket_summary.to_csv(TABLE_DIR / "stage07h_candidate_gain_by_bucket.csv", index=False, encoding="utf-8-sig")
    target_metrics.to_csv(TABLE_DIR / "stage07h_keypoint_target_metrics_copy.csv", index=False, encoding="utf-8-sig")
    gate.to_csv(TABLE_DIR / "stage07h_gate_table.csv", index=False, encoding="utf-8-sig")

    write_reports(stage7g_summary, stability, cat_summary, num_shift, gains, gate, figures, test_best, test_best_delta)

    top_shift_feature = str(cat_summary.iloc[0]["feature"]) if len(cat_summary) else ""
    top_numeric_shift = str(num_shift.iloc[0]["feature"]) if len(num_shift) else ""
    output_summary = {
        "output_version": OUTPUT_VERSION,
        "track_id": TRACK_ID,
        "source_stage": "stage07g_keypoint_segment_candidates_v0_1",
        "selected_by_val": selected,
        "selected_test_delta_vs_rbf": float(stage7g_summary["selected_test_delta_vs_rbf"]),
        "test_best_non_oracle": test_best,
        "test_best_delta_vs_rbf": float(test_best_delta),
        "gate_status": "no_upgrade",
        "stage08_physio_eeg_allowed": False,
        "top_categorical_shift_feature": top_shift_feature,
        "top_numeric_shift_feature": top_numeric_shift,
        "server_used": False,
        "credential_file_read": False,
        "raw_files_modified": False,
        "uses_physio": False,
        "uses_eeg": False,
        "uses_continuous_style": False,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "figures": figures,
    }
    (LOG_DIR / "stage07h_val_test_selection_diagnostics_summary.json").write_text(json.dumps(output_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
