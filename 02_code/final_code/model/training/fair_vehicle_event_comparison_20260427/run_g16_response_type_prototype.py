# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[4]
TRAINING_DIR = PROJECT_ROOT / "02_code" / "final_code" / "model" / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from event_conditioned_eval_support import annotate_event_meta, build_primary_selection_bundle
from run_event_conditioned_trajectory_baseline import DEFAULT_MANIFEST, build_sample_bundle_from_manifest
from run_g14_retrieval_reference import (
    add_physical_columns,
    build_available_feature_sets,
    df_to_markdown,
    group_rows,
    load_baseline_prediction_by_key,
    split_indices,
    standardize_from_train,
    summarize_variant,
)
from run_g15_retrieval_residual import (
    ensure_mask2d,
    fit_ridge_multioutput,
    predict_ridge,
    standardize_matrix,
)

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    HAS_SKLEARN = True
except Exception:
    LogisticRegression = None  # type: ignore[assignment]
    HAS_SKLEARN = False


REPORTS_DIR = PROJECT_ROOT / "04_project_logs" / "reports"
OUT_DIR = REPORTS_DIR / "g16_response_type_prototype_20260512"
FIG_DIR = OUT_DIR / "figures"
G11_CATALOG = REPORTS_DIR / "style_physio_eeg_g11_bad_case_attribution_20260509" / "bad_case_catalog.csv"
BASELINE_LOG = REPORTS_DIR / "current_model_version_result_log_20260509.csv"

FEATURE_SET_KEEP = {
    "触发前车辆和事件信息",
    "触发前车辆事件加连续风格",
    "触发前车辆事件加连续风格和肌电",
}
LABEL_SCHEMES = ["方向幅值", "方向幅值形态", "方向形态"]
CLASSIFIER_C_VALUES = [0.2, 1.0, 5.0]
RIDGE_ALPHAS = [1.0, 10.0, 100.0, 1000.0]
RESIDUAL_SCALES = [0.25, 0.50, 0.75]


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8-sig")


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8-sig")


def save_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def finite_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def response_peak_labels(y: np.ndarray, mask: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    y0 = np.asarray(y, dtype=np.float32)[:, :, 0]
    mask2d = ensure_mask2d(mask)
    for i in range(y0.shape[0]):
        valid = int(mask2d[i].sum())
        valid = max(1, min(valid, y0.shape[1]))
        curve = y0[i, :valid]
        peak_i = int(np.argmax(np.abs(curve)))
        peak_value = float(curve[peak_i])
        peak_abs = abs(peak_value)
        if peak_abs < 0.06:
            direction = "near_zero"
        elif peak_value > 0:
            direction = "positive"
        else:
            direction = "negative"
        if peak_abs >= 0.30:
            amp_bin = "large"
        elif peak_abs >= 0.10:
            amp_bin = "medium"
        else:
            amp_bin = "tiny"
        rows.append(
            {
                "true_peak_rel_value": peak_value,
                "true_peak_rel_abs": peak_abs,
                "true_peak_rel_time_s": peak_i * 2.0 / max(1, y0.shape[1]),
                "true_direction_label": direction,
                "true_amp_label": amp_bin,
            }
        )
    return pd.DataFrame(rows)


def build_label_frame(meta_annotated: pd.DataFrame, y_pool: np.ndarray, mask_pool: np.ndarray) -> pd.DataFrame:
    labels = response_peak_labels(y_pool, mask_pool)
    out = meta_annotated.reset_index(drop=True).copy()
    out = pd.concat([out, labels], axis=1)
    morph = out.get("eval_morphology_label", pd.Series(["unknown"] * len(out))).astype(str)
    morph = morph.replace({"nan": "unknown", "": "unknown"}).fillna("unknown")
    out["label_方向幅值"] = (
        out["true_direction_label"].astype(str) + "__" + out["true_amp_label"].astype(str)
    )
    out["label_方向幅值形态"] = out["label_方向幅值"].astype(str) + "__" + morph
    out["label_方向形态"] = out["true_direction_label"].astype(str) + "__" + morph
    return out


class NearestCentroidClassifier:
    def __init__(self) -> None:
        self.classes_: np.ndarray = np.asarray([], dtype=object)
        self.centroids_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "NearestCentroidClassifier":
        classes = np.asarray(sorted(pd.Series(y).astype(str).unique().tolist()), dtype=object)
        centroids = []
        for cls in classes:
            part = x[np.asarray(y).astype(str) == str(cls)]
            centroids.append(np.nanmean(part, axis=0))
        self.classes_ = classes
        self.centroids_ = np.asarray(centroids, dtype=np.float32)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.centroids_ is None or len(self.classes_) == 0:
            raise RuntimeError("classifier is not fitted")
        dist = (
            np.sum(x.astype(np.float32) * x.astype(np.float32), axis=1, keepdims=True)
            + np.sum(self.centroids_ * self.centroids_, axis=1, keepdims=True).T
            - 2.0 * x.astype(np.float32) @ self.centroids_.T
        )
        dist = np.maximum(dist, 0.0)
        logits = -np.sqrt(dist + 1e-6)
        logits = logits - np.max(logits, axis=1, keepdims=True)
        prob = np.exp(logits)
        prob = prob / np.sum(prob, axis=1, keepdims=True)
        return prob.astype(np.float32)

    def predict(self, x: np.ndarray) -> np.ndarray:
        prob = self.predict_proba(x)
        return self.classes_[np.argmax(prob, axis=1)]


def fit_classifier(x_train: np.ndarray, y_train: np.ndarray, c_value: float) -> Any:
    y_str = np.asarray(y_train).astype(str)
    classes = np.unique(y_str)
    if len(classes) < 2:
        raise ValueError("need at least two classes")
    if HAS_SKLEARN:
        try:
            clf = LogisticRegression(
                C=float(c_value),
                max_iter=1200,
                class_weight="balanced",
                solver="lbfgs",
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(x_train, y_str)
            return clf
        except Exception:
            pass
    return NearestCentroidClassifier().fit(x_train, y_str)


def classifier_predict_proba(clf: Any, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    classes = np.asarray(clf.classes_).astype(str)
    if hasattr(clf, "predict_proba"):
        prob = clf.predict_proba(x).astype(np.float32)
    else:
        pred = np.asarray(clf.predict(x)).astype(str)
        prob = np.zeros((len(pred), len(classes)), dtype=np.float32)
        cls_to_pos = {str(cls): i for i, cls in enumerate(classes)}
        for i, label in enumerate(pred):
            prob[i, cls_to_pos.get(str(label), 0)] = 1.0
    prob = np.nan_to_num(prob, nan=0.0, posinf=0.0, neginf=0.0)
    row_sum = np.sum(prob, axis=1, keepdims=True)
    row_sum[row_sum < 1e-8] = 1.0
    return classes, prob / row_sum


def class_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> dict[str, float]:
    y_t = np.asarray(y_true).astype(str)
    y_p = np.asarray(y_pred).astype(str)
    if len(y_t) == 0:
        return {f"{prefix}_accuracy": float("nan"), f"{prefix}_macro_f1": float("nan")}
    if HAS_SKLEARN:
        return {
            f"{prefix}_accuracy": float(accuracy_score(y_t, y_p)),
            f"{prefix}_macro_f1": float(f1_score(y_t, y_p, average="macro", zero_division=0)),
        }
    return {
        f"{prefix}_accuracy": float(np.mean(y_t == y_p)),
        f"{prefix}_macro_f1": float("nan"),
    }


def build_prototypes(
    y_train: np.ndarray,
    train_labels: np.ndarray,
    min_count: int = 3,
) -> tuple[dict[str, np.ndarray], dict[str, int], np.ndarray]:
    labels = np.asarray(train_labels).astype(str)
    global_mean = np.mean(y_train, axis=0).astype(np.float32)
    prototypes: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for label in sorted(pd.Series(labels).unique().tolist()):
        idx = np.where(labels == str(label))[0]
        counts[str(label)] = int(len(idx))
        if len(idx) >= min_count:
            prototypes[str(label)] = np.mean(y_train[idx], axis=0).astype(np.float32)
    return prototypes, counts, global_mean


def hard_prototype_prediction(labels: np.ndarray, prototypes: dict[str, np.ndarray], global_mean: np.ndarray) -> np.ndarray:
    pred = []
    for label in np.asarray(labels).astype(str):
        pred.append(prototypes.get(str(label), global_mean))
    return np.stack(pred, axis=0).astype(np.float32)


def soft_prototype_prediction(
    classes: np.ndarray,
    prob: np.ndarray,
    prototypes: dict[str, np.ndarray],
    global_mean: np.ndarray,
) -> np.ndarray:
    proto = np.stack([prototypes.get(str(cls), global_mean) for cls in classes.astype(str)], axis=0).astype(np.float32)
    return np.einsum("nc,ctd->ntd", prob.astype(np.float32), proto).astype(np.float32)


def prototype_feature_block(z: np.ndarray, idx: np.ndarray, prob: np.ndarray, pred: np.ndarray) -> np.ndarray:
    pred_curve = pred[:, :, 0].astype(np.float32)
    peak_abs = np.max(np.abs(pred_curve), axis=1, keepdims=True)
    end_value = pred_curve[:, -1:]
    mean_value = np.mean(pred_curve, axis=1, keepdims=True)
    std_value = np.std(pred_curve, axis=1, keepdims=True)
    max_prob = np.max(prob, axis=1, keepdims=True).astype(np.float32)
    entropy = -np.sum(prob * np.log(prob + 1e-8), axis=1, keepdims=True).astype(np.float32)
    return np.concatenate(
        [z[idx].astype(np.float32), prob.astype(np.float32), max_prob, entropy, peak_abs, end_value, mean_value, std_value],
        axis=1,
    ).astype(np.float32)


def evaluate_prediction(
    model_id: str,
    feature_set: str,
    label_scheme: str,
    mode: str,
    split_name: str,
    pred: np.ndarray,
    true: np.ndarray,
    mask: np.ndarray,
    ctx: np.ndarray,
    meta: pd.DataFrame,
    g11_keys: set[str],
    classifier_c: float | None = None,
    alpha: float | None = None,
    residual_scale: float | None = None,
    classifier_meta: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    mask2d = ensure_mask2d(mask)
    bundle = build_primary_selection_bundle(
        pred=pred,
        true=true,
        mask=mask2d,
        ctx_raw=ctx,
        meta_df=meta,
        split_name=split_name,
        seed=2026,
    )
    sample_df = add_physical_columns(bundle["sample_df"], pred, true, mask2d, ctx)
    sample_df["model_id"] = model_id
    sample_df["feature_set"] = feature_set
    sample_df["label_scheme"] = label_scheme
    sample_df["prototype_mode"] = mode
    sample_df["classifier_c"] = finite_float(classifier_c)
    sample_df["alpha"] = finite_float(alpha)
    sample_df["residual_scale"] = finite_float(residual_scale)
    row = summarize_variant(feature_set, 0, sample_df, bundle["selection_summary"], g11_keys)
    row.update(
        {
            "model_id": model_id,
            "split": split_name,
            "label_scheme": label_scheme,
            "prototype_mode": mode,
            "classifier_c": finite_float(classifier_c),
            "alpha": finite_float(alpha),
            "residual_scale": finite_float(residual_scale),
        }
    )
    if classifier_meta:
        row.update(classifier_meta)
    return row, sample_df


def select_best_deployable(val_df: pd.DataFrame) -> pd.DataFrame:
    deploy = val_df[~val_df["model_id"].astype(str).str.contains("真实响应类型")].copy()
    if deploy.empty:
        return pd.DataFrame()
    deploy = deploy.sort_values(["selection_score", "test_rmse", "g11_rmse"], ascending=True)
    return deploy.head(1).reset_index(drop=True)


def select_best_oracle(val_df: pd.DataFrame) -> pd.DataFrame:
    oracle = val_df[val_df["model_id"].astype(str).str.contains("真实响应类型")].copy()
    if oracle.empty:
        return pd.DataFrame()
    oracle = oracle.sort_values(["selection_score", "test_rmse", "g11_rmse"], ascending=True)
    return oracle.head(1).reset_index(drop=True)


def plot_selected_cases(
    out_path: Path,
    sample_df: pd.DataFrame,
    true: np.ndarray,
    ctx: np.ndarray,
    pred_map: dict[str, np.ndarray],
    sample_keys: list[str],
    baseline_map: dict[str, np.ndarray],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    key_to_i = {str(row.sample_key): int(i) for i, row in sample_df.reset_index(drop=True).iterrows()}
    selected = [key for key in sample_keys if key in key_to_i][:12]
    if not selected:
        return
    fig, axes = plt.subplots(3, 4, figsize=(18, 10), sharex=True)
    axes = axes.reshape(-1)
    t = np.arange(true.shape[1], dtype=np.float32) * 2.0 / max(1, true.shape[1])
    for ax, key in zip(axes, selected):
        i = key_to_i[key]
        anchor = float(ctx[i, 0])
        ax.plot(t, true[i, :, 0] + anchor, color="black", linewidth=2.0, label="true")
        if key in baseline_map:
            ax.plot(t, baseline_map[key][:, 0] + anchor, color="#1f77b4", linewidth=1.1, alpha=0.85, label="E10C")
        for name, pred in pred_map.items():
            ax.plot(t, pred[i, :, 0] + anchor, linewidth=1.2, alpha=0.9, label=name)
        row = sample_df.iloc[i]
        ax.set_title(f"{row.get('subj','?')} | {row.get('eval_morphology_label','?')} | {row.get('true_peak_abs_bin','?')}", fontsize=9)
        ax.axhline(0.0, color="#999999", linewidth=0.6)
        ax.grid(True, alpha=0.2)
    for ax in axes[len(selected) :]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(6, len(labels)))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def build_report(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    chosen_df: pd.DataFrame,
    classifier_df: pd.DataFrame,
    group_df: pd.DataFrame,
    g11_df: pd.DataFrame,
    regime_df: pd.DataFrame,
) -> str:
    baseline_text = "基准表读取失败或不存在。"
    if BASELINE_LOG.exists():
        try:
            base = pd.read_csv(BASELINE_LOG)
            keep = base[base["version"].astype(str).isin(["E2", "E5A", "E6", "E10C"])].copy()
            cols = [c for c in ["version", "test_rmse", "primary_rmse", "tail_rmse", "selection", "decision"] if c in keep.columns]
            baseline_text = df_to_markdown(keep[cols])
        except Exception:
            baseline_text = "基准表读取失败。"

    show_cols = [
        "model_id",
        "feature_set",
        "label_scheme",
        "prototype_mode",
        "test_rmse",
        "primary_rmse",
        "tail_rmse",
        "selection_score",
        "g11_rmse",
        "large_rmse",
        "reverse_rmse",
        "multi_rmse",
        "severe_under_amp_rate",
        "opposite_peak_rate",
        "classifier_c",
        "alpha",
        "residual_scale",
        "test_label_accuracy",
        "test_label_macro_f1",
    ]
    val_show = val_df.sort_values(["selection_score", "test_rmse"]).head(12)
    test_show = test_df.sort_values(["selection_score", "test_rmse"]).head(16)
    chosen_show = chosen_df[[c for c in show_cols if c in chosen_df.columns]].copy()

    clf_show = classifier_df.copy()
    for col in ["val_label_macro_f1", "val_label_accuracy", "test_label_macro_f1", "test_label_accuracy"]:
        if col not in clf_show.columns:
            clf_show[col] = np.nan
    clf_show = clf_show.sort_values(["val_label_macro_f1", "val_label_accuracy"], ascending=False).head(12)
    clf_cols = [
        "feature_set",
        "label_scheme",
        "classifier_c",
        "class_count",
        "val_label_accuracy",
        "val_label_macro_f1",
        "test_label_accuracy",
        "test_label_macro_f1",
    ]

    g11_text = "无 G11 逐样本结果。"
    if not g11_df.empty:
        group_cols = [
            "model_id",
            "feature_set",
            "label_scheme",
            "prototype_mode",
            "classifier_c",
            "alpha",
            "residual_scale",
        ]
        g11_summary = g11_df.groupby(group_cols, dropna=False).agg(
            sample_count=("sample_key", "count"),
            rmse=("rmse_2s_abs_steer", "mean"),
            tail_rmse=("rmse_tail_abs_steer", "mean"),
            severe_under_amp_rate=("severe_under_amp", "mean"),
            opposite_peak_rate=("opposite_at_true_peak", "mean"),
        ).reset_index()
        if not chosen_df.empty:
            chosen_keys = set(
                chosen_df[group_cols]
                .fillna("__NA__")
                .astype(str)
                .agg("||".join, axis=1)
                .tolist()
            )
            g11_summary["_key"] = (
                g11_summary[group_cols]
                .fillna("__NA__")
                .astype(str)
                .agg("||".join, axis=1)
            )
            chosen_g11 = g11_summary[g11_summary["_key"].isin(chosen_keys)].drop(columns=["_key"])
            if not chosen_g11.empty:
                g11_summary = chosen_g11
        g11_text = df_to_markdown(g11_summary)

    subj_text = "无分被试结果。"
    if not group_df.empty and not chosen_df.empty:
        group_key_cols = [
            "model_id",
            "feature_set",
            "label_scheme",
            "prototype_mode",
            "classifier_c",
            "alpha",
            "residual_scale",
        ]
        keys = set(
            chosen_df[group_key_cols]
            .fillna("__NA__")
            .astype(str)
            .agg("||".join, axis=1)
            .tolist()
        )
        subj = group_df[group_df["group_family"].eq("subj")].copy()
        for col in group_key_cols:
            if col not in subj.columns:
                subj[col] = np.nan
        subj["_key"] = subj[group_key_cols].fillna("__NA__").astype(str).agg("||".join, axis=1)
        subj = subj[subj["_key"].isin(keys)]
        subj_cols = ["model_id", "group_label", "sample_count", "rmse", "tail_rmse", "severe_under_amp_rate", "opposite_peak_rate"]
        subj_text = df_to_markdown(subj[[c for c in subj_cols if c in subj.columns]])

    regime_top = regime_df.head(20) if not regime_df.empty else pd.DataFrame()

    return f"""# G16 路线2：先判断响应类型，再选择原型轨迹

## 1. 这轮为什么做

G15 路线1说明：直接按触发前相似度找历史事件，整体 2 秒平均 RMSE 可以变低，但主响应阶段、尾段、综合选择指标和 G11 困难样本仍然不好。这说明“历史原型本身有价值”，但模型在推理时仍然不知道该选哪种方向、幅值和响应形态。

所以 G16 路线2改成两步：

1. 用训练集真实未来轨迹生成响应类型标签，例如方向、幅值等级和响应形态。这个标签只用于训练监督和诊断。
2. 推理时只看触发前车辆/事件、连续风格和肌电等可用信息，先预测响应类型，再用预测到的类型去选择训练集平均原型，或者按概率软组合多个原型。

这轮不是最终模型，而是专门回答一个问题：**如果先判断响应类型，是否能缓解旧模型“趋势像但方向、幅值和物理意义不对”的问题。**

## 2. 公平边界

- 样本清单仍使用：`{DEFAULT_MANIFEST}`。
- train/val/test 划分沿用当前 FAIR 协议。
- 原型轨迹只从训练集构建，测试集真实轨迹不会进入原型库。
- “真实响应类型原型上限”使用了测试集真实响应类型，只作为诊断上限，不能当部署模型。
- “预测响应类型原型/软原型/残差修正”推理时只使用触发前可用信息。
- 验证集用于选择特征组、标签方式、分类器强度和残差强度；测试集只做最终汇报。

## 3. 历史强基准

{baseline_text}

## 4. 响应类型判断器表现

{df_to_markdown(clf_show[[c for c in clf_cols if c in clf_show.columns]])}

## 5. 验证集筛选前 12

{df_to_markdown(val_show[[c for c in show_cols if c in val_show.columns]])}

## 6. 测试集候选结果

{df_to_markdown(test_show[[c for c in show_cols if c in test_show.columns]])}

## 7. 验证集选中的版本

{df_to_markdown(chosen_show)}

## 8. G11 困难样本

{g11_text}

## 9. 分被试结果

{subj_text}

## 10. 训练集响应类型数量前 20

{df_to_markdown(regime_top)}

## 11. 当前结论

1. 如果真实响应类型原型明显好于预测响应类型原型，说明训练集中存在可用的响应类型原型，但当前触发前信息还不足以稳定判断类型。
2. 如果预测响应类型原型接近 E10C/E5A，并改善 G11、幅值不足或错侧，说明“先判断响应类型”值得继续做成神经网络条件化模型。
3. 如果预测响应类型原型仍然只改善普通样本、不改善困难样本，就说明瓶颈不是输出结构本身，而是推理时缺少能判断方向/幅值/形态的信息，后续应优先查事件锚点、额外上下文、生理时序或多假设选择。

## 12. 产物

- 验证集筛选表：`{OUT_DIR / "g16_validation_screening.csv"}`
- 测试集候选表：`{OUT_DIR / "g16_test_all_candidates.csv"}`
- 选中结果：`{OUT_DIR / "g16_test_chosen_by_validation.csv"}`
- 分类器指标：`{OUT_DIR / "g16_classifier_metrics.csv"}`
- 响应类型数量：`{OUT_DIR / "g16_regime_counts.csv"}`
- 分组统计：`{OUT_DIR / "g16_group_summary.csv"}`
- G11 明细：`{OUT_DIR / "g16_g11_detail.csv"}`
- 预测数组：`{OUT_DIR / "g16_chosen_predictions_test.npz"}`
- 固定图：`{FIG_DIR / "g16_selected_g11_comparison.png"}`
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("G16 route2: loading FAIR sample bundle...", flush=True)
    x_pool, y_pool, _curve_pool, ctx_pool, mask_pool, meta_df, dropped = build_sample_bundle_from_manifest(
        DEFAULT_MANIFEST,
        max_train_samples=None,
        max_val_samples=None,
        max_test_samples=None,
        seed=2026,
    )
    train_idx, val_idx, test_idx = split_indices(meta_df)
    meta_annotated = annotate_event_meta(meta_df, y_pool, mask_pool)
    label_df = build_label_frame(meta_annotated, y_pool, mask_pool)

    y_train = y_pool[train_idx].astype(np.float32)
    y_val = y_pool[val_idx].astype(np.float32)
    y_test = y_pool[test_idx].astype(np.float32)
    mask_train = ensure_mask2d(mask_pool[train_idx])
    mask_val = ensure_mask2d(mask_pool[val_idx])
    mask_test = ensure_mask2d(mask_pool[test_idx])
    ctx_train = ctx_pool[train_idx].astype(np.float32)
    ctx_val = ctx_pool[val_idx].astype(np.float32)
    ctx_test = ctx_pool[test_idx].astype(np.float32)
    meta_val = label_df.iloc[val_idx].reset_index(drop=True)
    meta_test = label_df.iloc[test_idx].reset_index(drop=True)

    g11_catalog = pd.read_csv(G11_CATALOG) if G11_CATALOG.exists() else pd.DataFrame()
    g11_keys = set(g11_catalog["sample_key"].astype(str).tolist()) if not g11_catalog.empty else set()

    feature_sets, feature_names, context_meta = build_available_feature_sets(x_pool, ctx_pool, meta_annotated, train_idx)
    deployable_feature_sets = {k: v for k, v in feature_sets.items() if k in FEATURE_SET_KEEP}

    regime_rows: list[dict[str, Any]] = []
    for scheme in LABEL_SCHEMES:
        counts = label_df.iloc[train_idx][f"label_{scheme}"].astype(str).value_counts()
        for label, count in counts.items():
            regime_rows.append({"label_scheme": scheme, "label": str(label), "train_count": int(count)})
    regime_df = pd.DataFrame(regime_rows)
    save_csv(OUT_DIR / "g16_regime_counts.csv", regime_df)

    val_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    classifier_rows: list[dict[str, Any]] = []
    group_rows_all: list[dict[str, Any]] = []
    g11_frames: list[pd.DataFrame] = []

    cached_predictions: dict[tuple[str, str, str, str, float, float, float], tuple[np.ndarray, np.ndarray]] = {}
    cached_test_samples: dict[tuple[str, str, str, str, float, float, float], pd.DataFrame] = {}

    for scheme in LABEL_SCHEMES:
        label_col = f"label_{scheme}"
        labels_all = label_df[label_col].astype(str).to_numpy()
        y_label_train = labels_all[train_idx]
        y_label_val = labels_all[val_idx]
        y_label_test = labels_all[test_idx]
        prototypes, proto_counts, global_proto = build_prototypes(y_train, y_label_train, min_count=3)
        proto_meta = {
            "prototype_count": len(prototypes),
            "prototype_min_count": int(min(proto_counts.values())) if proto_counts else 0,
            "prototype_max_count": int(max(proto_counts.values())) if proto_counts else 0,
        }

        oracle_val_pred = hard_prototype_prediction(y_label_val, prototypes, global_proto)
        oracle_test_pred = hard_prototype_prediction(y_label_test, prototypes, global_proto)
        oracle_model = "G16O_真实响应类型原型上限"
        val_row, _ = evaluate_prediction(
            oracle_model,
            "真实响应类型",
            scheme,
            "真实类型硬原型",
            "val",
            oracle_val_pred,
            y_val,
            mask_val,
            ctx_val,
            meta_val,
            set(),
            classifier_meta=proto_meta,
        )
        test_row, test_sample = evaluate_prediction(
            oracle_model,
            "真实响应类型",
            scheme,
            "真实类型硬原型",
            "test",
            oracle_test_pred,
            y_test,
            mask_test,
            ctx_test,
            meta_test,
            g11_keys,
            classifier_meta=proto_meta,
        )
        val_rows.append(val_row)
        test_rows.append(test_row)
        key = (oracle_model, "真实响应类型", scheme, "真实类型硬原型", float("nan"), float("nan"), float("nan"))
        cached_predictions[key] = (oracle_val_pred, oracle_test_pred)
        cached_test_samples[key] = test_sample
        if g11_keys:
            g11_frames.append(test_sample[test_sample["sample_key"].astype(str).isin(g11_keys)].copy())
        for gr in group_rows("真实响应类型", 0, test_sample):
            gr.update(
                {
                    "model_id": oracle_model,
                    "label_scheme": scheme,
                    "prototype_mode": "真实类型硬原型",
                    "classifier_c": np.nan,
                    "alpha": np.nan,
                    "residual_scale": np.nan,
                }
            )
            group_rows_all.append(gr)

        for feature_set, raw_features in deployable_feature_sets.items():
            print(f"G16 scheme={scheme} feature={feature_set}", flush=True)
            z, _stats = standardize_from_train(raw_features, train_idx)
            x_train = z[train_idx].astype(np.float32)
            x_val = z[val_idx].astype(np.float32)
            x_test = z[test_idx].astype(np.float32)

            for c_value in CLASSIFIER_C_VALUES:
                try:
                    clf = fit_classifier(x_train, y_label_train, c_value)
                except Exception as exc:
                    classifier_rows.append(
                        {
                            "feature_set": feature_set,
                            "label_scheme": scheme,
                            "classifier_c": c_value,
                            "class_count": int(len(np.unique(y_label_train))),
                            "fit_error": str(exc),
                        }
                    )
                    continue
                classes, prob_train = classifier_predict_proba(clf, x_train)
                _classes_val, prob_val = classifier_predict_proba(clf, x_val)
                _classes_test, prob_test = classifier_predict_proba(clf, x_test)
                pred_train_label = classes[np.argmax(prob_train, axis=1)]
                pred_val_label = classes[np.argmax(prob_val, axis=1)]
                pred_test_label = classes[np.argmax(prob_test, axis=1)]
                clf_meta = {
                    **proto_meta,
                    "class_count": int(len(classes)),
                    "classifier_c": float(c_value),
                    **class_metrics(y_label_val, pred_val_label, "val_label"),
                    **class_metrics(y_label_test, pred_test_label, "test_label"),
                }
                classifier_rows.append({"feature_set": feature_set, "label_scheme": scheme, **clf_meta})

                hard_val = hard_prototype_prediction(pred_val_label, prototypes, global_proto)
                hard_test = hard_prototype_prediction(pred_test_label, prototypes, global_proto)
                for model_id, mode, val_pred, test_pred in [
                    ("G16A_预测响应类型硬原型", "预测类型硬原型", hard_val, hard_test),
                    ("G16B_预测响应类型软原型", "预测类型软原型", soft_prototype_prediction(classes, prob_val, prototypes, global_proto), soft_prototype_prediction(classes, prob_test, prototypes, global_proto)),
                ]:
                    val_row, _ = evaluate_prediction(
                        model_id,
                        feature_set,
                        scheme,
                        mode,
                        "val",
                        val_pred,
                        y_val,
                        mask_val,
                        ctx_val,
                        meta_val,
                        set(),
                        classifier_c=c_value,
                        classifier_meta=clf_meta,
                    )
                    test_row, test_sample = evaluate_prediction(
                        model_id,
                        feature_set,
                        scheme,
                        mode,
                        "test",
                        test_pred,
                        y_test,
                        mask_test,
                        ctx_test,
                        meta_test,
                        g11_keys,
                        classifier_c=c_value,
                        classifier_meta=clf_meta,
                    )
                    val_rows.append(val_row)
                    test_rows.append(test_row)
                    key = (model_id, feature_set, scheme, mode, float(c_value), float("nan"), float("nan"))
                    cached_predictions[key] = (val_pred, test_pred)
                    cached_test_samples[key] = test_sample
                    if g11_keys:
                        g11_frames.append(test_sample[test_sample["sample_key"].astype(str).isin(g11_keys)].copy())
                    for gr in group_rows(feature_set, 0, test_sample):
                        gr.update(
                            {
                                "model_id": model_id,
                                "label_scheme": scheme,
                                "prototype_mode": mode,
                                "classifier_c": float(c_value),
                                "alpha": np.nan,
                                "residual_scale": np.nan,
                            }
                        )
                        group_rows_all.append(gr)

                base_train = soft_prototype_prediction(classes, prob_train, prototypes, global_proto)
                base_val = soft_prototype_prediction(classes, prob_val, prototypes, global_proto)
                base_test = soft_prototype_prediction(classes, prob_test, prototypes, global_proto)
                feat_train = prototype_feature_block(z, train_idx, prob_train, base_train)
                feat_val = prototype_feature_block(z, val_idx, prob_val, base_val)
                feat_test = prototype_feature_block(z, test_idx, prob_test, base_test)
                feat_train_z, others_z, _ = standardize_matrix(feat_train, feat_val, feat_test)
                feat_val_z, feat_test_z = others_z
                target_residual = y_train[:, :, 0] - base_train[:, :, 0]
                for alpha in RIDGE_ALPHAS:
                    try:
                        weights = fit_ridge_multioutput(feat_train_z, target_residual, alpha=float(alpha))
                    except Exception:
                        continue
                    delta_val = predict_ridge(feat_val_z, weights)
                    delta_test = predict_ridge(feat_test_z, weights)
                    for scale in RESIDUAL_SCALES:
                        res_val = base_val.copy()
                        res_test = base_test.copy()
                        res_val[:, :, 0] = res_val[:, :, 0] + float(scale) * delta_val
                        res_test[:, :, 0] = res_test[:, :, 0] + float(scale) * delta_test
                        model_id = "G16C_预测响应类型软原型加残差"
                        mode = "软原型加残差"
                        val_row, _ = evaluate_prediction(
                            model_id,
                            feature_set,
                            scheme,
                            mode,
                            "val",
                            res_val,
                            y_val,
                            mask_val,
                            ctx_val,
                            meta_val,
                            set(),
                            classifier_c=c_value,
                            alpha=alpha,
                            residual_scale=scale,
                            classifier_meta=clf_meta,
                        )
                        test_row, test_sample = evaluate_prediction(
                            model_id,
                            feature_set,
                            scheme,
                            mode,
                            "test",
                            res_test,
                            y_test,
                            mask_test,
                            ctx_test,
                            meta_test,
                            g11_keys,
                            classifier_c=c_value,
                            alpha=alpha,
                            residual_scale=scale,
                            classifier_meta=clf_meta,
                        )
                        val_rows.append(val_row)
                        test_rows.append(test_row)
                        key = (model_id, feature_set, scheme, mode, float(c_value), float(alpha), float(scale))
                        cached_predictions[key] = (res_val, res_test)
                        cached_test_samples[key] = test_sample
                        if g11_keys:
                            g11_frames.append(test_sample[test_sample["sample_key"].astype(str).isin(g11_keys)].copy())
                        for gr in group_rows(feature_set, 0, test_sample):
                            gr.update(
                                {
                                    "model_id": model_id,
                                    "label_scheme": scheme,
                                    "prototype_mode": mode,
                                    "classifier_c": float(c_value),
                                    "alpha": float(alpha),
                                    "residual_scale": float(scale),
                                }
                            )
                            group_rows_all.append(gr)

    val_df = pd.DataFrame(val_rows)
    test_df = pd.DataFrame(test_rows)
    classifier_df = pd.DataFrame(classifier_rows)
    group_df = pd.DataFrame(group_rows_all)
    g11_df = pd.concat(g11_frames, ignore_index=True) if g11_frames else pd.DataFrame()

    save_csv(OUT_DIR / "g16_validation_screening.csv", val_df)
    save_csv(OUT_DIR / "g16_test_all_candidates.csv", test_df)
    save_csv(OUT_DIR / "g16_classifier_metrics.csv", classifier_df)
    save_csv(OUT_DIR / "g16_group_summary.csv", group_df)
    save_csv(OUT_DIR / "g16_g11_detail.csv", g11_df)

    chosen_rows = []
    chosen_pred_map: dict[str, np.ndarray] = {}
    chosen_samples: list[pd.DataFrame] = []
    for chosen in [select_best_oracle(val_df), select_best_deployable(val_df)]:
        if chosen.empty:
            continue
        row = chosen.iloc[0]
        mask = (
            test_df["model_id"].astype(str).eq(str(row["model_id"]))
            & test_df["feature_set"].astype(str).eq(str(row["feature_set"]))
            & test_df["label_scheme"].astype(str).eq(str(row["label_scheme"]))
            & test_df["prototype_mode"].astype(str).eq(str(row["prototype_mode"]))
        )
        for numeric_col in ("classifier_c", "alpha", "residual_scale"):
            target = row.get(numeric_col)
            if pd.isna(target):
                mask &= test_df[numeric_col].isna()
            else:
                mask &= np.isclose(test_df[numeric_col].astype(float), float(target), equal_nan=True)
        match = test_df[mask].head(1)
        if match.empty:
            continue
        test_row = match.iloc[0].to_dict()
        chosen_rows.append(test_row)
        cache_key = (
            str(test_row["model_id"]),
            str(test_row["feature_set"]),
            str(test_row["label_scheme"]),
            str(test_row["prototype_mode"]),
            finite_float(test_row.get("classifier_c")),
            finite_float(test_row.get("alpha")),
            finite_float(test_row.get("residual_scale")),
        )
        pred_pair = cached_predictions.get(cache_key)
        sample = cached_test_samples.get(cache_key)
        if pred_pair is not None:
            _, test_pred = pred_pair
            chosen_pred_map[str(test_row["model_id"])] = test_pred
        if sample is not None:
            chosen_samples.append(sample)

    chosen_df = pd.DataFrame(chosen_rows)
    save_csv(OUT_DIR / "g16_test_chosen_by_validation.csv", chosen_df)

    if chosen_pred_map:
        payload: dict[str, Any] = {
            "sample_key": meta_test["sample_key"].astype(str).to_numpy(dtype="<U512"),
            "true": y_test,
            "mask": mask_test,
            "ctx": ctx_test,
            "model_names": np.asarray(list(chosen_pred_map.keys()), dtype="<U128"),
        }
        for i, (_name, pred) in enumerate(chosen_pred_map.items()):
            payload[f"pred_{i}"] = pred
        np.savez_compressed(OUT_DIR / "g16_chosen_predictions_test.npz", **payload)

    if chosen_samples:
        first_sample = chosen_samples[-1]
        sample_keys: list[str] = []
        if not g11_df.empty:
            sample_keys = (
                g11_df.groupby("sample_key")["rmse_2s_abs_steer"]
                .mean()
                .sort_values(ascending=False)
                .head(12)
                .index.astype(str)
                .tolist()
            )
        baseline_map = load_baseline_prediction_by_key()
        plot_selected_cases(
            FIG_DIR / "g16_selected_g11_comparison.png",
            first_sample,
            y_test,
            ctx_test,
            chosen_pred_map,
            sample_keys,
            baseline_map,
        )

    save_json(
        OUT_DIR / "g16_run_meta.json",
        {
            "manifest": str(DEFAULT_MANIFEST),
            "out_dir": str(OUT_DIR),
            "train_count": int(len(train_idx)),
            "val_count": int(len(val_idx)),
            "test_count": int(len(test_idx)),
            "dropped_count": int(len(dropped)) if hasattr(dropped, "__len__") else None,
            "label_schemes": LABEL_SCHEMES,
            "feature_sets": list(deployable_feature_sets.keys()),
            "classifier_backend": "sklearn_logistic_regression" if HAS_SKLEARN else "nearest_centroid_fallback",
            "selection_rule": "validation split lowest selection_score then test_rmse; oracle kept only as diagnostic upper bound",
            "context_meta": context_meta,
            "feature_name_counts": {key: len(value) for key, value in feature_names.items()},
        },
    )

    report = build_report(val_df, test_df, chosen_df, classifier_df, group_df, g11_df, regime_df)
    write_text(OUT_DIR / "g16_response_type_prototype_report_cn.md", report)
    print(f"G16 done. Report: {OUT_DIR / 'g16_response_type_prototype_report_cn.md'}", flush=True)


if __name__ == "__main__":
    main()
