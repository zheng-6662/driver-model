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
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline


ROOT = Path("F:/data_set_process/data_process/05_rebuild_from_raw_20260511")
SCRIPT_DIR = ROOT / "07_multihypothesis" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from stage07f_response_factorized_candidates_v0_1 import (  # noqa: E402
    BRANCH_MODELS,
    KEYPOINT_MODEL,
    RBF_MODEL,
    SPLIT_STRATEGY,
    TOP1_MODEL,
    TRACK_ID,
    add_reference_deltas,
    evaluate_predictions,
    make_preprocessor,
    oracle_prediction,
    select_allowed_features,
)


OUTPUT_VERSION = "stage07g_keypoint_segment_candidates_v0_1"

STAGE7C_ROOT = ROOT / "07_multihypothesis" / "stage07c_candidate_trajectory_export_v0_1"
STAGE7E_ROOT = ROOT / "07_multihypothesis" / "stage07e_candidate_generation_redesign_v0_1"
TRAJECTORY_NPZ = STAGE7C_ROOT / "arrays" / "stage07c_candidate_trajectories.npz"
FEATURE_DIAG = STAGE7C_ROOT / "tables" / "candidate_feature_and_label_diagnosis.csv"
RESPONSE_TABLE = STAGE7E_ROOT / "tables" / "stage07e_response_label_table.csv"

OUT_ROOT = ROOT / "07_multihypothesis" / OUTPUT_VERSION
TABLE_DIR = OUT_ROOT / "tables"
FIG_DIR = OUT_ROOT / "figures"
LOG_DIR = OUT_ROOT / "logs"
REPORT_DIR = ROOT / "09_reports"

TARGETS = ["peak_signed", "peak_time_s", "onset_time_s", "tail_signed"]
RANDOM_STATE = 20260513


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIG_DIR, LOG_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def path_str(path: Path) -> str:
    return str(path).replace("\\", "/")


def smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def load_inputs() -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    if not TRAJECTORY_NPZ.exists():
        raise FileNotFoundError(TRAJECTORY_NPZ)
    if not FEATURE_DIAG.exists():
        raise FileNotFoundError(FEATURE_DIAG)
    if not RESPONSE_TABLE.exists():
        raise FileNotFoundError(RESPONSE_TABLE)
    z = dict(np.load(TRAJECTORY_NPZ, allow_pickle=True))
    features = pd.read_csv(FEATURE_DIAG)
    response = pd.read_csv(RESPONSE_TABLE)
    return z, features, response


def aligned_inputs() -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, int]]:
    z, features, response = load_inputs()
    sample_ids = z["sample_ids"].astype(str)
    split = z["split"].astype(str)
    features = features.set_index("sample_id").loc[sample_ids].reset_index()
    features["split"] = split
    response = response.set_index("sample_id").loc[sample_ids].reset_index()
    response["split"] = split
    meta_cols = ["sample_id", "event_uid", "subject", "session_stamp"]
    response = response.merge(features[meta_cols], on="sample_id", how="left", validate="one_to_one")
    names = [str(x) for x in z["candidate_model_names"].tolist()]
    candidate_idx = {name: names.index(name) for name in names}
    return z, features, response, candidate_idx


def make_regressor(kind: str = "rf") -> Any:
    if kind == "extra":
        return ExtraTreesRegressor(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    return RandomForestRegressor(
        n_estimators=450,
        max_depth=7,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def fit_target_models(
    features: pd.DataFrame,
    allowed_features: list[str],
    target_df: pd.DataFrame,
    train_mask: np.ndarray,
    prefix: str,
    kind: str,
) -> tuple[dict[str, Pipeline], pd.DataFrame]:
    models: dict[str, Pipeline] = {}
    pred = pd.DataFrame({"sample_id": features["sample_id"].astype(str), "split": features["split"].astype(str)})
    for target in TARGETS:
        pre, _, _ = make_preprocessor(features, allowed_features)
        model = Pipeline([("pre", pre), ("reg", make_regressor(kind))])
        model.fit(features.loc[train_mask, allowed_features], target_df.loc[train_mask, target].astype(float))
        models[target] = model
        pred[f"{prefix}_{target}"] = model.predict(features[allowed_features]).astype(float)
    return models, pred


def keypoints_from_predictions(pred: np.ndarray, mask: np.ndarray, label_time: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for i in range(pred.shape[0]):
        valid = mask[i] & np.isfinite(pred[i])
        vals = pred[i, valid]
        times = label_time[valid]
        if len(vals) == 0:
            rows.append({"peak_signed": 0.0, "peak_time_s": 0.0, "onset_time_s": 0.0, "tail_signed": 0.0})
            continue
        peak_idx = int(np.nanargmax(np.abs(vals)))
        peak_signed = float(vals[peak_idx])
        peak_time = float(times[peak_idx])
        peak_abs = abs(peak_signed)
        onset_threshold = max(0.05, 0.1 * peak_abs)
        onset_candidates = np.where(np.abs(vals) >= onset_threshold)[0]
        onset_time = float(times[int(onset_candidates[0])]) if len(onset_candidates) else 0.0
        rows.append(
            {
                "peak_signed": peak_signed,
                "peak_time_s": peak_time,
                "onset_time_s": onset_time,
                "tail_signed": float(vals[-1]),
            }
        )
    return pd.DataFrame(rows)


def true_keypoints(response: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "peak_signed": response["gt_peak_signed"].astype(float),
            "peak_time_s": response["gt_peak_time_s"].astype(float),
            "onset_time_s": response["gt_onset_time_s"].astype(float),
            "tail_signed": response["gt_tail_signed"].astype(float),
        }
    )


def clip_keypoints(kp: pd.DataFrame, train_true: pd.DataFrame) -> pd.DataFrame:
    out = kp.copy()
    peak_limit = float(np.nanpercentile(np.abs(train_true["peak_signed"]), 98) * 1.45)
    tail_limit = float(np.nanpercentile(np.abs(train_true["tail_signed"]), 98) * 1.75)
    out["peak_signed"] = out["peak_signed"].clip(-peak_limit, peak_limit)
    out["tail_signed"] = out["tail_signed"].clip(-tail_limit, tail_limit)
    out["peak_time_s"] = out["peak_time_s"].clip(0.08, 2.95)
    out["onset_time_s"] = out["onset_time_s"].clip(0.0, 2.6)
    out["onset_time_s"] = np.minimum(out["onset_time_s"], out["peak_time_s"] - 0.03)
    out["onset_time_s"] = out["onset_time_s"].clip(0.0, 2.6)
    return out


def piecewise_from_keypoints(keypoints: pd.DataFrame, label_time: np.ndarray) -> np.ndarray:
    pred = np.zeros((len(keypoints), len(label_time)), dtype=np.float32)
    end_time = float(label_time[-1])
    for i, row in keypoints.reset_index(drop=True).iterrows():
        peak = float(row["peak_signed"])
        peak_t = float(row["peak_time_s"])
        onset = float(row["onset_time_s"])
        tail = float(row["tail_signed"])
        peak_t = max(peak_t, onset + 0.035)
        y = np.zeros_like(label_time, dtype=np.float32)
        rise = (label_time >= onset) & (label_time <= peak_t)
        if rise.any():
            u = (label_time[rise] - onset) / max(peak_t - onset, 1e-6)
            y[rise] = peak * smoothstep(u)
        fall = label_time > peak_t
        if fall.any():
            v = (label_time[fall] - peak_t) / max(end_time - peak_t, 1e-6)
            y[fall] = peak + (tail - peak) * smoothstep(v)
        pred[i] = y
    return pred


def rbf_scaled_by_keypoints(rbf_pred: np.ndarray, keypoints: pd.DataFrame, label_time: np.ndarray) -> np.ndarray:
    out = np.zeros_like(rbf_pred, dtype=np.float32)
    ramp = smoothstep(label_time / max(float(label_time[-1]), 1e-6))
    for i, row in keypoints.reset_index(drop=True).iterrows():
        base = rbf_pred[i].astype(np.float32)
        peak_idx = int(np.nanargmax(np.abs(base)))
        base_peak = float(base[peak_idx])
        desired_peak = float(row["peak_signed"])
        if abs(base_peak) < 0.05:
            scale = 1.0
        else:
            scale = desired_peak / base_peak
        scale = float(np.clip(scale, -2.2, 2.2))
        cur = (base * scale).astype(np.float32)
        tail_delta = float(row["tail_signed"]) - float(cur[-1])
        cur = cur + (ramp * tail_delta).astype(np.float32)
        out[i] = cur
    return out


def blend_with_rbf(rbf: np.ndarray, other: np.ndarray, alpha: float) -> np.ndarray:
    return ((1.0 - alpha) * rbf + alpha * other).astype(np.float32)


def build_keypoint_predictions(
    z: dict[str, Any],
    response: pd.DataFrame,
    candidate_idx: dict[str, int],
    keypoint_pred: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    label_time = z["label_time_rel_s"].astype(np.float32)
    rbf = z["candidate_predictions"][:, candidate_idx[RBF_MODEL], :].astype(np.float32)
    train_mask = response["split"].astype(str).to_numpy() == "train"
    train_true = true_keypoints(response).loc[train_mask].reset_index(drop=True)

    abs_kp = clip_keypoints(
        keypoint_pred[["abs_peak_signed", "abs_peak_time_s", "abs_onset_time_s", "abs_tail_signed"]].rename(
            columns={
                "abs_peak_signed": "peak_signed",
                "abs_peak_time_s": "peak_time_s",
                "abs_onset_time_s": "onset_time_s",
                "abs_tail_signed": "tail_signed",
            }
        ),
        train_true,
    )
    resid_kp = clip_keypoints(
        keypoint_pred[["resid_peak_signed", "resid_peak_time_s", "resid_onset_time_s", "resid_tail_signed"]].rename(
            columns={
                "resid_peak_signed": "peak_signed",
                "resid_peak_time_s": "peak_time_s",
                "resid_onset_time_s": "onset_time_s",
                "resid_tail_signed": "tail_signed",
            }
        ),
        train_true,
    )
    oracle_kp = true_keypoints(response)

    abs_piece = piecewise_from_keypoints(abs_kp, label_time)
    resid_piece = piecewise_from_keypoints(resid_kp, label_time)
    oracle_piece = piecewise_from_keypoints(oracle_kp, label_time)
    abs_scaled = rbf_scaled_by_keypoints(rbf, abs_kp, label_time)
    resid_scaled = rbf_scaled_by_keypoints(rbf, resid_kp, label_time)
    oracle_scaled = rbf_scaled_by_keypoints(rbf, oracle_kp, label_time)

    predictions: dict[str, np.ndarray] = {
        RBF_MODEL: rbf,
        "segment_abs_rf_piecewise": abs_piece,
        "segment_resid_rf_piecewise": resid_piece,
        "segment_abs_rf_blend_25": blend_with_rbf(rbf, abs_piece, 0.25),
        "segment_abs_rf_blend_50": blend_with_rbf(rbf, abs_piece, 0.50),
        "segment_resid_rf_blend_25": blend_with_rbf(rbf, resid_piece, 0.25),
        "segment_resid_rf_blend_50": blend_with_rbf(rbf, resid_piece, 0.50),
        "rbf_abs_keypoint_scaled": abs_scaled,
        "rbf_resid_keypoint_scaled": resid_scaled,
        "rbf_abs_keypoint_scaled_blend_50": blend_with_rbf(rbf, abs_scaled, 0.50),
        "rbf_resid_keypoint_scaled_blend_50": blend_with_rbf(rbf, resid_scaled, 0.50),
        "keypoint_oracle_piecewise": oracle_piece,
        "keypoint_oracle_rbf_scaled": oracle_scaled,
    }
    for name in [KEYPOINT_MODEL, TOP1_MODEL, *BRANCH_MODELS]:
        if name in candidate_idx:
            predictions[name] = z["candidate_predictions"][:, candidate_idx[name], :].astype(np.float32)
    keypoint_out = pd.concat([keypoint_pred.reset_index(drop=True), abs_kp.add_prefix("clipped_abs_"), resid_kp.add_prefix("clipped_resid_")], axis=1)
    return predictions, keypoint_out


def target_metrics(pred: pd.DataFrame, true: pd.DataFrame, split: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for prefix in ["abs", "resid"]:
        for target in TARGETS:
            col = f"{prefix}_{target}"
            y_true = true[target].astype(float).to_numpy()
            y_pred = pred[col].astype(float).to_numpy()
            for split_name in ["train", "val", "test"]:
                mask = split == split_name
                if not mask.any():
                    continue
                err = y_pred[mask] - y_true[mask]
                corr = float(np.corrcoef(y_true[mask], y_pred[mask])[0, 1]) if mask.sum() > 2 and np.nanstd(y_pred[mask]) > 1e-8 and np.nanstd(y_true[mask]) > 1e-8 else float("nan")
                rows.append(
                    {
                        "model_prefix": prefix,
                        "target": target,
                        "split": split_name,
                        "n_samples": int(mask.sum()),
                        "rmse": float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))),
                        "mae": float(mean_absolute_error(y_true[mask], y_pred[mask])),
                        "bias": float(np.mean(err)),
                        "corr": corr,
                    }
                )
    return pd.DataFrame(rows)


def select_candidate(metrics: pd.DataFrame) -> tuple[str, pd.DataFrame, str]:
    val = metrics[metrics["split"] == "val"].copy()
    rbf = val[val["model_name"].eq(RBF_MODEL)].iloc[0]
    candidates = val[
        (~val["model_name"].eq(RBF_MODEL))
        & (~val["model_name"].astype(str).str.contains("oracle", case=False, na=False))
    ].copy()
    candidates["meets_rmse_improvement"] = candidates["rmse_steer"] < float(rbf["rmse_steer"]) - 0.002
    candidates["meets_noninferior_physical"] = (
        (candidates["rmse_steer"] <= float(rbf["rmse_steer"]) + 0.002)
        & (
            (candidates["wrong_side_rate"] < float(rbf["wrong_side_rate"]))
            | (candidates["large_response_recall"] > float(rbf["large_response_recall"]))
            | (candidates["difficult_top20_rmse"] < float(rbf["difficult_top20_rmse"]))
        )
    )
    if candidates["meets_rmse_improvement"].any():
        selected = str(candidates[candidates["meets_rmse_improvement"]].sort_values(["rmse_steer", "wrong_side_rate"]).iloc[0]["model_name"])
        reason = "val_rmse_improvement_gt_0_002"
    elif candidates["meets_noninferior_physical"].any():
        selected = str(candidates[candidates["meets_noninferior_physical"]].sort_values(["rmse_steer", "wrong_side_rate"]).iloc[0]["model_name"])
        reason = "val_noninferior_with_physical_gain"
    else:
        selected = RBF_MODEL
        reason = "no_candidate_passed_val_gate"
    table = candidates.sort_values(["rmse_steer", "wrong_side_rate"]).copy()
    table["selected_by_val_gate"] = table["model_name"].eq(selected).astype(int)
    table["selection_reason"] = reason
    return selected, table, reason


def sample_indices(meta: pd.DataFrame, sample_ids: list[str]) -> list[int]:
    lookup = {sid: i for i, sid in enumerate(meta["sample_id"].astype(str).tolist())}
    return [lookup[sid] for sid in sample_ids if sid in lookup]


def plot_prediction_grid(
    path: Path,
    sample_ids: list[str],
    y: np.ndarray,
    y_mask: np.ndarray,
    label_time: np.ndarray,
    meta: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    selected: str,
    title: str,
) -> None:
    ids = sample_indices(meta, sample_ids)[:12]
    if not ids:
        return
    ncols = 3
    nrows = int(np.ceil(len(ids) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14.5, 3.2 * nrows), squeeze=False)
    plot_models = [
        (RBF_MODEL, "#1f77b4", "RBF/KNN", "-"),
        (selected, "#d62728", "selected", "--"),
        ("segment_resid_rf_blend_50", "#ff7f0e", "resid blend", "--"),
        ("rbf_resid_keypoint_scaled", "#2ca02c", "rbf scaled", "--"),
        ("keypoint_segment_oracle", "#111111", "oracle*", "-."),
    ]
    for ax, idx in zip(axes.ravel(), ids):
        valid = y_mask[idx] & np.isfinite(y[idx])
        ax.plot(label_time[valid], y[idx, valid], color="#000000", linewidth=1.8, label="GT")
        for model_name, color, label, style in plot_models:
            if model_name not in predictions:
                continue
            pred = predictions[model_name][idx]
            valid_pred = valid & np.isfinite(pred)
            ax.plot(label_time[valid_pred], pred[valid_pred], color=color, linestyle=style, linewidth=1.05, alpha=0.9, label=label)
        sid = str(meta.at[idx, "sample_id"])
        short = sid.split("__")[-2] if "__" in sid else sid[-12:]
        ax.set_title(short, fontsize=8)
        ax.grid(True, alpha=0.22)
        ax.axhline(0.0, color="#dddddd", linewidth=0.8)
    for ax in axes.ravel()[len(ids) :]:
        ax.axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, fontsize=8)
    fig.suptitle(title, fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_metric_summary(metrics: pd.DataFrame, selected: str, path: Path) -> None:
    test = metrics[metrics["split"] == "test"].copy()
    keep = [RBF_MODEL, selected, "segment_resid_rf_blend_50", "rbf_resid_keypoint_scaled", "keypoint_segment_oracle"]
    keep_unique: list[str] = []
    for name in keep:
        if name not in keep_unique and name in set(test["model_name"]):
            keep_unique.append(name)
    test = test[test["model_name"].isin(keep_unique)].copy()
    test["order"] = test["model_name"].map({name: i for i, name in enumerate(keep_unique)})
    test = test.sort_values("order")
    labels = [x.replace("_", " ") for x in test["model_name"]]
    fig, axes = plt.subplots(1, 4, figsize=(16.2, 4.2))
    for ax, col, title in [
        (axes[0], "rmse_steer", "RMSE"),
        (axes[1], "wrong_side_rate", "Wrong-side"),
        (axes[2], "large_response_recall", "Large recall"),
        (axes[3], "difficult_top20_rmse", "Difficult RMSE"),
    ]:
        ax.bar(np.arange(len(test)), test[col].astype(float), color="#4c78a8")
        ax.set_title(title)
        ax.set_xticks(np.arange(len(test)), labels, rotation=35, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Stage 7g keypoint/segment candidates on test", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_target_scatter(target_metric_df: pd.DataFrame, keypoint_pred: pd.DataFrame, true: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5))
    for row, prefix in enumerate(["abs", "resid"]):
        for col_idx, target in enumerate(TARGETS):
            ax = axes[row, col_idx]
            col = f"{prefix}_{target}"
            split = keypoint_pred["split"].astype(str)
            for split_name, color in [("val", "#ff7f0e"), ("test", "#d62728")]:
                mask = split.eq(split_name).to_numpy()
                ax.scatter(true.loc[mask, target], keypoint_pred.loc[mask, col], s=16, alpha=0.75, color=color, label=split_name)
            lo = float(np.nanmin([true[target].min(), keypoint_pred[col].min()]))
            hi = float(np.nanmax([true[target].max(), keypoint_pred[col].max()]))
            ax.plot([lo, hi], [lo, hi], color="#888888", linewidth=0.8)
            ax.set_title(f"{prefix} {target}", fontsize=9)
            ax.grid(True, alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle("Stage 7g keypoint target predictions", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_reports(
    metrics: pd.DataFrame,
    target_metric_df: pd.DataFrame,
    selected: str,
    selection_table: pd.DataFrame,
    gate: pd.DataFrame,
    figures: dict[str, str],
) -> None:
    test = metrics[metrics["split"] == "test"].set_index("model_name")

    def get(model: str, col: str) -> float:
        if model not in test.index or col not in test.columns:
            return float("nan")
        return float(test.loc[model, col])

    rbf_rmse = get(RBF_MODEL, "rmse_steer")
    selected_rmse = get(selected, "rmse_steer")
    oracle_rmse = get("keypoint_segment_oracle", "rmse_steer")
    gate_status = str(gate.set_index("gate_item").loc["deployable_upgrade", "status"])
    test_non_oracle = metrics[
        (metrics["split"] == "test")
        & (~metrics["model_name"].eq(RBF_MODEL))
        & (~metrics["model_name"].astype(str).str.contains("oracle", case=False, na=False))
    ].sort_values("rmse_steer")
    if len(test_non_oracle):
        test_best_model = str(test_non_oracle.iloc[0]["model_name"])
        test_best_rmse = float(test_non_oracle.iloc[0]["rmse_steer"])
        test_best_delta = float(test_non_oracle.iloc[0]["rmse_delta_vs_rbf"])
    else:
        test_best_model = "none"
        test_best_rmse = float("nan")
        test_best_delta = float("nan")
    target_text = target_metric_df[target_metric_df["split"].isin(["val", "test"])][
        ["model_prefix", "target", "split", "rmse", "mae", "bias", "corr"]
    ].to_string(index=False)
    selection_text = selection_table[["model_name", "rmse_steer", "rmse_delta_vs_rbf", "wrong_side_rate", "large_response_recall", "selected_by_val_gate"]].head(12).to_string(index=False)

    user = f"""# Stage 7g 用户查看版：keypoint/segment 车辆-only 候选 v0.1

## 这个阶段为什么做

Stage 7f 的响应类型原型只证明了 oracle 空间存在，但 validation gate 没有批准升级。Stage 7g 尝试更直接地预测响应关键点：主峰方向/幅值、峰值时间、启动时间和尾段值，再用这些关键点生成分段轨迹或校正 RBF/KNN 轨迹。

## 这个阶段检查了什么

- 只使用事件前车辆、道路/事件上下文和已有候选预测自身形态特征。
- 不使用 subject ID、session ID、test 标签、生理、脑电和连续风格。
- 关键点回归模型只在 train split 拟合。
- val 选择候选，test 只报告一次。

## 目前发现了什么

- val 选择策略：`{selected}`。
- test 上 selected RMSE={selected_rmse:.6f}，RBF/KNN RMSE={rbf_rmse:.6f}，delta={selected_rmse - rbf_rmse:+.6f}。
- keypoint/segment oracle RMSE={oracle_rmse:.6f}，只作为诊断上限。
- gate={gate_status}。
- test 上事后最好的非 oracle 候选是 `{test_best_model}`，RMSE={test_best_rmse:.6f}，delta={test_best_delta:+.6f}；但它不是 val gate 选中的策略，不能作为可部署升级结论。

## 关键点预测质量

```text
{target_text}
```

## val 策略选择表

```text
{selection_text}
```

## 哪些结果可信

可信的是：这一轮严格使用 train 拟合关键点，val 选择候选，test 最终报告，没有引入生理/脑电/连续风格或服务器信息。它可以判断“关键点/分段候选”是否比 Stage 7f 的纯响应类型原型更有前景。

## 哪些结果还不能下结论

不能把 keypoint/segment oracle 当成可部署模型；如果 validation 选择仍退回 RBF/KNN 或 test 没有稳定提升，就不能进入生理/EEG 有效性结论。

## 下一阶段是否可以继续

如果 gate 仍是 no_upgrade，下一步应复核关键点回归误差和候选生成形态，而不是继续堆 selector。只有车辆-only 候选生成和非 oracle 选择稳定后，才适合重新评估生理/EEG。

## 推荐优先查看

1. `{figures["metric_summary"]}`
2. `{figures["target_scatter"]}`
3. `{figures["fixed_predictions"]}`
4. `{figures["oracle_gain_predictions"]}`
5. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07g_keypoint_segment_candidates_v0_1/tables/stage07g_gate_table.csv`
"""
    (REPORT_DIR / "stage07g_keypoint_segment_candidates_user_summary_cn.md").write_text(user, encoding="utf-8-sig")

    tech = f"""# Stage 7g 技术报告：keypoint/segment vehicle-only candidates v0.1

## Scope

- Track: `{TRACK_ID}`
- Input trajectories: `{path_str(TRAJECTORY_NPZ)}`
- Response labels: `{path_str(RESPONSE_TABLE)}`
- No server used. Credential file not read.
- Excluded: subject ID, session ID, physio, EEG, continuous style, test labels as inputs.

## Selected Policy

- selected_policy=`{selected}`
- gate=`{gate_status}`
- test_delta_vs_rbf={selected_rmse - rbf_rmse:+.6f}

## Test Summary

- RBF/KNN RMSE={rbf_rmse:.6f}
- selected RMSE={selected_rmse:.6f}
- keypoint/segment oracle RMSE={oracle_rmse:.6f}
- best non-oracle by test only: `{test_best_model}`, RMSE={test_best_rmse:.6f}, delta={test_best_delta:+.6f}; this is diagnostic only because it was not selected by validation.

## Target Metrics

```text
{target_text}
```

## Gate

```text
{gate.to_string(index=False)}
```
"""
    (REPORT_DIR / "stage07g_keypoint_segment_candidates_v0_1_cn.md").write_text(tech, encoding="utf-8-sig")


def main() -> None:
    ensure_dirs()
    z, features, response, candidate_idx = aligned_inputs()
    y = z["y_true"].astype(np.float32)
    y_mask = z["y_mask"].astype(bool)
    label_time = z["label_time_rel_s"].astype(np.float32)
    split = response["split"].astype(str).to_numpy()
    train_mask = split == "train"
    train_idx = np.where(train_mask)[0]

    allowed, feature_audit = select_allowed_features(features)
    true_kp = true_keypoints(response)
    rbf_pred = z["candidate_predictions"][:, candidate_idx[RBF_MODEL], :].astype(np.float32)
    rbf_kp = keypoints_from_predictions(rbf_pred, y_mask, label_time)
    residual_targets = true_kp - rbf_kp

    _, abs_pred = fit_target_models(features, allowed, true_kp, train_mask, "abs", "rf")
    _, resid_delta = fit_target_models(features, allowed, residual_targets, train_mask, "delta", "extra")
    resid_pred = pd.DataFrame({"sample_id": response["sample_id"].astype(str), "split": response["split"].astype(str)})
    for target in TARGETS:
        resid_pred[f"resid_{target}"] = rbf_kp[target].astype(float).to_numpy() + resid_delta[f"delta_{target}"].astype(float).to_numpy()

    keypoint_pred = pd.concat([abs_pred, resid_pred.drop(columns=["sample_id", "split"])], axis=1)
    predictions, keypoint_out = build_keypoint_predictions(z, response, candidate_idx, keypoint_pred)
    oracle_pred, oracle_diag = oracle_prediction(
        y,
        y_mask,
        predictions,
        [
            RBF_MODEL,
            "segment_abs_rf_piecewise",
            "segment_resid_rf_piecewise",
            "segment_abs_rf_blend_25",
            "segment_abs_rf_blend_50",
            "segment_resid_rf_blend_25",
            "segment_resid_rf_blend_50",
            "rbf_abs_keypoint_scaled",
            "rbf_resid_keypoint_scaled",
            "rbf_abs_keypoint_scaled_blend_50",
            "rbf_resid_keypoint_scaled_blend_50",
        ],
    )
    predictions["keypoint_segment_oracle"] = oracle_pred

    meta = response[["sample_id", "event_uid", "subject", "session_stamp", "split"]].rename(columns={"split": SPLIT_STRATEGY})
    metrics, per_sample = evaluate_predictions(y, y_mask, label_time, meta, train_idx, predictions)
    metrics = add_reference_deltas(metrics)
    target_metric_df = target_metrics(keypoint_pred, true_kp, split)
    selected, selection_table, selection_reason = select_candidate(metrics)

    test = metrics[metrics["split"] == "test"].set_index("model_name")
    rbf_test_rmse = float(test.loc[RBF_MODEL, "rmse_steer"])
    selected_test_rmse = float(test.loc[selected, "rmse_steer"])
    selected_test = test.loc[selected]
    rbf_test = test.loc[RBF_MODEL]
    physical_gain = (
        float(selected_test["wrong_side_rate"]) < float(rbf_test["wrong_side_rate"])
        or float(selected_test["large_response_recall"]) > float(rbf_test["large_response_recall"])
        or float(selected_test["difficult_top20_rmse"]) < float(rbf_test["difficult_top20_rmse"])
    )
    gate_status = "upgrade" if selected != RBF_MODEL and selected_test_rmse < rbf_test_rmse - 1e-6 and physical_gain else "no_upgrade"
    gate = pd.DataFrame(
        [
            {"gate_item": "selected_policy", "status": selected, "evidence": f"validation gate reason: {selection_reason}"},
            {"gate_item": "deployable_upgrade", "status": gate_status, "evidence": f"test delta vs RBF {selected_test_rmse - rbf_test_rmse:+.6f}; physical_gain={physical_gain}"},
            {"gate_item": "keypoint_segment_oracle", "status": "diagnostic_only", "evidence": "oracle uses true labels and is not deployable"},
            {"gate_item": "stage08_physio_eeg_allowed", "status": "blocked", "evidence": "vehicle-only keypoint/segment candidate route is not yet stable"},
            {"gate_item": "server_used", "status": "no", "evidence": "local run only; credential file not read"},
        ]
    )

    oracle_gain = oracle_diag.assign(sample_id=response["sample_id"].astype(str).to_numpy(), split=split)
    oracle_gain["gain_over_rbf"] = oracle_gain[f"{RBF_MODEL}__sample_rmse"] - oracle_gain["oracle_sample_rmse"]
    test_ids = response.loc[split == "test", "sample_id"].astype(str).head(12).tolist()
    oracle_ids = oracle_gain[oracle_gain["split"] == "test"].sort_values("gain_over_rbf", ascending=False)["sample_id"].astype(str).head(12).tolist()

    figures = {
        "metric_summary": path_str(FIG_DIR / "stage07g_metric_summary_test.png"),
        "target_scatter": path_str(FIG_DIR / "stage07g_keypoint_target_scatter.png"),
        "fixed_predictions": path_str(FIG_DIR / "stage07g_fixed_predictions_test.png"),
        "oracle_gain_predictions": path_str(FIG_DIR / "stage07g_oracle_gain_predictions_test.png"),
    }
    plot_metric_summary(metrics, selected, Path(figures["metric_summary"]))
    plot_target_scatter(target_metric_df, keypoint_pred, true_kp, Path(figures["target_scatter"]))
    plot_prediction_grid(Path(figures["fixed_predictions"]), test_ids, y, y_mask, label_time, response, predictions, selected, "Stage 7g fixed test keypoint/segment candidates")
    plot_prediction_grid(Path(figures["oracle_gain_predictions"]), oracle_ids, y, y_mask, label_time, response, predictions, selected, "Stage 7g largest keypoint/segment oracle gains")

    feature_audit.to_csv(TABLE_DIR / "stage07g_feature_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"feature": allowed}).to_csv(TABLE_DIR / "stage07g_allowed_features.csv", index=False, encoding="utf-8-sig")
    keypoint_out.to_csv(TABLE_DIR / "stage07g_keypoint_prediction_table.csv", index=False, encoding="utf-8-sig")
    target_metric_df.to_csv(TABLE_DIR / "stage07g_keypoint_target_metrics.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(TABLE_DIR / "stage07g_candidate_metrics.csv", index=False, encoding="utf-8-sig")
    per_sample.to_csv(TABLE_DIR / "stage07g_candidate_per_sample_metrics.csv", index=False, encoding="utf-8-sig")
    selection_table.to_csv(TABLE_DIR / "stage07g_validation_selection_table.csv", index=False, encoding="utf-8-sig")
    oracle_gain.to_csv(TABLE_DIR / "stage07g_oracle_diag.csv", index=False, encoding="utf-8-sig")
    gate.to_csv(TABLE_DIR / "stage07g_gate_table.csv", index=False, encoding="utf-8-sig")

    write_reports(metrics, target_metric_df, selected, selection_table, gate, figures)

    oracle_test_rmse = float(test.loc["keypoint_segment_oracle", "rmse_steer"])
    summary = {
        "output_version": OUTPUT_VERSION,
        "track_id": TRACK_ID,
        "selected_policy": selected,
        "gate_status": gate_status,
        "rbf_test_rmse": rbf_test_rmse,
        "selected_test_rmse": selected_test_rmse,
        "selected_test_delta_vs_rbf": selected_test_rmse - rbf_test_rmse,
        "keypoint_segment_oracle_test_rmse": oracle_test_rmse,
        "keypoint_segment_oracle_delta_vs_rbf": oracle_test_rmse - rbf_test_rmse,
        "allowed_feature_count": int(len(allowed)),
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
    (LOG_DIR / "stage07g_keypoint_segment_candidates_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
