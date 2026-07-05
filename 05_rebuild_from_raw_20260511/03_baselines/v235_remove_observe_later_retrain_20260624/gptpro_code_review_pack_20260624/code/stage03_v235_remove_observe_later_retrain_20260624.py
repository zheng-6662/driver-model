#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v235 删除 observe_later_like 样本后的受控重训实验。

本脚本回答一个非常具体的问题：
如果把 v234 标出的“锚点前几秒看不出差异、随后变化很大”的
observe_later_like 样本从 train/val/test 中全部拿掉，再重新训练轻量预测层，
模型在剩余测试样本上的指标会怎样？

实验边界：
1. 不覆盖 v221/v222a/v225 的正式结果；
2. 不把 test 用于模型选择，所有超参仍只按过滤后的 validation split 选择；
3. 不做端到端底座网络重训。这里重训的是 v222a light residual/融合层，
   并新增一个同 feature schema 的 absolute Ridge 对照；
4. 底座候选曲线来自既有 v222a cache，因此本轮主要检验“删样本后校准层是否变好”，
   不是证明整个方法已经解决了后移锚点问题。
"""

from __future__ import annotations

import json
import pickle
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[3]
REBUILD_ROOT = REPO_ROOT / "05_rebuild_from_raw_20260511"
BASE_DIR = REBUILD_ROOT / "03_baselines"
SCRIPT_DIR = BASE_DIR / "scripts"
CACHE_DIR = BASE_DIR / "v222a_candidate_curve_cache_20260622"
OLD_LIGHT_DIR = BASE_DIR / "v222a_light_fusion_residual_20260622"
V234_DIR = BASE_DIR / "v234_short_observation_prediction_layer_20260624"
REMOVE_TABLE = V234_DIR / "tables" / "v234_all_split_observe_later_like_counts.csv"
FORMAL_LOCK_IMPACT_TABLE = V234_DIR / "tables" / "v234_remove_observe_later_impact_summary.csv"
OUT_DIR = BASE_DIR / "v235_remove_observe_later_retrain_20260624"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
REPORT_DIR = OUT_DIR / "reports"
MODEL_DIR = OUT_DIR / "models"
LOG_DIR = OUT_DIR / "logs"


# 这两个是 v221/v225 中用于正式对照的固定 formal lock，不参与本轮选择。
FORMAL_MODEL_LOCK = {
    "loose_main_pool": "avg_joint_focus",
    "strict_main_pool": "peak_floor_090",
}

# 新增 absolute Ridge 对照。它与 residual calibration 使用相同的 feature_matrix，
# 只在过滤后的 train split 拟合，按过滤后的 validation split 参与选择。
ABS_RIDGE_ALPHAS = [1.0, 10.0, 100.0, 1000.0]


if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import stage03_v222a_light_fusion_residual_20260622 as v222a_light  # noqa: E402


def ensure_dirs() -> None:
    """创建本轮输出目录。"""

    for path in [TABLE_DIR, FIG_DIR, REPORT_DIR, MODEL_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def clean_out_dir() -> None:
    """清理旧 v235 输出，避免旧文件和本轮结果混在一起。"""

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    ensure_dirs()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """统一用 utf-8-sig 导出，方便 Windows/Excel 直接检查中文表。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def safe_float_token(value: float) -> str:
    """把浮点超参转换为文件名安全片段。"""

    return str(value).replace(".", "p")


def assert_finite(name: str, arr: np.ndarray) -> None:
    """训练输入、标签和预测曲线都必须是有限值。"""

    if not np.isfinite(arr).all():
        bad = int(np.size(arr) - np.isfinite(arr).sum())
        raise AssertionError(f"{name} 包含非有限值：bad={bad}")


def load_remove_ids() -> Tuple[pd.DataFrame, set[str]]:
    """读取 v234 的 observe_later_like 样本清单。"""

    if not REMOVE_TABLE.exists():
        raise FileNotFoundError(f"缺少 v234 删除依据表：{REMOVE_TABLE}")
    df = pd.read_csv(REMOVE_TABLE, encoding="utf-8-sig")
    required = {"sample_id", "observe_later_like", "split"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise AssertionError("v234 observe_later 表缺少字段：" + ", ".join(missing))

    flag = df["observe_later_like"].astype(str).str.lower().isin(["true", "1", "yes"])
    remove_ids = set(df.loc[flag, "sample_id"].astype(str).tolist())
    if not remove_ids:
        raise AssertionError("v234 observe_later_like=True 样本数为 0，无法做删除实验")
    return df, remove_ids


def align_sample_manifest(pool_key: str, event_uid: np.ndarray, sample_manifest: pd.DataFrame) -> pd.DataFrame:
    """校验 sample_manifest 与 NPZ 中 event_uid 顺序一致。"""

    pool_rows = sample_manifest[sample_manifest["pool_key"].astype(str).eq(pool_key)].copy().reset_index(drop=True)
    if len(pool_rows) != len(event_uid):
        raise AssertionError(f"{pool_key} sample_manifest 行数不匹配：manifest={len(pool_rows)}, npz={len(event_uid)}")
    manifest_uid = pool_rows["event_uid"].astype(str).to_numpy()
    if not np.array_equal(manifest_uid, event_uid.astype(str)):
        raise AssertionError(f"{pool_key} sample_manifest 与 NPZ event_uid 顺序不一致")
    return pool_rows


def mask_predictions(predictions: Dict[str, np.ndarray], mask: np.ndarray) -> Dict[str, np.ndarray]:
    """按样本 mask 截取 name -> curve 预测字典。"""

    return {name: pred[mask] for name, pred in predictions.items()}


def metrics_for_mask(
    pool_key: str,
    pool_name: str,
    split_values: np.ndarray,
    true_steer: np.ndarray,
    predictions: Dict[str, np.ndarray],
    variant_types: Dict[str, str],
    mask: np.ndarray,
    splits: Iterable[str],
) -> pd.DataFrame:
    """在指定样本子集上按 split 计算指标。"""

    if not mask.any():
        return pd.DataFrame()
    return v222a_light.metrics_for_splits(
        pool_key,
        pool_name,
        split_values[mask],
        true_steer[mask],
        mask_predictions(predictions, mask),
        variant_types,
        splits=splits,
    )


def fit_abs_ridge_variants(
    cache: Dict[str, object],
    train_mask: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], List[Dict[str, object]], Dict[str, Dict[str, object]]]:
    """用过滤后的 train split 重新拟合 absolute Ridge 曲线模型。"""

    X = cache["feature_matrix"]
    y = cache["true_steer"]
    assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray)
    assert_finite("feature_matrix", X)
    assert_finite("true_steer", y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_mask])
    X_all = scaler.transform(X)

    preds: Dict[str, np.ndarray] = {}
    rows: List[Dict[str, object]] = []
    payloads: Dict[str, Dict[str, object]] = {}

    for alpha in ABS_RIDGE_ALPHAS:
        model = Ridge(alpha=alpha)
        model.fit(X_train, y[train_mask])
        pred = np.asarray(model.predict(X_all), dtype=np.float32)
        output_name = f"v235_filtered_abs_ridge_a{safe_float_token(alpha)}"
        assert_finite(output_name, pred)
        preds[output_name] = pred
        rows.append(
            {
                "output_name": output_name,
                "variant_type": "filtered_absolute_ridge",
                "base_candidate": "",
                "ridge_alpha": alpha,
                "residual_bound_rad": np.nan,
                "fit_split": "filtered_train",
                "selected_by": "filtered_validation_only",
                "test_used_for_selection": False,
                "feature_count": int(X.shape[1]),
                "component_count": 0,
            }
        )
        payloads[output_name] = {
            "model_kind": "filtered_absolute_ridge",
            "ridge_alpha": alpha,
            "scaler": scaler,
            "model": model,
            "feature_names": list(cache["feature_names"]),
            "fit_split": "filtered_train",
            "selected_by": "filtered_validation_only",
            "test_used_for_selection": False,
        }
    return preds, rows, payloads


def rename_v222a_residual_outputs(
    residual_preds: Dict[str, np.ndarray],
    residual_rows: List[Dict[str, object]],
    residual_payloads: Dict[str, Dict[str, object]],
) -> Tuple[Dict[str, np.ndarray], List[Dict[str, object]], Dict[str, Dict[str, object]]]:
    """把复用 v222a 函数产生的输出名前缀改成 v235，避免混淆结果身份。"""

    renamed_preds: Dict[str, np.ndarray] = {}
    renamed_rows: List[Dict[str, object]] = []
    renamed_payloads: Dict[str, Dict[str, object]] = {}

    for old_name, pred in residual_preds.items():
        new_name = old_name.replace("v222a_bounded_residual_", "v235_filtered_bounded_residual_", 1)
        renamed_preds[new_name] = pred

    for row in residual_rows:
        old_name = str(row["output_name"])
        new_name = old_name.replace("v222a_bounded_residual_", "v235_filtered_bounded_residual_", 1)
        new_row = dict(row)
        new_row["output_name"] = new_name
        new_row["fit_split"] = "filtered_train"
        new_row["selected_by"] = "filtered_validation_only"
        new_row["source_template_output_name"] = old_name
        renamed_rows.append(new_row)

    for old_name, payload in residual_payloads.items():
        new_name = old_name.replace("v222a_bounded_residual_", "v235_filtered_bounded_residual_", 1)
        new_payload = dict(payload)
        new_payload["fit_split"] = "filtered_train"
        new_payload["selected_by"] = "filtered_validation_only"
        new_payload["source_template_output_name"] = old_name
        renamed_payloads[new_name] = new_payload
    return renamed_preds, renamed_rows, renamed_payloads


def load_old_selected_prediction(pool_key: str, event_uid: np.ndarray) -> Tuple[str, str, np.ndarray]:
    """读取 v222a 旧 selected 输出，用来在同一过滤集上做公平对照。"""

    path = OLD_LIGHT_DIR / f"v222a_selected_predictions_{pool_key}.npz"
    if not path.exists():
        raise FileNotFoundError(f"缺少旧 v222a selected 预测：{path}")
    with np.load(path, allow_pickle=False) as data:
        old_event_uid = data["event_uid"].astype(str)
        if not np.array_equal(old_event_uid, event_uid.astype(str)):
            raise AssertionError(f"{pool_key} 旧 selected 预测与 cache event_uid 顺序不一致")
        name = str(data["selected_output_name"].astype(str)[0])
        variant_type = str(data["selected_variant_type"].astype(str)[0])
        pred = data["pred_v222a_val_selected"].astype(np.float32)
    return name, variant_type, pred


def selected_per_sample_all(
    pool_key: str,
    pool_rows: pd.DataFrame,
    split_values: np.ndarray,
    output_name: str,
    variant_type: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    remove_mask: np.ndarray,
) -> pd.DataFrame:
    """导出新 selected 输出在全部样本上的逐样本误差，并标记是否被删除。"""

    out = v222a_light.selected_per_sample_metrics(
        pool_key,
        pool_rows,
        split_values,
        output_name,
        y_true,
        y_pred,
    )
    out["variant_type"] = variant_type
    out["removed_by_v235"] = remove_mask
    out["eval_subset"] = np.where(remove_mask, "removed_holdout", "kept_filtered")
    return out


def save_selected_model(pool_key: str, output_name: str, payload: Dict[str, object]) -> Path:
    """保存过滤后 validation-selected 模型，便于复现实验。"""

    path = MODEL_DIR / f"v235_{pool_key}_filtered_selected.pkl"
    save_payload = dict(payload)
    save_payload["pool_key"] = pool_key
    save_payload["output_name"] = output_name
    save_payload["selected_by"] = "filtered_validation_only"
    save_payload["test_used_for_selection"] = False
    with path.open("wb") as f:
        pickle.dump(save_payload, f)
    return path


def build_comparison_summary(
    old_full_metrics: pd.DataFrame,
    old_filtered_metrics: pd.DataFrame,
    new_filtered_metrics: pd.DataFrame,
    new_removed_metrics: pd.DataFrame,
    formal_lock_filtered: pd.DataFrame,
    formal_lock_impact: pd.DataFrame,
) -> pd.DataFrame:
    """汇总 test split 上几个关键对照，避免把过滤收益和重训收益混在一起。"""

    rows: List[Dict[str, object]] = []
    for pool_key in sorted(new_filtered_metrics["pool_key"].dropna().astype(str).unique()):
        new_test = new_filtered_metrics[
            new_filtered_metrics["pool_key"].astype(str).eq(pool_key) & new_filtered_metrics["split"].eq("test")
        ]
        old_filtered_test = old_filtered_metrics[
            old_filtered_metrics["pool_key"].astype(str).eq(pool_key) & old_filtered_metrics["split"].eq("test")
        ]
        old_full_test = old_full_metrics[
            old_full_metrics["pool_key"].astype(str).eq(pool_key) & old_full_metrics["split"].eq("test")
        ]
        removed_test = new_removed_metrics[
            new_removed_metrics["pool_key"].astype(str).eq(pool_key) & new_removed_metrics["split"].eq("test")
        ]
        formal_test = formal_lock_filtered[
            formal_lock_filtered["pool_key"].astype(str).eq(pool_key) & formal_lock_filtered["split"].eq("test")
        ]
        v234_before = formal_lock_impact[
            formal_lock_impact["pool_key"].astype(str).eq(pool_key) & formal_lock_impact["subset"].eq("before")
        ]
        v234_after = formal_lock_impact[
            formal_lock_impact["pool_key"].astype(str).eq(pool_key)
            & formal_lock_impact["subset"].eq("after_remove_observe_later")
        ]

        if new_test.empty or old_filtered_test.empty or old_full_test.empty:
            continue
        new_row = new_test.iloc[0]
        old_filtered_row = old_filtered_test.iloc[0]
        old_full_row = old_full_test.iloc[0]
        row = {
            "pool_key": pool_key,
            "old_v222a_selected_output": old_full_row["output_name"],
            "new_v235_selected_output": new_row["output_name"],
            "old_full_test_n": int(old_full_row["n"]),
            "old_full_test_rmse": float(old_full_row["steer_rmse"]),
            "old_full_test_tail_rmse": float(old_full_row["steer_tail_rmse_1to2s"]),
            "old_filtered_test_n": int(old_filtered_row["n"]),
            "old_filtered_test_rmse": float(old_filtered_row["steer_rmse"]),
            "old_filtered_test_tail_rmse": float(old_filtered_row["steer_tail_rmse_1to2s"]),
            "new_filtered_test_n": int(new_row["n"]),
            "new_filtered_test_rmse": float(new_row["steer_rmse"]),
            "new_filtered_test_tail_rmse": float(new_row["steer_tail_rmse_1to2s"]),
            "new_filtered_test_under_rate": float(new_row["steer_severe_under_rate"]),
            "delta_new_vs_old_filtered_rmse": float(new_row["steer_rmse"] - old_filtered_row["steer_rmse"]),
            "delta_new_vs_old_filtered_tail": float(
                new_row["steer_tail_rmse_1to2s"] - old_filtered_row["steer_tail_rmse_1to2s"]
            ),
            "delta_old_filtered_vs_old_full_rmse": float(old_filtered_row["steer_rmse"] - old_full_row["steer_rmse"]),
            "delta_old_filtered_vs_old_full_tail": float(
                old_filtered_row["steer_tail_rmse_1to2s"] - old_full_row["steer_tail_rmse_1to2s"]
            ),
        }
        if not removed_test.empty:
            removed_row = removed_test.iloc[0]
            row.update(
                {
                    "new_removed_test_n": int(removed_row["n"]),
                    "new_removed_test_rmse": float(removed_row["steer_rmse"]),
                    "new_removed_test_tail_rmse": float(removed_row["steer_tail_rmse_1to2s"]),
                }
            )
        if not formal_test.empty:
            formal_row = formal_test.iloc[0]
            row.update(
                {
                    "formal_lock_filtered_output": formal_row["output_name"],
                    "formal_lock_filtered_test_rmse": float(formal_row["steer_rmse"]),
                    "formal_lock_filtered_test_tail_rmse": float(formal_row["steer_tail_rmse_1to2s"]),
                }
            )
        if not v234_before.empty and not v234_after.empty:
            row.update(
                {
                    "v234_formal_lock_before_rmse_mean": float(v234_before.iloc[0]["rmse_mean"]),
                    "v234_formal_lock_after_remove_rmse_mean": float(v234_after.iloc[0]["rmse_mean"]),
                    "v234_formal_lock_after_remove_tail_mean": float(v234_after.iloc[0]["tail_rmse_mean"]),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_test_comparison(summary: pd.DataFrame) -> List[Path]:
    """画 test RMSE/tail RMSE 对比图，直观看过滤收益和重训收益。"""

    paths: List[Path] = []
    if summary.empty:
        return paths

    metrics = [
        ("rmse", "Test RMSE", "v235_test_rmse_comparison.png"),
        ("tail", "Test tail RMSE (1-2s)", "v235_test_tail_rmse_comparison.png"),
    ]
    for metric_key, ylabel, filename in metrics:
        if metric_key == "rmse":
            cols = [
                ("old_full_test_rmse", "old full"),
                ("old_filtered_test_rmse", "old filtered"),
                ("new_filtered_test_rmse", "retrained filtered"),
            ]
        else:
            cols = [
                ("old_full_test_tail_rmse", "old full"),
                ("old_filtered_test_tail_rmse", "old filtered"),
                ("new_filtered_test_tail_rmse", "retrained filtered"),
            ]

        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        x = np.arange(len(summary))
        width = 0.24
        colors = ["#7f7f7f", "#2ca02c", "#1f77b4"]
        for offset, (col, label), color in zip([-width, 0.0, width], cols, colors):
            ax.bar(x + offset, summary[col].astype(float), width=width, label=label, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(summary["pool_key"].astype(str).tolist(), rotation=0)
        ax.set_ylabel(ylabel)
        ax.set_title("v235 remove observe_later_like retrain")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        path = FIG_DIR / filename
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def zip_outputs() -> Path:
    """打包本轮结果并验证 ZIP 可读。"""

    zip_path = OUT_DIR / "v235_remove_observe_later_retrain_pack.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(OUT_DIR.rglob("*")):
            if path == zip_path or not path.is_file():
                continue
            zf.write(path, path.relative_to(OUT_DIR))
    with zipfile.ZipFile(zip_path, "r") as zf:
        bad = zf.testzip()
    if bad is not None:
        raise AssertionError(f"ZIP 校验失败：{bad}")
    return zip_path


def make_report(
    remove_counts: pd.DataFrame,
    validation_selection: pd.DataFrame,
    selected_metrics: pd.DataFrame,
    old_filtered_metrics: pd.DataFrame,
    removed_metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    zip_path: Path,
) -> None:
    """生成中文实验报告。"""

    selected_val = validation_selection[validation_selection["validation_rank"].eq(1)].copy()
    selected_test = selected_metrics[selected_metrics["split"].eq("test")].copy()
    old_filtered_test = old_filtered_metrics[old_filtered_metrics["split"].eq("test")].copy()
    removed_test = removed_metrics[removed_metrics["split"].eq("test")].copy()

    lines: List[str] = []
    lines.append("# v235 删除 observe_later_like 样本后的受控重训报告")
    lines.append("")
    lines.append("## 结论")
    lines.append("")
    lines.append("- 本轮把 v234 标记的 observe_later_like 样本从 train/val/test 全部剔除后，重新训练 v222a light prediction layer。")
    lines.append("- 模型选择仍只使用过滤后的 validation split；test 只在选型固定后报告。")
    lines.append("- 这是校准/融合层重训，不是 v216/v218 底座候选网络的端到端重训。")
    lines.append("- 因此本轮适合判断“删掉这类样本是否让剩余任务更稳定”，不应直接当作最终正式榜单。")
    lines.append("")
    lines.append("## 删除规模")
    lines.append("")
    for row in remove_counts.sort_values(["pool_key", "split"]).itertuples(index=False):
        lines.append(
            f"- {row.pool_key}/{row.split}: 原始 {row.original_n}，删除 {row.removed_n}，保留 {row.kept_n}，"
            f"删除比例 {row.removed_ratio:.3f}"
        )
    lines.append("")
    lines.append("## Validation-selected 模型")
    lines.append("")
    for row in selected_val.itertuples(index=False):
        test_row = selected_test[selected_test["pool_key"].eq(row.pool_key)].iloc[0]
        old_row = old_filtered_test[old_filtered_test["pool_key"].eq(row.pool_key)].iloc[0]
        lines.append(
            f"- {row.pool_key}: `{row.output_name}`，variant={row.variant_type}，"
            f"filtered val score={row.selection_score:.6f}；"
            f"filtered test RMSE {old_row.steer_rmse:.6f} -> {test_row.steer_rmse:.6f}，"
            f"tail {old_row.steer_tail_rmse_1to2s:.6f} -> {test_row.steer_tail_rmse_1to2s:.6f}"
        )
    lines.append("")
    lines.append("## 关键对照")
    lines.append("")
    if comparison.empty:
        lines.append("- comparison summary 为空，请检查上游指标表。")
    else:
        for row in comparison.itertuples(index=False):
            lines.append(
                f"- {row.pool_key}: 旧模型 full test RMSE={row.old_full_test_rmse:.6f}；"
                f"旧模型删除后同一 test 子集 RMSE={row.old_filtered_test_rmse:.6f}；"
                f"删除后重训 RMSE={row.new_filtered_test_rmse:.6f}；"
                f"重训相对旧过滤子集 delta={row.delta_new_vs_old_filtered_rmse:+.6f}"
            )
    lines.append("")
    lines.append("## 被删除样本上的诊断")
    lines.append("")
    if removed_test.empty:
        lines.append("- test split 中没有 removed_holdout 样本。")
    else:
        for row in removed_test.itertuples(index=False):
            lines.append(
                f"- {row.pool_key}: removed test n={row.n}，重训模型 RMSE={row.steer_rmse:.6f}，"
                f"tail={row.steer_tail_rmse_1to2s:.6f}，under={row.steer_severe_under_rate:.6f}"
            )
    lines.append("")
    lines.append("## 输出")
    lines.append("")
    lines.append("- `tables/v235_comparison_summary.csv`：主对照表。")
    lines.append("- `tables/v235_selected_metrics_filtered.csv`：删除后重训模型在保留样本上的指标。")
    lines.append("- `tables/v235_old_selected_metrics_filtered.csv`：旧 v222a selected 模型在同一保留样本上的指标。")
    lines.append("- `tables/v235_selected_metrics_removed_holdout.csv`：删除后重训模型在被删除样本上的诊断指标。")
    lines.append("- `figures/v235_test_rmse_comparison.png` 与 `figures/v235_test_tail_rmse_comparison.png`：test 对比图。")
    lines.append(f"- ZIP：`{zip_path.name}`")
    lines.append("")

    (REPORT_DIR / "v235_remove_observe_later_retrain_cn.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    """主流程：读取 v234 删除清单，过滤样本，重训轻量层，导出对照结果。"""

    clean_out_dir()
    if not CACHE_DIR.exists():
        raise FileNotFoundError(f"缺少 v222a cache：{CACHE_DIR}")
    sample_manifest_path = CACHE_DIR / "sample_manifest.csv"
    if not sample_manifest_path.exists():
        raise FileNotFoundError(f"缺少 sample_manifest：{sample_manifest_path}")
    old_metrics_path = OLD_LIGHT_DIR / "tables" / "v222a_selected_metrics.csv"
    if not old_metrics_path.exists():
        raise FileNotFoundError(f"缺少旧 v222a selected metrics：{old_metrics_path}")

    remove_df, remove_ids = load_remove_ids()
    sample_manifest = pd.read_csv(sample_manifest_path, encoding="utf-8-sig")
    old_full_metrics = pd.read_csv(old_metrics_path, encoding="utf-8-sig")
    formal_lock_impact = (
        pd.read_csv(FORMAL_LOCK_IMPACT_TABLE, encoding="utf-8-sig")
        if FORMAL_LOCK_IMPACT_TABLE.exists()
        else pd.DataFrame()
    )

    all_remove_counts: List[pd.DataFrame] = []
    all_validation_metrics: List[pd.DataFrame] = []
    all_validation_ranked: List[pd.DataFrame] = []
    all_selected_metrics: List[pd.DataFrame] = []
    all_removed_metrics: List[pd.DataFrame] = []
    all_old_filtered_metrics: List[pd.DataFrame] = []
    all_old_removed_metrics: List[pd.DataFrame] = []
    all_formal_lock_filtered: List[pd.DataFrame] = []
    all_per_sample: List[pd.DataFrame] = []
    all_feature_audits: List[pd.DataFrame] = []
    all_weight_rows: List[pd.DataFrame] = []
    all_model_rows: List[Dict[str, object]] = []
    selected_model_rows: List[Dict[str, object]] = []

    for cache_path in sorted(CACHE_DIR.glob("candidate_predictions_*.npz")):
        cache = v222a_light.load_pool_cache(cache_path)
        pool_key = str(cache["pool_key"])
        event_uid = cache["event_uid"]
        split_values = cache["split"]
        true_steer = cache["true_steer"]
        assert isinstance(event_uid, np.ndarray)
        assert isinstance(split_values, np.ndarray)
        assert isinstance(true_steer, np.ndarray)

        pool_rows = align_sample_manifest(pool_key, event_uid, sample_manifest)
        pool_name = str(pool_rows["pool_name"].iloc[0]) if "pool_name" in pool_rows.columns else pool_key

        remove_mask = np.asarray([str(uid) in remove_ids for uid in event_uid.astype(str)], dtype=bool)
        kept_mask = ~remove_mask
        train_mask = split_values.astype(str) == "train"
        val_mask = split_values.astype(str) == "val"
        filtered_train_mask = train_mask & kept_mask
        filtered_val_mask = val_mask & kept_mask
        if not filtered_train_mask.any() or not filtered_val_mask.any():
            raise AssertionError(f"{pool_key} 删除后缺少 train 或 val 样本")

        count_rows = []
        for split_name in ["train", "val", "test", "all"]:
            split_mask = np.ones(len(split_values), dtype=bool) if split_name == "all" else split_values.astype(str) == split_name
            original_n = int(split_mask.sum())
            removed_n = int((split_mask & remove_mask).sum())
            kept_n = int((split_mask & kept_mask).sum())
            count_rows.append(
                {
                    "pool_key": pool_key,
                    "pool_name": pool_name,
                    "split": split_name,
                    "original_n": original_n,
                    "removed_n": removed_n,
                    "kept_n": kept_n,
                    "removed_ratio": removed_n / original_n if original_n else np.nan,
                }
            )
        remove_counts = pd.DataFrame(count_rows)
        all_remove_counts.append(remove_counts)

        feature_audit = v222a_light.assert_feature_schema(list(cache["feature_names"]), pool_key)
        all_feature_audits.append(feature_audit)

        pred_map = v222a_light.candidate_prediction_map(cache)
        baseline_preds = {name: pred_map[name] for name in v222a_light.FORMAL_CANDIDATES}
        baseline_variant_types = {name: "fixed_formal_baseline_from_old_cache" for name in baseline_preds}

        convex_pred, convex_weights, convex_payload = v222a_light.fit_convex_formal_blend(
            cache,
            pred_map,
            filtered_train_mask,
        )
        convex_name = "v235_filtered_convex_formal_blend"
        convex_weights.insert(0, "pool_key", pool_key)
        convex_weights.insert(1, "output_name", convex_name)
        all_weight_rows.append(convex_weights)
        convex_payload = dict(convex_payload)
        convex_payload["fit_split"] = "filtered_train"
        convex_payload["selected_by"] = "filtered_validation_only"

        residual_preds, residual_rows, residual_payloads = v222a_light.fit_bounded_residual_variants(
            cache,
            pred_map,
            filtered_train_mask,
        )
        residual_preds, residual_rows, residual_payloads = rename_v222a_residual_outputs(
            residual_preds,
            residual_rows,
            residual_payloads,
        )

        abs_preds, abs_rows, abs_payloads = fit_abs_ridge_variants(cache, filtered_train_mask)

        learned_preds: Dict[str, np.ndarray] = {convex_name: convex_pred}
        learned_preds.update(residual_preds)
        learned_preds.update(abs_preds)
        learned_variant_types: Dict[str, str] = {convex_name: "filtered_convex_formal_blend"}
        learned_variant_types.update({name: "filtered_bounded_residual" for name in residual_preds})
        learned_variant_types.update({name: "filtered_absolute_ridge" for name in abs_preds})

        selection_preds: Dict[str, np.ndarray] = {}
        selection_preds.update(baseline_preds)
        selection_preds.update(learned_preds)
        selection_variant_types: Dict[str, str] = {}
        selection_variant_types.update(baseline_variant_types)
        selection_variant_types.update(learned_variant_types)

        validation_metrics = metrics_for_mask(
            pool_key,
            pool_name,
            split_values,
            true_steer,
            selection_preds,
            selection_variant_types,
            kept_mask,
            splits=["train", "val"],
        )
        validation_ranked = v222a_light.select_by_validation(validation_metrics)
        selected = validation_ranked[validation_ranked["validation_rank"].eq(1)].iloc[0]
        selected_name = str(selected["output_name"])
        selected_variant_type = str(selected["variant_type"])
        selected_pred = selection_preds[selected_name]

        selected_metrics = metrics_for_mask(
            pool_key,
            pool_name,
            split_values,
            true_steer,
            {selected_name: selected_pred},
            {selected_name: selected_variant_type},
            kept_mask,
            splits=["all", "train", "val", "test"],
        )
        selected_metrics["selected_by"] = "filtered_validation_only"
        selected_metrics["test_used_for_selection"] = False

        removed_metrics = metrics_for_mask(
            pool_key,
            pool_name,
            split_values,
            true_steer,
            {selected_name: selected_pred},
            {selected_name: selected_variant_type},
            remove_mask,
            splits=["all", "train", "val", "test"],
        )
        if not removed_metrics.empty:
            removed_metrics["eval_subset"] = "removed_holdout"
            removed_metrics["selected_by"] = "filtered_validation_only"
            removed_metrics["test_used_for_selection"] = False

        old_name, old_variant_type, old_pred = load_old_selected_prediction(pool_key, event_uid)
        old_filtered_metrics = metrics_for_mask(
            pool_key,
            pool_name,
            split_values,
            true_steer,
            {old_name: old_pred},
            {old_name: f"old_v222a_selected_{old_variant_type}"},
            kept_mask,
            splits=["all", "train", "val", "test"],
        )
        old_filtered_metrics["source"] = "old_v222a_selected_on_filtered_subset"
        old_removed_metrics = metrics_for_mask(
            pool_key,
            pool_name,
            split_values,
            true_steer,
            {old_name: old_pred},
            {old_name: f"old_v222a_selected_{old_variant_type}"},
            remove_mask,
            splits=["all", "train", "val", "test"],
        )
        if not old_removed_metrics.empty:
            old_removed_metrics["source"] = "old_v222a_selected_on_removed_subset"

        formal_lock_name = FORMAL_MODEL_LOCK.get(pool_key, "")
        if formal_lock_name and formal_lock_name in pred_map:
            formal_lock_filtered = metrics_for_mask(
                pool_key,
                pool_name,
                split_values,
                true_steer,
                {formal_lock_name: pred_map[formal_lock_name]},
                {formal_lock_name: "formal_lock_from_old_cache"},
                kept_mask,
                splits=["all", "train", "val", "test"],
            )
            all_formal_lock_filtered.append(formal_lock_filtered)

        per_sample = selected_per_sample_all(
            pool_key,
            pool_rows,
            split_values,
            selected_name,
            selected_variant_type,
            true_steer,
            selected_pred,
            remove_mask,
        )

        selected_cache_path = OUT_DIR / f"v235_filtered_selected_predictions_{pool_key}.npz"
        np.savez_compressed(
            selected_cache_path,
            pred_v235_filtered_selected=selected_pred.astype(np.float32),
            selected_output_name=np.array([selected_name], dtype="U180"),
            selected_variant_type=np.array([selected_variant_type], dtype="U80"),
            true_steer=true_steer.astype(np.float32),
            split=split_values.astype("U16"),
            array_index=cache["array_index"],
            event_uid=event_uid.astype("U180"),
            removed_by_v235=remove_mask,
        )

        payload: Dict[str, object]
        if selected_name == convex_name:
            payload = convex_payload
        elif selected_name in residual_payloads:
            payload = residual_payloads[selected_name]
        elif selected_name in abs_payloads:
            payload = abs_payloads[selected_name]
        else:
            payload = {
                "model_kind": "fixed_formal_baseline_from_old_cache",
                "base_candidate": selected_name,
                "fit_split": "none",
                "selected_by": "filtered_validation_only",
                "test_used_for_selection": False,
            }
        model_path = save_selected_model(pool_key, selected_name, payload)

        selected_model_rows.append(
            {
                "pool_key": pool_key,
                "pool_name": pool_name,
                "output_name": selected_name,
                "variant_type": selected_variant_type,
                "model_path": str(model_path.relative_to(REPO_ROOT)),
                "selected_cache_path": str(selected_cache_path.relative_to(REPO_ROOT)),
            }
        )

        all_model_rows.append(
            {
                "pool_key": pool_key,
                "pool_name": pool_name,
                "output_name": convex_name,
                "variant_type": "filtered_convex_formal_blend",
                "base_candidate": "",
                "ridge_alpha": np.nan,
                "residual_bound_rad": np.nan,
                "fit_split": "filtered_train",
                "selected_by": "filtered_validation_only",
                "test_used_for_selection": False,
                "feature_count": 0,
                "component_count": len(v222a_light.FORMAL_CANDIDATES),
            }
        )
        for row in residual_rows + abs_rows:
            row = dict(row)
            row["pool_key"] = pool_key
            row["pool_name"] = pool_name
            all_model_rows.append(row)

        all_validation_metrics.append(validation_metrics)
        all_validation_ranked.append(validation_ranked)
        all_selected_metrics.append(selected_metrics)
        all_removed_metrics.append(removed_metrics)
        all_old_filtered_metrics.append(old_filtered_metrics)
        all_old_removed_metrics.append(old_removed_metrics)
        all_per_sample.append(per_sample)

    remove_counts_all = pd.concat(all_remove_counts, ignore_index=True)
    validation_metrics_all = pd.concat(all_validation_metrics, ignore_index=True)
    validation_ranked_all = pd.concat(all_validation_ranked, ignore_index=True)
    selected_metrics_all = pd.concat(all_selected_metrics, ignore_index=True)
    removed_metrics_all = pd.concat([df for df in all_removed_metrics if not df.empty], ignore_index=True)
    old_filtered_metrics_all = pd.concat(all_old_filtered_metrics, ignore_index=True)
    old_removed_metrics_all = pd.concat([df for df in all_old_removed_metrics if not df.empty], ignore_index=True)
    formal_lock_filtered_all = (
        pd.concat(all_formal_lock_filtered, ignore_index=True) if all_formal_lock_filtered else pd.DataFrame()
    )
    per_sample_all = pd.concat(all_per_sample, ignore_index=True)
    feature_audit_all = pd.concat(all_feature_audits, ignore_index=True)
    blend_weights_all = pd.concat(all_weight_rows, ignore_index=True)
    model_manifest = pd.DataFrame(all_model_rows)
    selected_models = pd.DataFrame(selected_model_rows)

    comparison = build_comparison_summary(
        old_full_metrics,
        old_filtered_metrics_all,
        selected_metrics_all,
        removed_metrics_all,
        formal_lock_filtered_all,
        formal_lock_impact,
    )
    figure_paths = plot_test_comparison(comparison)

    leakage_guard = pd.DataFrame(
        [
            {
                "check_name": "feature_schema_forbidden_tokens",
                "status": "pass" if feature_audit_all["guard_status"].eq("pass").all() else "fail",
                "detail": "feature_matrix 不含 split/subject/true/oracle/RMSE 等禁用字段",
            },
            {
                "check_name": "remove_id_source",
                "status": "pass",
                "detail": f"删除样本来自 {REMOVE_TABLE.relative_to(REPO_ROOT)}",
            },
            {
                "check_name": "selection_uses_filtered_validation_only",
                "status": "pass",
                "detail": "过滤后模型只按 kept val split 排序；test 只在 selected 后报告",
            },
            {
                "check_name": "train_only_fit_after_removal",
                "status": "pass",
                "detail": "所有 learned 变体只在 kept train split 拟合",
            },
        ]
    )

    write_csv(remove_df, TABLE_DIR / "v235_remove_id_source_from_v234.csv")
    write_csv(remove_counts_all, TABLE_DIR / "v235_removed_sample_counts.csv")
    write_csv(validation_metrics_all, TABLE_DIR / "v235_validation_metrics_pre_rank_filtered.csv")
    write_csv(validation_ranked_all, TABLE_DIR / "v235_validation_selection_filtered.csv")
    write_csv(selected_metrics_all, TABLE_DIR / "v235_selected_metrics_filtered.csv")
    write_csv(removed_metrics_all, TABLE_DIR / "v235_selected_metrics_removed_holdout.csv")
    write_csv(old_filtered_metrics_all, TABLE_DIR / "v235_old_selected_metrics_filtered.csv")
    write_csv(old_removed_metrics_all, TABLE_DIR / "v235_old_selected_metrics_removed_holdout.csv")
    write_csv(formal_lock_filtered_all, TABLE_DIR / "v235_formal_lock_metrics_filtered.csv")
    write_csv(per_sample_all, TABLE_DIR / "v235_selected_per_sample_metrics_all.csv")
    write_csv(blend_weights_all, TABLE_DIR / "v235_convex_blend_weights.csv")
    write_csv(model_manifest, TABLE_DIR / "v235_model_manifest.csv")
    write_csv(selected_models, TABLE_DIR / "v235_selected_model_paths.csv")
    write_csv(feature_audit_all, TABLE_DIR / "v235_feature_schema_audit.csv")
    write_csv(leakage_guard, TABLE_DIR / "v235_leakage_guard_result.csv")
    write_csv(comparison, TABLE_DIR / "v235_comparison_summary.csv")

    manifest = {
        "stage": "v235_remove_observe_later_retrain",
        "created_by": Path(__file__).name,
        "cache_dir": str(CACHE_DIR),
        "old_light_dir": str(OLD_LIGHT_DIR),
        "remove_table": str(REMOVE_TABLE),
        "output_dir": str(OUT_DIR),
        "remove_unique_ids": int(len(remove_ids)),
        "formal_model_lock": FORMAL_MODEL_LOCK,
        "base_note": "底座候选曲线来自既有 v222a cache；本轮重训 light residual/convex/absolute ridge 层。",
        "abs_ridge_alphas": ABS_RIDGE_ALPHAS,
        "residual_bases": v222a_light.RESIDUAL_BASES,
        "residual_bounds": v222a_light.RESIDUAL_BOUNDS,
        "selection_score": {
            "formula": "steer_rmse + tail_weight * steer_tail_rmse_1to2s + under_weight * strong_response_severe_under_rate",
            "tail_weight": v222a_light.SELECTION_TAIL_WEIGHT,
            "under_weight": v222a_light.SELECTION_UNDER_WEIGHT,
            "split": "filtered_val",
        },
        "test_used_for_selection": False,
        "selected_models": selected_model_rows,
        "figures": [str(path.relative_to(OUT_DIR)) for path in figure_paths],
    }
    (LOG_DIR / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    zip_path = zip_outputs()
    make_report(
        remove_counts_all,
        validation_ranked_all,
        selected_metrics_all,
        old_filtered_metrics_all,
        removed_metrics_all,
        comparison,
        zip_path,
    )
    zip_path = zip_outputs()

    print("v235 remove observe_later retrain finished.")
    print(f"output_dir={OUT_DIR}")
    print(f"comparison={TABLE_DIR / 'v235_comparison_summary.csv'}")
    print(f"zip={zip_path}")


if __name__ == "__main__":
    main()
