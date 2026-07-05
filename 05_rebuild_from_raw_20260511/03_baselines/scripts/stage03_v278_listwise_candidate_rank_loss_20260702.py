#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v278 listwise candidate rank loss.

v276/v277 使用候选收益回归：直接回归某个候选相对 latest 的收益。v278 改成组内排序任务：
对同一个事件的 40 个候选，先把真实 tail RMSE 转成“组内相对好坏分数”，再训练模型选择组内更好候选。

本轮目的：
- 验证“候选选择损失”是否比绝对收益回归更适合多未来候选；
- 比较 vehicle-only、vehicle+bio、vehicle+style+bio 三组排序特征；
- 生理仍只作为候选排序辅助，不做 gate / 删除样本 / residual 修正。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


ROOT = Path(r"F:\data_set_process\data_process")
REBUILD = ROOT / "05_rebuild_from_raw_20260511"
BASELINES = REBUILD / "03_baselines"

OUT = BASELINES / "v278_listwise_candidate_rank_loss_20260702"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
ZIP_PATH = BASELINES / "v278_listwise_candidate_rank_loss_20260702_pack.zip"

V277_SCRIPT = BASELINES / "scripts" / "stage03_v277_style_bio_candidate_gain_model_20260702.py"
V278_SCRIPT = BASELINES / "scripts" / "stage03_v278_listwise_candidate_rank_loss_20260702.py"
SEED = 27802


def import_module_from_path(module_name: str, path: Path):
    """按路径导入前序脚本，复用 v277 的状态特征加载和 v276 的评价函数。"""

    if not path.exists():
        raise FileNotFoundError(f"缺少依赖脚本：{path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法导入脚本：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V277 = import_module_from_path("stage03_v277_for_v278", V277_SCRIPT)
V276 = V277.V276


def ensure_dirs() -> None:
    for folder in (TABLES, FIGURES, REPORTS, LOGS):
        folder.mkdir(parents=True, exist_ok=True)


def clean_out_dir() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    ensure_dirs()
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def add_listwise_rank_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    为每个事件内部构造相对排序标签。

    tail RMSE 越低，rank_target_z 越高。这里不使用跨事件绝对误差大小，
    避免模型只学“这个事件整体难不难”，而是迫使它学习“同事件内哪个候选更好”。
    """

    out = df.copy()
    out["rank_target_z"] = np.nan
    for _, idx in out.groupby("event_uid", sort=False).groups.items():
        y = pd.to_numeric(out.loc[idx, "target_tail_rmse_v241"], errors="coerce").to_numpy(dtype=float)
        med = float(np.nanmedian(y))
        std = float(np.nanstd(y))
        if not np.isfinite(std) or std < 1e-9:
            std = 1.0
        out.loc[idx, "rank_target_z"] = (med - y) / std
    return out


def feature_sets(df: pd.DataFrame) -> Dict[str, List[str]]:
    """定义三组 listwise ranker 输入，重点比较生理是否带来增量。"""

    base = [
        "mapped_delay_ms",
        "neighbor_rank_vehicle",
        "vehicle_distance",
        "pred_pair_base_hgb",
        "pred_pair_vehicle_hgb",
    ]
    vehicle_scores = [
        col
        for col in V276.V265_SCORE_COLS
        if col in df.columns and col.startswith("score_vehicle_") and "_bio_" not in col
    ] + ["pred_gain_vehicle"]
    bio_cols = [
        "bio_distance",
        "pred_pair_bio_hgb",
        "pred_pair_vehicle_bio_hgb",
        "pred_pair_vehicle_bio_badweighted_hgb",
    ] + [col for col in V276.V265_SCORE_COLS if col in df.columns and "bio" in col] + [
        "pred_gain_vehicle_bio260_sp64",
        "bio271_distance_calibrated",
    ]
    style_cols = ["style_distance_v253_current"]
    out = {
        "listrank_vehicle": base + vehicle_scores,
        "listrank_vehicle_bio": base + vehicle_scores + bio_cols,
        "listrank_vehicle_style_bio": base + vehicle_scores + bio_cols + style_cols,
    }
    return {name: [col for col in cols if col in df.columns] for name, cols in out.items()}


def fit_predict_listrank(df: pd.DataFrame, cols: List[str]) -> Tuple[np.ndarray, int]:
    """训练组内排序标签回归器，并返回所有候选的排序分数。"""

    train_mask = df["split"].astype(str).eq("train").to_numpy()
    X = df[cols].replace([np.inf, -np.inf], np.nan)
    X = X.loc[:, X.notna().any(axis=0)].copy()
    y = pd.to_numeric(df["rank_target_z"], errors="coerce")
    sample_weight = 1.0 + np.minimum(3.0, np.abs(np.nan_to_num(y.to_numpy(dtype=float), nan=0.0)))

    model = HistGradientBoostingRegressor(
        max_iter=180,
        learning_rate=0.04,
        max_leaf_nodes=15,
        l2_regularization=0.5,
        random_state=SEED,
    )
    model.fit(X.loc[train_mask], y.loc[train_mask], sample_weight=sample_weight[train_mask])
    return model.predict(X), int(X.shape[1])


def run_rankers(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """训练三组 ranker，并生成 top candidate / threshold search / chosen configs。"""

    feature_rows: List[Dict[str, object]] = []
    pred_rows: List[pd.DataFrame] = []
    top_rows: List[pd.DataFrame] = []
    for name, cols in feature_sets(df).items():
        pred_rank_score, feature_n = fit_predict_listrank(df, cols)
        compact = df[
            [
                "event_uid",
                "split",
                "subject",
                "prototype_event_uid",
                "mapped_delay_ms",
                "target_tail_rmse_v241",
                "latest_tail_rmse_v241",
                "target_gain_vs_latest",
                "rank_target_z",
            ]
        ].copy()
        compact["feature_set"] = name
        compact["pred_rank_score"] = pred_rank_score
        pred_rows.append(compact)
        top_rows.append(V276.top_candidate_per_event(df, name, pred_rank_score))
        feature_rows.append({"feature_set": name, "feature_n": feature_n, "features": "|".join(cols)})

    top = pd.concat(top_rows, ignore_index=True)
    search, selected = V276.threshold_search(top)
    chosen = V276.choose_configs(search)
    return pd.DataFrame(feature_rows), pd.concat(pred_rows, ignore_index=True), selected, search, chosen


def decision_summary(event_table: pd.DataFrame, chosen: pd.DataFrame) -> pd.DataFrame:
    rows = V276.baseline_decision(event_table[event_table["split"].astype(str).eq("test")])
    for chosen_type in ["best_any", "best_active", "best_stable_active", "best_noharm_all", "test_best_diagnostic"]:
        sub = chosen[chosen["chosen_type"].astype(str).eq(chosen_type)]
        if sub.empty:
            continue
        row = sub.iloc[0]
        rows.append(
            {
                "source": chosen_type,
                "label": f"{row['feature_set']} threshold={row['threshold']}",
                "rmse": float(row["test_bad_top10_selected_rmse"]),
                "deployable": bool(row["deployable"]),
                "override_rate": float(row["test_bad_top10_override_rate"]),
                "val_bad_delta": float(row["val_bad_top10_delta_vs_latest"]),
                "val_all_delta": float(row["val_all_delta_vs_latest"]),
                "stable_pass": bool(row["stable_pass"]),
                "delta_vs_fixed_latest": float(row["test_bad_top10_selected_rmse"]) - float(V276.FIXED_WAIT_LATEST_BADTOP10),
                "passes_fixed_latest": bool(float(row["test_bad_top10_selected_rmse"]) < float(V276.FIXED_WAIT_LATEST_BADTOP10)),
            }
        )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:  # noqa: BLE001
        return df.to_csv(index=False)


def plot_decision(summary: pd.DataFrame) -> None:
    labels = [str(x).replace("_", "\n") for x in summary["source"]]
    values = [float(x) for x in summary["rmse"]]
    colors = ["#4C78A8" if bool(x) else "#A1765C" for x in summary["deployable"]]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(np.arange(len(values)), values, color=colors)
    ax.axhline(float(V276.FIXED_WAIT_LATEST_BADTOP10), color="#E45756", linestyle="--", linewidth=1.3, label="fixed wait-latest")
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("test bad_top10 tail RMSE")
    ax.set_title("v278: listwise candidate rank loss")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES / "v278_test_badtop10_listrank_candidate_loss.png", dpi=180)
    plt.close(fig)


def write_report(summary: pd.DataFrame, chosen: pd.DataFrame, search: pd.DataFrame, feature_audit: pd.DataFrame) -> None:
    top_val = search.sort_values(["selection_score", "val_bad_top10_selected_rmse"]).head(18)
    top_test = search[search["test_bad_top10_override_n"] > 0].sort_values("test_bad_top10_selected_rmse").head(18)
    report = f"""# v278 listwise candidate rank loss

## 本轮目的

- 把候选选择从“绝对收益回归”改成“同事件组内排序标签”。
- 比较 vehicle-only、vehicle+bio、vehicle+style+bio。
- 若生理能补足车辆锚点前信息不足，应该看到 `listrank_vehicle_bio` 或 `listrank_vehicle_style_bio` 优于 `listrank_vehicle`。
- 阈值仍只由 val 选择，test 只报告。

## test bad_top10 决策收口

{markdown_table(summary)}

## val 选择出的配置

{markdown_table(chosen[[
    "chosen_type",
    "deployable",
    "feature_set",
    "threshold",
    "val_bad_top10_delta_vs_latest",
    "val_all_delta_vs_latest",
    "test_bad_top10_selected_rmse",
    "test_bad_top10_delta_vs_latest",
    "test_bad_top10_override_rate",
    "stable_pass",
]])}

## search top by val

{markdown_table(top_val[[
    "feature_set",
    "threshold",
    "val_bad_top10_delta_vs_latest",
    "val_all_delta_vs_latest",
    "test_bad_top10_selected_rmse",
    "test_bad_top10_delta_vs_latest",
    "test_bad_top10_override_rate",
    "stable_pass",
]])}

## search top by test diagnostic

{markdown_table(top_test[[
    "feature_set",
    "threshold",
    "val_bad_top10_delta_vs_latest",
    "val_all_delta_vs_latest",
    "test_bad_top10_selected_rmse",
    "test_bad_top10_delta_vs_latest",
    "test_bad_top10_override_rate",
    "stable_pass",
]])}

## 特征组

{markdown_table(feature_audit[["feature_set", "feature_n"]])}

## 判读

- listwise rank loss 的 vehicle-only diagnostic 可以暴露更大的候选选择 headroom。
- 如果 vehicle+bio 低于 vehicle-only，说明生理能帮助组内候选排序。
- 如果 vehicle+bio 仍差于 vehicle-only，说明当前生理没有在候选选择损失层面提供稳定增量。
- deployable 配置若 test 覆盖率为 0，则不能算差样本本质改善。

## 关键图

- `figures\\v278_test_badtop10_listrank_candidate_loss.png`
"""
    (REPORTS / "v278_listwise_candidate_rank_loss_cn.md").write_text(report, encoding="utf-8")


def build_inventory() -> pd.DataFrame:
    rows = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "relative_path": str(path.relative_to(OUT)).replace("\\", "/"),
                    "bytes": int(path.stat().st_size),
                    "sha256": file_sha256(path),
                }
            )
    return pd.DataFrame(rows)


def build_zip() -> str | None:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(OUT.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=str(path.relative_to(OUT)).replace("\\", "/"))
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        return zf.testzip()


def main() -> None:
    print("[v278] listwise candidate rank loss")
    clean_out_dir()

    df, event_table, event_feature_audit, guard = V277.load_enriched_inputs()
    df = add_listwise_rank_target(df)
    feature_audit, predictions, selected, search, chosen = run_rankers(df)
    summary = decision_summary(event_table, chosen)

    write_csv(event_feature_audit, TABLES / "v278_event_feature_audit_from_v277.csv")
    write_csv(feature_audit, TABLES / "v278_listrank_feature_set_audit.csv")
    write_csv(predictions, TABLES / "v278_candidate_listrank_predictions_compact.csv")
    write_csv(selected, TABLES / "v278_selected_by_strategy.csv")
    write_csv(search, TABLES / "v278_threshold_search.csv")
    write_csv(chosen, TABLES / "v278_chosen_configs.csv")
    write_csv(summary, TABLES / "v278_decision_summary.csv")

    input_paths = [
        V278_SCRIPT,
        V277_SCRIPT,
        V277.V276_SCRIPT,
        V277.V252_SCRIPT,
        V277.STYLE_FEATURES,
        V277.V271_EVENT_CONTEXT,
        V277.V271_SCREENING,
        V276.V267_PAIR,
        V276.V267_SELECTED,
        V276.V265_EVENT_SCORES,
    ]
    hashes = pd.DataFrame(
        [
            {"path": str(path), "exists": bool(path.exists()), "sha256": file_sha256(path) if path.exists() else ""}
            for path in input_paths
        ]
    )
    write_csv(hashes, LOGS / "input_file_hashes.csv")

    plot_decision(summary)
    write_report(summary, chosen, search, feature_audit)
    write_csv(build_inventory(), LOGS / "file_inventory.csv")
    zip_bad_file = build_zip()

    deployable = summary[summary["deployable"].astype(bool) & summary["source"].astype(str).str.startswith("best")]
    diagnostic = summary[summary["source"].astype(str).eq("test_best_diagnostic")]
    bio_search = search[search["feature_set"].astype(str).str.contains("bio", na=False)]
    vehicle_diag = search[search["feature_set"].astype(str).eq("listrank_vehicle")]
    guardrail = {
        "pass": bool(zip_bad_file is None and guard.get("v267_guardrail_pass", False) and guard.get("v265_guardrail_pass", False)),
        "zip_testzip": zip_bad_file is None,
        "v267_guardrail_pass": bool(guard.get("v267_guardrail_pass", False)),
        "v265_guardrail_pass": bool(guard.get("v265_guardrail_pass", False)),
        "event_n": int(event_table["event_uid"].nunique()),
        "candidate_rows": int(len(df)),
        "feature_set_n": int(len(feature_audit)),
        "search_rows": int(len(search)),
        "chosen_rows": int(len(chosen)),
        "best_val_chosen_deployable_test_badtop10": float(deployable["rmse"].min()) if not deployable.empty else math.nan,
        "best_test_diagnostic_badtop10": float(diagnostic["rmse"].min()) if not diagnostic.empty else math.nan,
        "best_vehicle_only_diagnostic_badtop10": float(vehicle_diag["test_bad_top10_selected_rmse"].min()) if not vehicle_diag.empty else math.nan,
        "best_bio_feature_diagnostic_badtop10": float(bio_search["test_bad_top10_selected_rmse"].min()) if not bio_search.empty else math.nan,
        "fixed_wait_latest_badtop10": float(V276.FIXED_WAIT_LATEST_BADTOP10),
        "best_deployable_passes_fixed_latest": bool(not deployable.empty and float(deployable["rmse"].min()) < float(V276.FIXED_WAIT_LATEST_BADTOP10)),
        "best_diagnostic_passes_fixed_latest": bool(not diagnostic.empty and float(diagnostic["rmse"].min()) < float(V276.FIXED_WAIT_LATEST_BADTOP10)),
        "bio_beats_vehicle_diagnostic": bool(
            not bio_search.empty
            and not vehicle_diag.empty
            and float(bio_search["test_bad_top10_selected_rmse"].min()) < float(vehicle_diag["test_bad_top10_selected_rmse"].min())
        ),
    }
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    if not bool(guardrail["pass"]):
        raise AssertionError("v278 guardrail 未通过：\n" + json.dumps(guardrail, ensure_ascii=False, indent=2))

    print(f"[v278] report={REPORTS / 'v278_listwise_candidate_rank_loss_cn.md'}")
    print(f"[v278] zip={ZIP_PATH}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
