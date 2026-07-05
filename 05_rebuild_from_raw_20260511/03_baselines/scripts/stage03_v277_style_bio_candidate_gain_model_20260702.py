#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v277 style + calibrated physiology candidate gain model.

本轮承接 v276：不再把生理当直接 selector / override，而是在 v267 full vehicle top40
候选池内学习“某个候选轨迹是否比 latest 更好”。v277 进一步加入两类状态信息：

1. v253a 当前任务口径下重建的 last60_guard3 驾驶风格特征；
2. v271 个体/recording 校准后的 raw physiology summary / PCA 筛选特征。

关键边界：
- 不删样本；
- 不做 v222a gate；
- 不做轻量 residual 修正；
- threshold 只由 val 选择，test 只报告；
- test_best_diagnostic 只作为事后诊断，不作为可部署结果。
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
from typing import Dict, Iterable, List, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"F:\data_set_process\data_process")
REBUILD = ROOT / "05_rebuild_from_raw_20260511"
BASELINES = REBUILD / "03_baselines"

OUT = BASELINES / "v277_style_bio_candidate_gain_model_20260702"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
ZIP_PATH = BASELINES / "v277_style_bio_candidate_gain_model_20260702_pack.zip"

V276_SCRIPT = BASELINES / "scripts" / "stage03_v276_bio_assisted_candidate_gain_model_20260702.py"
V252_SCRIPT = BASELINES / "scripts" / "stage03_v252_input_similarity_future_divergence_20260701.py"
STYLE_FEATURES = BASELINES / "v253_state_signal_disambiguation_audit_20260701" / "tables" / "v253a_current_style_features_last60_guard3.csv"
V271_EVENT_CONTEXT = BASELINES / "v271_calibrated_raw_physio_state_20260702" / "tables" / "v271_event_context_table.csv"
V271_SCREENING = BASELINES / "v271_calibrated_raw_physio_state_20260702" / "tables" / "v271_raw_feature_screening_train_only.csv"
V277_SCRIPT = BASELINES / "scripts" / "stage03_v277_style_bio_candidate_gain_model_20260702.py"

STYLE_FEATURE_CAP = 96
BIO271_FEATURE_CAP = 96


def import_module_from_path(module_name: str, path: Path):
    """按路径导入前序脚本，复用已经锁定的数据构造和评价函数。"""

    if not path.exists():
        raise FileNotFoundError(f"缺少依赖脚本：{path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法导入脚本：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V276 = import_module_from_path("stage03_v276_for_v277", V276_SCRIPT)
V252 = import_module_from_path("stage03_v252_for_v277", V252_SCRIPT)


def ensure_dirs() -> None:
    """创建 v277 输出目录。"""

    for folder in (TABLES, FIGURES, REPORTS, LOGS):
        folder.mkdir(parents=True, exist_ok=True)


def clean_out_dir() -> None:
    """只清理 v277 自己的输出。"""

    if OUT.exists():
        shutil.rmtree(OUT)
    ensure_dirs()
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """CSV 统一使用 utf-8-sig，便于 Windows/Excel 打开。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def file_sha256(path: Path) -> str:
    """计算文件 SHA256，用于输入追踪。"""

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finite_numeric_cols(df: pd.DataFrame, prefixes: Tuple[str, ...], skip_substrings: Tuple[str, ...]) -> List[str]:
    """选择指定前缀下的数值列，并排除明显的元数据字段。"""

    cols: List[str] = []
    for col in df.columns:
        if not any(col.startswith(prefix) for prefix in prefixes):
            continue
        if any(item in col for item in skip_substrings):
            continue
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            cols.append(col)
    return cols


def standardize_event_features(
    table: pd.DataFrame,
    cols: List[str],
    prefix: str,
    cap: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    对事件级状态特征做 train-only 标准化。

    这里的特征是 query 事件自己的状态，不使用未来标签；缺失值用 train median 填充，
    缺失比例超过 1% 时额外加入 missing indicator。
    """

    base = table[["event_uid", "split"] + cols].drop_duplicates("event_uid").copy()
    train_mask = base["split"].astype(str).eq("train").to_numpy()
    feature_arrays: List[Tuple[str, np.ndarray]] = []
    audit_rows: List[Dict[str, object]] = []

    for col in cols:
        raw = pd.to_numeric(base[col], errors="coerce").to_numpy(dtype=float)
        train_values = raw[train_mask & np.isfinite(raw)]
        if train_values.size < 20:
            continue
        med = float(np.median(train_values))
        std = float(np.std(train_values))
        if not np.isfinite(std) or std < 1e-9:
            continue
        missing = ~np.isfinite(raw)
        z = (np.where(missing, med, raw) - med) / std
        feature_arrays.append((f"{prefix}{col}", z.astype(np.float32)))
        audit_rows.append(
            {
                "source_feature": col,
                "standardized_feature": f"{prefix}{col}",
                "train_finite_n": int(train_values.size),
                "train_median": med,
                "train_std": std,
                "all_missing_rate": float(missing.mean()),
                "missing_indicator_added": bool(missing.mean() > 0.01),
            }
        )
        if missing.mean() > 0.01:
            feature_arrays.append((f"{prefix}{col}__missing", missing.astype(np.float32)))

    feature_arrays = feature_arrays[:cap]
    out = base[["event_uid"]].copy()
    for name, arr in feature_arrays:
        out[name] = arr
    return out, pd.DataFrame(audit_rows)


def load_style_delay0_features() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    把 v253a 的 7002 行 delay-level 风格特征映射回 1167 个事件。

    v253a 特征按 v252 manifest row_index 对齐；v277 只使用 delay=0 的事件状态，
    保持和 v271/v276 的事件级输入口径一致。
    """

    loaded = V252.load_fixed_inputs()
    manifest = loaded["data"].manifest.copy().reset_index().rename(columns={"index": "row_index"})
    style = pd.read_csv(STYLE_FEATURES, encoding="utf-8-sig", low_memory=False)
    style0 = manifest[manifest["delay_ms"].eq(0)][["event_uid", "row_index", "split"]].merge(
        style, on="row_index", how="left", validate="one_to_one"
    )
    skip = ("window", "start_s", "end_s", "uses_post", "overlaps", "file_exists", "status")
    style_cols = finite_numeric_cols(style0, prefixes=("style_",), skip_substrings=skip)
    style_z, audit = standardize_event_features(style0, style_cols, "stylez__", STYLE_FEATURE_CAP)
    audit.insert(0, "feature_block", "style_last60_guard3_delay0")
    return style_z, audit


def load_bio271_features() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    读取 v271 已筛选的校准 raw 生理特征。

    使用 v271 train-only screening 的前若干特征，避免把 500 多个生理列无选择地塞入模型。
    """

    v271 = pd.read_csv(V271_EVENT_CONTEXT, encoding="utf-8-sig", low_memory=False)
    screening = pd.read_csv(V271_SCREENING, encoding="utf-8-sig")
    bio_cols = [col for col in screening["feature"].astype(str).tolist() if col in v271.columns][:BIO271_FEATURE_CAP]
    if "raw_physio_ok" in v271.columns:
        bio_cols = ["raw_physio_ok"] + bio_cols
    bio_z, audit = standardize_event_features(v271[["event_uid", "split"] + bio_cols].copy(), bio_cols, "bio271z__", BIO271_FEATURE_CAP)
    audit.insert(0, "feature_block", "v271_calibrated_raw_physio_screened")
    return bio_z, audit


def add_pair_distance(pair_df: pd.DataFrame, event_z: pd.DataFrame, prefix: str, out_col: str) -> None:
    """
    为每个 query-prototype pair 计算状态空间距离。

    距离只用当前事件和候选 prototype 的锚点前状态特征；不使用未来轨迹。
    """

    cols = [col for col in event_z.columns if col.startswith(prefix)]
    uid_to_pos = {uid: i for i, uid in enumerate(event_z["event_uid"].astype(str).tolist())}
    mat = event_z[cols].to_numpy(dtype=np.float32)
    query_pos = pair_df["event_uid"].astype(str).map(uid_to_pos)
    proto_pos = pair_df["prototype_event_uid"].astype(str).map(uid_to_pos)
    out = np.full(len(pair_df), np.nan, dtype=np.float32)
    ok = query_pos.notna().to_numpy() & proto_pos.notna().to_numpy()
    if ok.any() and len(cols) > 0:
        q = query_pos[ok].astype(int).to_numpy()
        p = proto_pos[ok].astype(int).to_numpy()
        diff = mat[q] - mat[p]
        out[ok] = np.sqrt(np.nanmean(diff * diff, axis=1))
    pair_df[out_col] = out


def load_enriched_inputs() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """加载 v276 基础候选表，并加入风格、生理事件状态和 pair 距离。"""

    df, event_table, guard = V276.load_inputs()
    style_z, style_audit = load_style_delay0_features()
    bio_z, bio_audit = load_bio271_features()

    df = df.merge(style_z, on="event_uid", how="left", validate="many_to_one")
    df = df.merge(bio_z, on="event_uid", how="left", validate="many_to_one")
    add_pair_distance(df, style_z, "stylez__", "style_distance_v253_current")
    add_pair_distance(df, bio_z, "bio271z__", "bio271_distance_calibrated")

    feature_audit = pd.concat([style_audit, bio_audit], ignore_index=True)
    guard.update(
        {
            "style_event_n": int(style_z["event_uid"].nunique()),
            "style_feature_n": int(sum(col.startswith("stylez__") for col in style_z.columns)),
            "bio271_event_n": int(bio_z["event_uid"].nunique()),
            "bio271_feature_n": int(sum(col.startswith("bio271z__") for col in bio_z.columns)),
        }
    )
    return df, event_table, feature_audit, guard


def feature_sets(df: pd.DataFrame) -> Dict[str, List[str]]:
    """构造 v277 候选收益模型的特征组。"""

    base_vehicle = [
        "mapped_delay_ms",
        "neighbor_rank_vehicle",
        "vehicle_distance",
        "pred_pair_base_hgb",
        "pred_pair_vehicle_hgb",
    ]
    v265_cols = [col for col in V276.V265_SCORE_COLS if col in df.columns]
    vehicle_scores = [col for col in v265_cols if col.startswith("score_vehicle_") and "_bio_" not in col] + ["pred_gain_vehicle"]
    style_query = [col for col in df.columns if col.startswith("stylez__")][:STYLE_FEATURE_CAP]
    bio_query = [col for col in df.columns if col.startswith("bio271z__")][:BIO271_FEATURE_CAP]
    out = {
        "candidate_vehicle": base_vehicle + vehicle_scores,
        "candidate_vehicle_style_dist": base_vehicle + ["style_distance_v253_current"],
        "candidate_vehicle_bio271_dist": base_vehicle + ["bio271_distance_calibrated"],
        "candidate_vehicle_style_bio_dist": base_vehicle + ["style_distance_v253_current", "bio271_distance_calibrated", "bio_distance"],
        "candidate_vehicle_style_query": base_vehicle + ["style_distance_v253_current"] + style_query,
        "candidate_vehicle_style_bio_query": base_vehicle
        + ["style_distance_v253_current", "bio271_distance_calibrated", "bio_distance"]
        + style_query
        + bio_query,
    }
    return {name: [col for col in cols if col in df.columns] for name, cols in out.items()}


def build_model_outputs(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """训练各特征组的候选收益模型，并完成 val 阈值选择。"""

    feature_rows: List[Dict[str, object]] = []
    pred_rows: List[pd.DataFrame] = []
    top_rows: List[pd.DataFrame] = []
    for name, cols in feature_sets(df).items():
        pred_gain = V276.fit_predict_gain(df, cols)
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
            ]
        ].copy()
        compact["feature_set"] = name
        compact["pred_gain_vs_latest"] = pred_gain
        pred_rows.append(compact)
        top_rows.append(V276.top_candidate_per_event(df, name, pred_gain))
        feature_rows.append({"feature_set": name, "feature_n": int(len(cols)), "features": "|".join(cols)})

    predictions = pd.concat(pred_rows, ignore_index=True)
    top = pd.concat(top_rows, ignore_index=True)
    search, selected = V276.threshold_search(top)
    chosen = V276.choose_configs(search)
    return pd.DataFrame(feature_rows), predictions, selected, search.merge(chosen[["chosen_type", "feature_set", "threshold"]], on=["feature_set", "threshold"], how="left")


def decision_summary(event_table: pd.DataFrame, chosen: pd.DataFrame) -> pd.DataFrame:
    """生成 test bad_top10 的核心收口表。"""

    rows = V276.baseline_decision(event_table[event_table["split"].astype(str).eq("test")])
    for chosen_type in ["best_any", "best_active", "best_stable_active", "best_noharm_all", "test_best_diagnostic"]:
        sub = chosen[chosen["chosen_type"].astype(str).eq(chosen_type)].copy()
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


def plot_decision(summary: pd.DataFrame) -> None:
    """画出 test bad_top10 RMSE 对照图。"""

    labels = []
    values = []
    colors = []
    for _, row in summary.iterrows():
        labels.append(str(row["source"]).replace("_", "\n"))
        values.append(float(row["rmse"]))
        colors.append("#4C78A8" if bool(row.get("deployable", False)) else "#A1765C")
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.bar(np.arange(len(values)), values, color=colors)
    ax.axhline(float(V276.FIXED_WAIT_LATEST_BADTOP10), color="#E45756", linestyle="--", linewidth=1.3, label="fixed wait-latest")
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("test bad_top10 tail RMSE")
    ax.set_title("v277: style + calibrated physiology candidate gain model")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES / "v277_test_badtop10_style_bio_candidate_gain.png", dpi=180)
    plt.close(fig)


def markdown_table(df: pd.DataFrame) -> str:
    """尽量用 pandas markdown；缺少 tabulate 时回退为 CSV 文本。"""

    try:
        return df.to_markdown(index=False)
    except Exception:  # noqa: BLE001
        return df.to_csv(index=False)


def write_report(summary: pd.DataFrame, chosen: pd.DataFrame, search: pd.DataFrame, feature_audit: pd.DataFrame) -> None:
    """写中文报告。"""

    top_val = search.sort_values(["selection_score", "val_bad_top10_selected_rmse"]).head(18)
    top_test = search[search["test_bad_top10_override_n"] > 0].sort_values("test_bad_top10_selected_rmse").head(18)
    report = f"""# v277 style + calibrated physiology candidate gain model

## 本轮目的

- 在 v276 的 candidate gain 框架上加入驾驶风格和 v271 校准 raw 生理。
- 风格来自 v253a 当前任务口径的 `last60_guard3`，只取 delay=0 事件状态。
- 生理来自 v271 train-only 筛选后的 calibrated raw summary / PCA 特征。
- 同时加入 query-prototype 的 style distance 和 bio271 distance，让模型有机会在车辆相似候选内做状态消歧。
- threshold 只由 val 选择，test 只报告。

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

## 特征审计

- style audited usable feature count: `{int((feature_audit["feature_block"] == "style_last60_guard3_delay0").sum())}`；query feature cap used in model: `{STYLE_FEATURE_CAP}`
- bio271 audited usable feature count: `{int((feature_audit["feature_block"] == "v271_calibrated_raw_physio_screened").sum())}`；query feature cap used in model: `{BIO271_FEATURE_CAP}`

## 判读

- 若 best_stable_active 到 test bad_top10 仍为 `0.6950` 且覆盖率为 `0`，说明 val 能选到的稳定策略没有真正修正 test 差样本。
- 若 test_best_diagnostic 也不能低于 fixed wait-latest，说明加入驾驶风格和校准生理后，连事后少量 headroom 都没有扩大。
- 若 style/bio query 特征只在 val 上触发、不在 test bad_top10 覆盖，说明它更像验证集局部模式，而不是可泛化状态消歧信号。

## 关键图

- `figures\\v277_test_badtop10_style_bio_candidate_gain.png`
"""
    (REPORTS / "v277_style_bio_candidate_gain_model_cn.md").write_text(report, encoding="utf-8")


def build_inventory() -> pd.DataFrame:
    """列出 v277 产物。"""

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
    """打包 v277 产物并做 zip 自检。"""

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(OUT.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=str(path.relative_to(OUT)).replace("\\", "/"))
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        return zf.testzip()


def main() -> None:
    print("[v277] style + calibrated physiology candidate gain model")
    clean_out_dir()

    df, event_table, event_feature_audit, guard = load_enriched_inputs()
    feature_audit, predictions, selected, search_with_choice = build_model_outputs(df)
    chosen = V276.choose_configs(search_with_choice.drop(columns=["chosen_type"], errors="ignore"))
    summary = decision_summary(event_table, chosen)

    write_csv(event_feature_audit, TABLES / "v277_event_feature_audit.csv")
    write_csv(feature_audit, TABLES / "v277_model_feature_set_audit.csv")
    write_csv(predictions, TABLES / "v277_candidate_gain_predictions_compact.csv")
    write_csv(selected, TABLES / "v277_selected_by_strategy.csv")
    write_csv(search_with_choice, TABLES / "v277_threshold_search.csv")
    write_csv(chosen, TABLES / "v277_chosen_configs.csv")
    write_csv(summary, TABLES / "v277_decision_summary.csv")

    input_paths = [
        V277_SCRIPT,
        V276_SCRIPT,
        V252_SCRIPT,
        STYLE_FEATURES,
        V271_EVENT_CONTEXT,
        V271_SCREENING,
        V276.V267_PAIR,
        V276.V267_SELECTED,
        V276.V265_EVENT_SCORES,
    ]
    hashes = pd.DataFrame(
        [
            {
                "path": str(path),
                "exists": bool(path.exists()),
                "sha256": file_sha256(path) if path.exists() else "",
            }
            for path in input_paths
        ]
    )
    write_csv(hashes, LOGS / "input_file_hashes.csv")

    plot_decision(summary)
    write_report(summary, chosen, search_with_choice, event_feature_audit)

    inventory = build_inventory()
    write_csv(inventory, LOGS / "file_inventory.csv")
    zip_bad_file = build_zip()

    deployable = summary[summary["deployable"].astype(bool) & summary["source"].astype(str).str.startswith("best")]
    diagnostic = summary[summary["source"].astype(str).eq("test_best_diagnostic")]
    guardrail = {
        "pass": bool(zip_bad_file is None and guard.get("v267_guardrail_pass", False) and guard.get("v265_guardrail_pass", False)),
        "zip_testzip": zip_bad_file is None,
        "v267_guardrail_pass": bool(guard.get("v267_guardrail_pass", False)),
        "v265_guardrail_pass": bool(guard.get("v265_guardrail_pass", False)),
        "event_n": int(event_table["event_uid"].nunique()),
        "candidate_rows": int(len(df)),
        "style_feature_n": int(guard.get("style_feature_n", 0)),
        "bio271_feature_n": int(guard.get("bio271_feature_n", 0)),
        "feature_set_n": int(len(feature_audit)),
        "search_rows": int(len(search_with_choice)),
        "chosen_rows": int(len(chosen)),
        "best_val_chosen_deployable_test_badtop10": float(deployable["rmse"].min()) if not deployable.empty else math.nan,
        "best_test_diagnostic_badtop10": float(diagnostic["rmse"].min()) if not diagnostic.empty else math.nan,
        "fixed_wait_latest_badtop10": float(V276.FIXED_WAIT_LATEST_BADTOP10),
        "best_deployable_passes_fixed_latest": bool(not deployable.empty and float(deployable["rmse"].min()) < float(V276.FIXED_WAIT_LATEST_BADTOP10)),
        "best_diagnostic_passes_fixed_latest": bool(not diagnostic.empty and float(diagnostic["rmse"].min()) < float(V276.FIXED_WAIT_LATEST_BADTOP10)),
    }
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    if not bool(guardrail["pass"]):
        raise AssertionError("v277 guardrail 未通过：\n" + json.dumps(guardrail, ensure_ascii=False, indent=2))

    print(f"[v277] report={REPORTS / 'v277_style_bio_candidate_gain_model_cn.md'}")
    print(f"[v277] zip={ZIP_PATH}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
