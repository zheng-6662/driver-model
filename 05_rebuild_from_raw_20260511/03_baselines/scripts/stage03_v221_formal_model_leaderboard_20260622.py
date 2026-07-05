#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v221：按 GPTPro 最新建议先做“统一评估框架”，不训练新模型。

这个脚本只读取 v216/v217/v218/v219 已经生成的 CSV 结果，目的有三点：
1. 把不同版本的候选曲线放到同一张长表中，明确哪些是 formal，哪些只能诊断；
2. 输出整体、强事件、极强峰值、普通样本、场景和 do-no-harm 相关指标；
3. 生成中文报告和 ZIP，为后续 v222a 软融合/受限残差提供固定基线。

重要边界：
- 不训练模型；
- 不改变候选池；
- 不把 oracle、true-label fallback、W3_B4_original_soft 或 diagnostic-only row 放入 formal leaderboard；
- test 只用于报告，不用于选择 v222a 的阈值、模型族或 alpha。
"""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
REBUILD_ROOT = REPO_ROOT / "05_rebuild_from_raw_20260511"
BASE_DIR = REBUILD_ROOT / "03_baselines"
OUT_DIR = BASE_DIR / "v221_formal_model_leaderboard_20260622"
TABLE_DIR = OUT_DIR / "tables"
REPORT_DIR = OUT_DIR / "reports"
FIG_DIR = OUT_DIR / "figures"
LOG_DIR = OUT_DIR / "logs"


PER_SAMPLE_SOURCES = [
    (
        "v216",
        BASE_DIR
        / "v216_joint_driver_vehicle_prediction_20260619"
        / "tables"
        / "v216_per_sample_steering_metrics.csv",
    ),
    (
        "v217",
        BASE_DIR / "v217_event_aware_joint_fusion_20260619" / "tables" / "v217_per_sample_metrics.csv",
    ),
    (
        "v218_diagnostic",
        BASE_DIR / "v218_peak_protected_joint_training_20260620" / "tables" / "v218_per_sample_metrics.csv",
    ),
    (
        "v219",
        BASE_DIR / "v219_ridge_residual_stack_20260620" / "tables" / "v219_per_sample_metrics.csv",
    ),
]

OVERALL_SOURCES = [
    (
        "v216",
        BASE_DIR
        / "v216_joint_driver_vehicle_prediction_20260619"
        / "tables"
        / "v216_metrics_by_model_split.csv",
    ),
    (
        "v217",
        BASE_DIR / "v217_event_aware_joint_fusion_20260619" / "tables" / "v217_metrics_by_model_split.csv",
    ),
    (
        "v218_diagnostic",
        BASE_DIR
        / "v218_peak_protected_joint_training_20260620"
        / "tables"
        / "v218_metrics_by_model_split.csv",
    ),
    (
        "v219",
        BASE_DIR / "v219_ridge_residual_stack_20260620" / "tables" / "v219_metrics_by_model_split.csv",
    ),
]

# 正式候选：来自 v216/v217/v219 已有输出，都是可部署候选/固定验证集组合/轻量残差。
FORMAL_MODEL_ALLOWLIST = {
    "steering_only",
    "joint_equal",
    "joint_steer_focus",
    "avg_joint_focus",
    "global_blend",
    "global_blend_val",
    "peak_floor_090",
    "ridge_residual_joint",
    "ridge_residual_peakfloor",
    "v219_val_selected",
}

# 诊断候选：用于解释 trade-off，不进入 formal selected-config。
DIAGNOSTIC_MODEL_ALLOWLIST = {
    "zero_change",
    "v110_cached",
    "avg_steer_joint",
    "peak_floor_075",
    "peak_floor_100",
    "event_soft_guard",
    "ridge_abs",
    "ridge_residual_global",
    "ridge_residual_global_weighted",
    "v218_peak_tail_mild",
    "v218_peak_tail_strong",
    "v218_val_selected",
}

FORBIDDEN_FORMAL_SUBSTRINGS = [
    "W3_B4_original_soft",
    "oracle",
    "true_label",
    "fallback",
]

NUMERIC_COLUMNS = [
    "array_index",
    "steer_sample_rmse",
    "true_steer_peak_abs",
    "pred_steer_peak_abs",
    "steer_peak_amp_ratio",
    "steer_direction_ok",
    "steer_severe_under",
    "steer_rmse",
    "steer_tail_rmse_1to2s",
    "steer_direction_acc",
    "steer_severe_under_rate",
]


def ensure_dirs() -> None:
    """创建输出目录。"""

    for path in [TABLE_DIR, REPORT_DIR, FIG_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def clean_out_dir() -> None:
    """清理旧 v221 输出，避免混入上一次运行结果。"""

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    ensure_dirs()


def read_csv_utf8(path: Path) -> pd.DataFrame:
    """读取项目 CSV，明确报错缺失依赖。"""

    if not path.exists():
        raise FileNotFoundError(f"缺少 v221 输入表：{path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def normalize_model_name(name: object) -> str:
    """统一少数同义模型名，避免同一候选分裂成两行。"""

    mapping = {
        "global_blend_val": "global_blend",
    }
    return mapping.get(str(name), str(name))


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """把数值列转成 numeric，便于聚合；缺失列保持不变。"""

    out = df.copy()
    for col in NUMERIC_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def assign_scope(df: pd.DataFrame) -> pd.DataFrame:
    """为每一行标记 formal/diagnostic/excluded。"""

    out = df.copy()
    out["model_name_raw"] = out["model_name"].astype(str)
    out["model_name"] = out["model_name"].map(normalize_model_name)
    out["leaderboard_scope"] = "excluded"
    out.loc[out["model_name"].isin(FORMAL_MODEL_ALLOWLIST), "leaderboard_scope"] = "formal"
    out.loc[out["model_name"].isin(DIAGNOSTIC_MODEL_ALLOWLIST), "leaderboard_scope"] = "diagnostic"
    assert_no_forbidden_formal_models(out)
    return out


def assert_no_forbidden_formal_models(df: pd.DataFrame) -> None:
    """正式榜单中不能出现 oracle、fallback 或 W3_B4_original_soft。"""

    formal = df[df["leaderboard_scope"].eq("formal")]
    bad_chunks: List[pd.DataFrame] = []
    for text in FORBIDDEN_FORMAL_SUBSTRINGS:
        mask = formal["model_name"].astype(str).str.contains(text, case=False, regex=False, na=False)
        if mask.any():
            bad_chunks.append(formal.loc[mask, ["source_version", "pool_key", "model_name"]].drop_duplicates())
    if bad_chunks:
        bad = pd.concat(bad_chunks, ignore_index=True)
        raise AssertionError("正式榜单发现禁用候选：\n" + bad.to_string(index=False))


def load_per_sample_tables() -> pd.DataFrame:
    """读取 v216-v219 逐样本表并合并。"""

    frames: List[pd.DataFrame] = []
    for source_name, path in PER_SAMPLE_SOURCES:
        df = read_csv_utf8(path)
        df["source_version"] = source_name
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged = assign_scope(coerce_numeric(merged))
    return add_buckets(merged)


def load_overall_metric_tables() -> pd.DataFrame:
    """读取各版本整体 metrics-by-split 表。"""

    frames: List[pd.DataFrame] = []
    for source_name, path in OVERALL_SOURCES:
        df = read_csv_utf8(path)
        df["source_version"] = source_name
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged = assign_scope(coerce_numeric(merged))
    return merged


def add_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """增加评估分桶；这些是评估标签，不允许后续 v222a 直接当推理特征。"""

    out = df.copy()
    peak = out["true_steer_peak_abs"].fillna(0.0)
    out["bucket_strong_event"] = peak >= 1.5
    out["bucket_extreme_peak"] = peak >= 3.0
    out["bucket_normal_curve"] = peak < 1.5
    out["severe_under"] = out["steer_severe_under"].fillna(0).astype(int)
    if "steer_direction_ok" in out.columns:
        direction_ok = out["steer_direction_ok"].fillna(np.nan)
        out["wrong_side"] = np.where(direction_ok.isna(), np.nan, np.where(direction_ok < 0.5, 1.0, 0.0))
    else:
        out["wrong_side"] = np.nan
    out["peak_ratio"] = out["steer_peak_amp_ratio"]
    return out


def dedupe_by_best_source(df: pd.DataFrame) -> pd.DataFrame:
    """同一候选可能在多个版本重复出现，按优先级保留一份。

    v219/v217/v216 对共同候选的逐样本误差一致或等价时，保留更新版本；
    v218 只作诊断，不覆盖 formal 候选。
    """

    priority = {
        "v219": 4,
        "v217": 3,
        "v216": 2,
        "v218_diagnostic": 1,
    }
    out = df.copy()
    out["_source_priority"] = out["source_version"].map(priority).fillna(0)
    key_cols = ["pool_key", "array_index", "event_uid", "split", "model_name"]
    out = out.sort_values(["_source_priority"], ascending=False)
    out = out.drop_duplicates(subset=key_cols, keep="first")
    return out.drop(columns=["_source_priority"])


def dedupe_overall_by_best_source(df: pd.DataFrame) -> pd.DataFrame:
    """整体表按同样逻辑去重。"""

    priority = {
        "v219": 4,
        "v217": 3,
        "v216": 2,
        "v218_diagnostic": 1,
    }
    out = df.copy()
    out["_source_priority"] = out["source_version"].map(priority).fillna(0)
    key_cols = ["pool_key", "split", "model_name"]
    out = out.sort_values(["_source_priority"], ascending=False)
    out = out.drop_duplicates(subset=key_cols, keep="first")
    return out.drop(columns=["_source_priority"])


def summarize_per_sample(df: pd.DataFrame, group_cols: Iterable[str]) -> pd.DataFrame:
    """按逐样本表汇总强事件/普通样本等分组指标。"""

    rows = []
    group_cols = list(group_cols)
    for keys, one in df.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(group_cols, keys)}
        row.update(
            {
                "n_rows": int(len(one)),
                "mean_sample_rmse": float(one["steer_sample_rmse"].mean()),
                "median_sample_rmse": float(one["steer_sample_rmse"].median()),
                "p90_sample_rmse": float(one["steer_sample_rmse"].quantile(0.90)),
                "severe_under_rate": float(one["severe_under"].mean()),
                "wrong_side_rate": float(one["wrong_side"].mean(skipna=True)),
                "mean_peak_ratio": float(one["peak_ratio"].mean()),
                "median_peak_ratio": float(one["peak_ratio"].median()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_bucket_metrics(per_sample: pd.DataFrame) -> pd.DataFrame:
    """生成强事件、极强峰值、普通样本和场景分组表。"""

    formal = per_sample[per_sample["leaderboard_scope"].eq("formal")].copy()
    frames: List[pd.DataFrame] = []
    bucket_defs = [
        ("bucket_strong_event", "strong_event"),
        ("bucket_extreme_peak", "extreme_peak"),
        ("bucket_normal_curve", "normal_curve"),
    ]
    for col, name in bucket_defs:
        cur = formal[formal[col]].copy()
        if cur.empty:
            continue
        tab = summarize_per_sample(cur, ["pool_key", "pool_name", "split", "model_name"])
        tab.insert(0, "bucket", name)
        frames.append(tab)

    scene = summarize_per_sample(formal, ["pool_key", "pool_name", "split", "scene_type", "model_name"])
    scene.insert(0, "bucket", "scene_type")
    frames.append(scene)
    return pd.concat(frames, ignore_index=True, sort=False)


def build_noharm_table(per_sample: pd.DataFrame) -> pd.DataFrame:
    """以每个 pool 内的基准候选为参照，计算逐样本改善和伤害比例。"""

    formal_test = per_sample[
        per_sample["leaderboard_scope"].eq("formal") & per_sample["split"].eq("test")
    ].copy()
    baseline_by_pool = {
        "loose_main_pool": "avg_joint_focus",
        "strict_main_pool": "peak_floor_090",
    }
    rows = []
    for pool_key, baseline_name in baseline_by_pool.items():
        pool = formal_test[formal_test["pool_key"].eq(pool_key)].copy()
        if pool.empty:
            continue
        base = pool[pool["model_name"].eq(baseline_name)][
            ["array_index", "event_uid", "steer_sample_rmse"]
        ].rename(columns={"steer_sample_rmse": "baseline_rmse"})
        if base.empty:
            continue
        joined = pool.merge(base, on=["array_index", "event_uid"], how="inner")
        joined["delta_vs_baseline"] = joined["steer_sample_rmse"] - joined["baseline_rmse"]
        for model_name, one in joined.groupby("model_name", sort=True):
            rows.append(
                {
                    "pool_key": pool_key,
                    "baseline_model": baseline_name,
                    "model_name": model_name,
                    "n_rows": int(len(one)),
                    "mean_delta_vs_baseline": float(one["delta_vs_baseline"].mean()),
                    "improved_ratio": float((one["delta_vs_baseline"] < -1.0e-9).mean()),
                    "harmed_ratio": float((one["delta_vs_baseline"] > 1.0e-9).mean()),
                    "large_harm_ratio_delta_gt_0p05": float((one["delta_vs_baseline"] > 0.05).mean()),
                    "worst_delta": float(one["delta_vs_baseline"].max()),
                    "best_delta": float(one["delta_vs_baseline"].min()),
                }
            )
    return pd.DataFrame(rows)


def build_universal_failure_cases(per_sample: pd.DataFrame) -> pd.DataFrame:
    """找出所有 formal 候选都预测不好的测试样本。"""

    formal_test = per_sample[
        per_sample["leaderboard_scope"].eq("formal") & per_sample["split"].eq("test")
    ].copy()
    key_cols = ["pool_key", "pool_name", "array_index", "event_uid", "subject", "scene_type"]
    rows = []
    for keys, one in formal_test.groupby(key_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(key_cols, keys)}
        row["model_count"] = int(one["model_name"].nunique())
        row["best_rmse"] = float(one["steer_sample_rmse"].min())
        row["mean_rmse"] = float(one["steer_sample_rmse"].mean())
        row["worst_rmse"] = float(one["steer_sample_rmse"].max())
        row["true_steer_peak_abs"] = float(one["true_steer_peak_abs"].max())
        row["all_bad_ge_0p75"] = bool(row["best_rmse"] >= 0.75)
        row["all_bad_ge_0p50"] = bool(row["best_rmse"] >= 0.50)
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["all_bad_ge_0p75", "all_bad_ge_0p50", "best_rmse", "mean_rmse"],
        ascending=[False, False, False, False],
    ).head(120)


def build_extreme_worst_table(bucket_metrics: pd.DataFrame) -> pd.DataFrame:
    """极强峰值组中，找表现最差的正式候选。"""

    cur = bucket_metrics[
        bucket_metrics["bucket"].eq("extreme_peak") & bucket_metrics["split"].eq("test")
    ].copy()
    if cur.empty:
        return cur
    return cur.sort_values(["pool_key", "severe_under_rate", "mean_sample_rmse"], ascending=[True, False, False])


def build_decision_summary(overall: pd.DataFrame, bucket: pd.DataFrame, noharm: pd.DataFrame) -> pd.DataFrame:
    """汇总 v222a 前的候选决策依据。"""

    formal_test = overall[overall["leaderboard_scope"].eq("formal") & overall["split"].eq("test")].copy()
    rows = []
    for pool_key in sorted(formal_test["pool_key"].dropna().unique()):
        pool = formal_test[formal_test["pool_key"].eq(pool_key)].copy()
        strong = bucket[
            bucket["bucket"].eq("strong_event")
            & bucket["split"].eq("test")
            & bucket["pool_key"].eq(pool_key)
        ].copy()
        normal = bucket[
            bucket["bucket"].eq("normal_curve")
            & bucket["split"].eq("test")
            & bucket["pool_key"].eq(pool_key)
        ].copy()
        noharm_pool = noharm[noharm["pool_key"].eq(pool_key)].copy() if not noharm.empty else pd.DataFrame()
        best_overall = pool.sort_values(["steer_rmse", "steer_tail_rmse_1to2s"]).iloc[0]
        lowest_under = pool.sort_values(["steer_severe_under_rate", "steer_rmse"]).iloc[0]
        best_tail = pool.sort_values(["steer_tail_rmse_1to2s", "steer_rmse"]).iloc[0]
        best_strong_rmse = strong.sort_values(["mean_sample_rmse", "severe_under_rate"]).iloc[0] if not strong.empty else None
        best_strong_under = strong.sort_values(["severe_under_rate", "mean_sample_rmse"]).iloc[0] if not strong.empty else None
        best_normal = normal.sort_values(["mean_sample_rmse", "p90_sample_rmse"]).iloc[0] if not normal.empty else None
        noharm_alt = noharm_pool[noharm_pool["model_name"].ne(noharm_pool["baseline_model"])].copy()
        least_harm = (
            noharm_alt.sort_values(["large_harm_ratio_delta_gt_0p05", "harmed_ratio"]).iloc[0]
            if not noharm_alt.empty
            else None
        )
        rows.append(
            {
                "pool_key": pool_key,
                "pool_name": str(best_overall["pool_name"]),
                "base_best_overall_test": str(best_overall["model_name"]),
                "base_best_overall_test_rmse": float(best_overall["steer_rmse"]),
                "base_best_tail_test": str(best_tail["model_name"]),
                "base_best_tail_test_rmse_1to2s": float(best_tail["steer_tail_rmse_1to2s"]),
                "base_lowest_under_test": str(lowest_under["model_name"]),
                "base_lowest_under_test_rate": float(lowest_under["steer_severe_under_rate"]),
                "base_best_strong_rmse_test": "" if best_strong_rmse is None else str(best_strong_rmse["model_name"]),
                "base_best_strong_mean_sample_rmse": np.nan
                if best_strong_rmse is None
                else float(best_strong_rmse["mean_sample_rmse"]),
                "base_best_strong_under_test": "" if best_strong_under is None else str(best_strong_under["model_name"]),
                "base_best_strong_under_rate": np.nan
                if best_strong_under is None
                else float(best_strong_under["severe_under_rate"]),
                "base_best_normal_curve_test": "" if best_normal is None else str(best_normal["model_name"]),
                "base_best_normal_curve_mean_sample_rmse": np.nan
                if best_normal is None
                else float(best_normal["mean_sample_rmse"]),
                "least_harm_vs_reference": "" if least_harm is None else str(least_harm["model_name"]),
                "least_harm_large_harm_ratio": np.nan
                if least_harm is None
                else float(least_harm["large_harm_ratio_delta_gt_0p05"]),
            }
        )
    return pd.DataFrame(rows)


def to_markdown_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    """安全输出 Markdown 表。"""

    if df.empty:
        return "_无数据_"
    return df.head(max_rows).to_markdown(index=False)


def write_report(
    overall: pd.DataFrame,
    bucket: pd.DataFrame,
    noharm: pd.DataFrame,
    failures: pd.DataFrame,
    extreme_worst: pd.DataFrame,
    decision: pd.DataFrame,
) -> Path:
    """生成中文报告。"""

    formal_test = overall[overall["leaderboard_scope"].eq("formal") & overall["split"].eq("test")].copy()
    lines = [
        "# v221 统一评估框架报告",
        "",
        "## 范围",
        "",
        "- 本轮只读取 v216/v217/v218/v219 已有 CSV 输出，不训练新模型。",
        "- formal leaderboard 只包含允许候选；v218 强峰值训练、零变化、旧 v110 等仅作为诊断或背景。",
        "- test 只用于最终报告，不用于选择 v222a 的阈值、模型族或残差强度。",
        "",
        "## 候选决策摘要",
        "",
        to_markdown_table(decision),
        "",
        "## GPTPro 要求的五个问题",
        "",
        "### 1. 哪个模型总体 RMSE 最低？",
        "",
        to_markdown_table(
            formal_test.sort_values(["pool_key", "steer_rmse"])[
                ["pool_key", "pool_name", "model_name", "steer_rmse", "steer_tail_rmse_1to2s", "steer_severe_under_rate"]
            ],
            max_rows=24,
        ),
        "",
        "### 2. 哪个模型强反应低估率最低？",
        "",
        to_markdown_table(
            bucket[
                bucket["bucket"].eq("strong_event") & bucket["split"].eq("test")
            ].sort_values(["pool_key", "severe_under_rate", "mean_sample_rmse"])[
                [
                    "pool_key",
                    "pool_name",
                    "model_name",
                    "n_rows",
                    "mean_sample_rmse",
                    "severe_under_rate",
                    "mean_peak_ratio",
                ]
            ],
            max_rows=24,
        ),
        "",
        "### 3. 哪个模型对普通弯道/普通样本伤害最小？",
        "",
        to_markdown_table(
            bucket[
                bucket["bucket"].eq("normal_curve") & bucket["split"].eq("test")
            ].sort_values(["pool_key", "mean_sample_rmse", "p90_sample_rmse"])[
                ["pool_key", "pool_name", "model_name", "n_rows", "mean_sample_rmse", "p90_sample_rmse"]
            ],
            max_rows=24,
        ),
        "",
        "### 4. 哪个模型在极强峰值 >=3 rad 上表现最差？",
        "",
        to_markdown_table(
            extreme_worst[
                ["pool_key", "pool_name", "model_name", "n_rows", "mean_sample_rmse", "severe_under_rate", "mean_peak_ratio"]
            ],
            max_rows=24,
        ),
        "",
        "### 5. 哪些样本每个模型都预测不好？",
        "",
        to_markdown_table(
            failures[
                [
                    "pool_key",
                    "array_index",
                    "event_uid",
                    "subject",
                    "scene_type",
                    "best_rmse",
                    "mean_rmse",
                    "true_steer_peak_abs",
                ]
            ],
            max_rows=40,
        ),
        "",
        "## Do-no-harm 参考",
        "",
        to_markdown_table(noharm.sort_values(["pool_key", "large_harm_ratio_delta_gt_0p05", "harmed_ratio"]), max_rows=40),
        "",
        "## 当前结论",
        "",
        "- v222a 可以继续作为下一步，但必须基于 v221 的 `v221_candidate_decision_summary.csv` 固定候选和基准。",
        "- 不建议直接进入 v222b/v223；当前应先做轻量软融合和受限残差。",
        "- v218 代表“强峰值 loss 直接内化训练”的诊断，不应作为新主线。",
    ]
    path = REPORT_DIR / "v221_formal_model_leaderboard_report_cn.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_html_index(report_path: Path) -> Path:
    """写简单 HTML 入口，方便本地浏览。"""

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>v221 统一评估框架</title>
  <style>
    body {{ font-family: "Microsoft YaHei", Arial, sans-serif; margin: 32px; line-height: 1.6; }}
    code {{ background: #f5f5f5; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>v221 统一评估框架</h1>
  <p>本页为入口。主要结论见 <code>{report_path.relative_to(OUT_DIR).as_posix()}</code>。</p>
  <ul>
    <li><code>tables/v221_formal_overall_metrics.csv</code></li>
    <li><code>tables/v221_model_bucket_metrics.csv</code></li>
    <li><code>tables/v221_per_sample_model_errors.csv</code></li>
    <li><code>tables/v221_universal_failure_cases.csv</code></li>
    <li><code>tables/v221_candidate_decision_summary.csv</code></li>
  </ul>
</body>
</html>
"""
    path = OUT_DIR / "v221_formal_model_leaderboard_index.html"
    path.write_text(html, encoding="utf-8")
    return path


def zip_outputs() -> Path:
    """打包并校验 v221 输出。"""

    zip_path = OUT_DIR / "v221_formal_model_leaderboard_pack.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in OUT_DIR.rglob("*"):
            if path == zip_path or path.is_dir():
                continue
            zf.write(path, path.relative_to(OUT_DIR))
    with zipfile.ZipFile(zip_path) as zf:
        bad_file = zf.testzip()
        file_count = len(zf.namelist())
    verify = {"zip_path": str(zip_path), "bad_file": bad_file, "file_count": file_count}
    (LOG_DIR / "v221_zip_verify.json").write_text(json.dumps(verify, ensure_ascii=False, indent=2), encoding="utf-8")
    if bad_file is not None:
        raise AssertionError(f"ZIP 校验失败：{bad_file}")
    return zip_path


def main() -> None:
    """v221 主入口。"""

    clean_out_dir()
    per_sample_raw = load_per_sample_tables()
    overall_raw = load_overall_metric_tables()

    per_sample = dedupe_by_best_source(per_sample_raw)
    overall = dedupe_overall_by_best_source(overall_raw)
    formal_overall = overall[overall["leaderboard_scope"].eq("formal")].copy()
    diagnostic_overall = overall[overall["leaderboard_scope"].eq("diagnostic")].copy()
    bucket = build_bucket_metrics(per_sample)
    noharm = build_noharm_table(per_sample)
    failures = build_universal_failure_cases(per_sample)
    extreme_worst = build_extreme_worst_table(bucket)
    decision = build_decision_summary(formal_overall, bucket, noharm)

    per_sample.to_csv(TABLE_DIR / "v221_per_sample_model_errors.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(TABLE_DIR / "v221_all_scope_overall_metrics.csv", index=False, encoding="utf-8-sig")
    formal_overall.to_csv(TABLE_DIR / "v221_formal_overall_metrics.csv", index=False, encoding="utf-8-sig")
    formal_overall.to_csv(TABLE_DIR / "v221_model_overall_metrics.csv", index=False, encoding="utf-8-sig")
    diagnostic_overall.to_csv(TABLE_DIR / "v221_diagnostic_overall_metrics.csv", index=False, encoding="utf-8-sig")
    bucket.to_csv(TABLE_DIR / "v221_model_bucket_metrics.csv", index=False, encoding="utf-8-sig")
    noharm.to_csv(TABLE_DIR / "v221_noharm_vs_reference.csv", index=False, encoding="utf-8-sig")
    failures.to_csv(TABLE_DIR / "v221_universal_failure_cases.csv", index=False, encoding="utf-8-sig")
    extreme_worst.to_csv(TABLE_DIR / "v221_extreme_peak_worst_models.csv", index=False, encoding="utf-8-sig")
    decision.to_csv(TABLE_DIR / "v221_candidate_decision_summary.csv", index=False, encoding="utf-8-sig")

    report_path = write_report(formal_overall, bucket, noharm, failures, extreme_worst, decision)
    html_path = write_html_index(report_path)
    run_manifest = {
        "out_dir": str(OUT_DIR),
        "input_per_sample_sources": [str(path) for _, path in PER_SAMPLE_SOURCES],
        "input_overall_sources": [str(path) for _, path in OVERALL_SOURCES],
        "formal_model_allowlist": sorted(FORMAL_MODEL_ALLOWLIST),
        "diagnostic_model_allowlist": sorted(DIAGNOSTIC_MODEL_ALLOWLIST),
        "per_sample_rows": int(len(per_sample)),
        "formal_overall_rows": int(len(formal_overall)),
        "report_path": str(report_path),
        "html_path": str(html_path),
    }
    (LOG_DIR / "v221_run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    zip_path = zip_outputs()
    print(f"out_dir={OUT_DIR}")
    print(f"report={report_path}")
    print(f"html={html_path}")
    print(f"zip={zip_path}")


if __name__ == "__main__":
    main()
