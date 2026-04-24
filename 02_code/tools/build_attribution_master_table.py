#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_attribution_master_table.py
=================================
只读聚合脚本：把 conditioned v2 / baseline 的 sample-level metrics、
manifest、driver style 等 join 成一张 attribution master table，
用于后续事件对齐误差归因分析。

不改训练、不改协议、不改任何源文件。

输出：
  - reports/attribution_master_table.csv   (sample-level wide table)
  - reports/attribution_event_table.csv    (event-level 附表)
  - 终端打印简要统计

用法：
  python tools/build_attribution_master_table.py
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# ── 路径常量 ─────────────────────────────────────────────────────────
REPO_ROOT = Path(r"F:\data_set_process\data_process")

# 主骨架：baseline vs conditioned sample-level comparison
COMPARISON_CSV = (
    REPO_ROOT
    / "reports"
    / "v3_selection_conditioned_interaction_pilot_20260327"
    / "task_2_conditioned_v2"
    / "formal_eval"
    / "sample_level_comparison.csv"
)

# conditioned v2 formal run manifest（含完整样本元数据）
MANIFEST_CSV = (
    REPO_ROOT
    / "tmp"
    / "event_conditioned_runs"
    / "EXP_EVENT_CONDITIONED_TRAJECTORY_V2_FORMAL_20260327_000432"
    / "sample_manifest_used.csv"
)

# event-level metrics（baseline 与 conditioned 混合）
EVENT_CSV = (
    REPO_ROOT
    / "reports"
    / "v3_selection_conditioned_interaction_pilot_20260327"
    / "task_2_conditioned_v2"
    / "formal_eval"
    / "conditioned_event_sample_metrics.csv"
)

BASELINE_EVENT_CSV = (
    REPO_ROOT
    / "reports"
    / "v3_selection_conditioned_interaction_pilot_20260327"
    / "task_2_conditioned_v2"
    / "formal_eval"
    / "baseline_event_sample_metrics.csv"
)

# driver-level style vectors
DRIVER_STYLE_CSV = (
    REPO_ROOT / "reports" / "style_probe_artifacts" / "driver_style_vectors.csv"
)

# 输出
OUT_DIR = REPO_ROOT / "reports"
OUT_MASTER = OUT_DIR / "attribution_master_table.csv"
OUT_EVENT = OUT_DIR / "attribution_event_table.csv"


def load_and_validate():
    """加载所有源 CSV 并做基本校验。"""
    missing = []
    for name, path in [
        ("comparison", COMPARISON_CSV),
        ("manifest", MANIFEST_CSV),
        ("event_conditioned", EVENT_CSV),
        ("driver_style", DRIVER_STYLE_CSV),
    ]:
        if not path.exists():
            missing.append(f"  {name}: {path}")
    if missing:
        print("[ERROR] 以下源文件不存在：")
        for m in missing:
            print(m)
        sys.exit(1)

    comp = pd.read_csv(COMPARISON_CSV)
    manifest = pd.read_csv(MANIFEST_CSV)
    style_drv = pd.read_csv(DRIVER_STYLE_CSV)
    evt_cond = pd.read_csv(EVENT_CSV)
    evt_base = pd.read_csv(BASELINE_EVENT_CSV) if BASELINE_EVENT_CSV.exists() else None

    return comp, manifest, style_drv, evt_cond, evt_base


def build_master_table(comp, manifest, style_drv):
    """构建 sample-level attribution master table。"""

    # ── 1. 主骨架：comparison 表已含 baseline + conditioned + delta ──
    master = comp.copy()

    # ── 2. 补派生 flag ──────────────────────────────────────────────
    master["improved_overall_flag"] = (
        master["delta_rmse_2s_abs_steer"] < 0
    ).astype(int)
    master["improved_tail_flag"] = (
        master["delta_rmse_tail_abs_steer"] < 0
    ).astype(int)

    # ── 3. 从 manifest join 额外元数据 ──────────────────────────────
    # 只取 comparison 中没有但 manifest 中有价值的字段
    manifest_extra_cols = [
        "sample_key",
        "recording_id",
        "anchor_s",
        "anchor_idx",
        "event_start_s",
        "event_end_s",
        "event_duration_s",
        "trigger_score",
        "primary_score",
        "event_level",
        "trigger_type",
        "curvature_anchor",
        "episode_id",
        "d3_included",
        "d3_mechanism_tag_anchor",
        "d3_mechanism_tag_episode",
    ]
    # 过滤掉 manifest 中不存在的列
    manifest_extra_cols = [
        c for c in manifest_extra_cols if c in manifest.columns
    ]
    manifest_subset = manifest[manifest_extra_cols].drop_duplicates(
        subset=["sample_key"]
    )

    n_before = len(master)
    master = master.merge(manifest_subset, on="sample_key", how="left")
    assert len(master) == n_before, (
        f"manifest join 改变了行数: {n_before} -> {len(master)}"
    )

    # ── 4. join driver-level style 向量 ─────────────────────────────
    # 选取 style 中最有代表性的聚合特征（median），用于切片
    style_key_cols = ["driver_id"]
    style_feature_cols = [
        "steer_abs_mean__median",
        "steer_abs_std__median",
        "steer_rate_abs_mean__median",
        "brake_usage_ratio__median",
        "speed_mean__median",
        "speed_std__median",
        "ay_abs_mean__median",
        "yaw_rate_abs_mean__median",
    ]
    # 过滤掉 style 中不存在的列
    style_feature_cols = [
        c for c in style_feature_cols if c in style_drv.columns
    ]
    style_subset = style_drv[style_key_cols + style_feature_cols].copy()
    style_subset = style_subset.rename(
        columns={c: f"driver_style_{c}" for c in style_feature_cols}
    )
    style_subset = style_subset.rename(columns={"driver_id": "subj"})

    n_before = len(master)
    master = master.merge(style_subset, on="subj", how="left")
    assert len(master) == n_before, (
        f"style join 改变了行数: {n_before} -> {len(master)}"
    )

    # ── 5. 添加 latency proxy bucket ───────────────────────────────
    # 用 anchor_s 和 event_start_s 的差值作为简易 latency proxy
    if "anchor_s" in master.columns and "event_start_s" in master.columns:
        master["anchor_to_event_start_s"] = (
            master["anchor_s"] - master["event_start_s"]
        )
        # 分桶：按四分位
        try:
            master["latency_proxy_bucket"] = pd.qcut(
                master["anchor_to_event_start_s"],
                q=4,
                labels=["Q1_fast", "Q2", "Q3", "Q4_slow"],
                duplicates="drop",
            )
        except ValueError:
            # 数据分布不允许四分位时退化为简单分类
            master["latency_proxy_bucket"] = pd.cut(
                master["anchor_to_event_start_s"],
                bins=4,
                labels=["Q1_fast", "Q2", "Q3", "Q4_slow"],
            )

    return master


def build_event_table(evt_cond, evt_base):
    """构建 event-level 附表，合并 conditioned 和 baseline 的事件指标。"""
    parts = []
    if evt_cond is not None and len(evt_cond) > 0:
        df = evt_cond.copy()
        if "model_name" not in df.columns:
            df["model_name"] = "conditioned"
        parts.append(df)
    if evt_base is not None and len(evt_base) > 0:
        df = evt_base.copy()
        if "model_name" not in df.columns:
            df["model_name"] = "baseline"
        parts.append(df)

    if not parts:
        return pd.DataFrame()

    event_table = pd.concat(parts, ignore_index=True)
    return event_table


def print_summary(master, event_table):
    """打印简要统计。"""
    print("=" * 60)
    print("Attribution Master Table 汇总")
    print("=" * 60)
    print(f"总行数: {len(master)}")
    print(f"被试: {sorted(master['subj'].unique())}")
    print(f"列数: {len(master.columns)}")
    print()

    # 改善统计
    n_improved_overall = master["improved_overall_flag"].sum()
    n_improved_tail = master["improved_tail_flag"].sum()
    print(f"overall RMSE 改善样本: {n_improved_overall}/{len(master)} "
          f"({100*n_improved_overall/len(master):.1f}%)")
    print(f"tail RMSE 改善样本:    {n_improved_tail}/{len(master)} "
          f"({100*n_improved_tail/len(master):.1f}%)")
    print()

    # 按切片的 delta 均值
    for slice_col in [
        "eval_morphology_label",
        "interaction_slice",
        "structure_slice",
        "reversal_slice",
    ]:
        if slice_col in master.columns:
            grp = master.groupby(slice_col).agg(
                count=("delta_rmse_2s_abs_steer", "count"),
                delta_rmse_2s_mean=("delta_rmse_2s_abs_steer", "mean"),
                delta_rmse_tail_mean=("delta_rmse_tail_abs_steer", "mean"),
                delta_boundary_shift_mean=(
                    "delta_boundary_shift_abs_err", "mean"
                ),
                improved_tail_pct=("improved_tail_flag", "mean"),
            )
            grp["improved_tail_pct"] = (grp["improved_tail_pct"] * 100).round(1)
            print(f"── 按 {slice_col} 切片 ──")
            print(grp.to_string())
            print()

    # 按被试的 delta 均值
    if "subj" in master.columns:
        grp = master.groupby("subj").agg(
            count=("delta_rmse_2s_abs_steer", "count"),
            delta_rmse_2s_mean=("delta_rmse_2s_abs_steer", "mean"),
            delta_rmse_tail_mean=("delta_rmse_tail_abs_steer", "mean"),
            improved_tail_pct=("improved_tail_flag", "mean"),
        )
        grp["improved_tail_pct"] = (grp["improved_tail_pct"] * 100).round(1)
        print("── 按 subj 切片 ──")
        print(grp.to_string())
        print()

    # latency proxy bucket
    if "latency_proxy_bucket" in master.columns:
        grp = master.groupby("latency_proxy_bucket", observed=False).agg(
            count=("delta_rmse_2s_abs_steer", "count"),
            delta_rmse_2s_mean=("delta_rmse_2s_abs_steer", "mean"),
            delta_rmse_tail_mean=("delta_rmse_tail_abs_steer", "mean"),
            improved_tail_pct=("improved_tail_flag", "mean"),
        )
        grp["improved_tail_pct"] = (grp["improved_tail_pct"] * 100).round(1)
        print("── 按 latency_proxy_bucket 切片 ──")
        print(grp.to_string())
        print()

    # event table
    if len(event_table) > 0:
        print(f"Event Table 总行数: {len(event_table)}")
        print(f"Event 模型: {sorted(event_table['model_name'].unique())}")
        print(f"Event 类型: {sorted(event_table['event_name'].unique())}")
        print()

    print("=" * 60)
    print(f"输出文件:")
    print(f"  Master: {OUT_MASTER}")
    print(f"  Event:  {OUT_EVENT}")
    print("=" * 60)


def main():
    print("加载源文件...")
    comp, manifest, style_drv, evt_cond, evt_base = load_and_validate()

    print("构建 master table...")
    master = build_master_table(comp, manifest, style_drv)

    print("构建 event table...")
    event_table = build_event_table(evt_cond, evt_base)

    # 保存
    master.to_csv(OUT_MASTER, index=False, encoding="utf-8-sig")
    if len(event_table) > 0:
        event_table.to_csv(OUT_EVENT, index=False, encoding="utf-8-sig")

    print()
    print_summary(master, event_table)


if __name__ == "__main__":
    main()
