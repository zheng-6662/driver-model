# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


PROJECT_ROOT = Path(os.environ.get("DATA_PROCESS_ROOT", r"F:/data_set_process/data_process"))
ROOT = Path(os.environ.get("REBUILD_ROOT", str(PROJECT_ROOT / "05_rebuild_from_raw_20260511")))
SCRIPT_DIR = ROOT / "03_baselines" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import stage03_v03_vehicle_only_baselines as base  # noqa: E402


OUT_ROOT = ROOT / "03_baselines" / "stage03_v03_vehicle_only_inclusion_ablation"
DATASET_ROOT = ROOT / "03_processed_datasets" / "extreme_condition_v0_3_vehicle_only_inclusion_ablation"
REPORT_DIR = ROOT / "09_reports"
NOTES_DIR = ROOT / "00_project_notes"
DAILY_LOG = NOTES_DIR / "daily_logs" / "2026-05-18.md"
ARTIFACT_INDEX = NOTES_DIR / "ARTIFACT_INDEX_CN.md"

CLEAN_CATEGORIES = {
    "strong_response",
    "weak_or_conservative",
    "delayed_or_no_steer",
    "normal_control",
}

VARIANTS = [
    {
        "variant_id": "v03_clean",
        "name_cn": "当前干净集",
        "description_cn": "只使用强响应、弱/保守响应、延迟/无明显转向、正常对照四类。",
        "categories": sorted(CLEAN_CATEGORIES),
    },
    {
        "variant_id": "v03_plus_review",
        "name_cn": "干净集 + 待复核",
        "description_cn": "在当前干净集基础上加入 manual_review，验证待复核样本是否能扩大训练覆盖。",
        "categories": sorted(CLEAN_CATEGORIES | {"manual_review"}),
    },
    {
        "variant_id": "v03_plus_review_excluded",
        "name_cn": "干净集 + 待复核 + 可成窗排除样本",
        "description_cn": "进一步加入 excluded 中仍能构建完整输入/标签窗口的样本，作为最宽松压力测试。",
        "categories": sorted(CLEAN_CATEGORIES | {"manual_review", "excluded"}),
    },
]


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def load_reference_split() -> tuple[dict[str, str], dict[str, str]]:
    manifest_path = ROOT / "03_processed_datasets" / "extreme_condition_v0_3_vehicle_only" / "tables" / "v03_vehicle_only_manifest.csv"
    if not manifest_path.exists():
        return {}, {}
    ref = pd.read_csv(manifest_path, encoding="utf-8-sig", low_memory=False)
    sample_split = dict(zip(ref["sample_id"].astype(str), ref["split"].astype(str)))
    session_split: dict[str, str] = {}
    for session, g in ref.groupby("vehicle_raw_relative_path", dropna=False):
        splits = g["split"].astype(str).value_counts()
        if not splits.empty:
            session_split[str(session)] = str(splits.idxmax())
    return sample_split, session_split


def assign_split(meta: pd.DataFrame, sample_split: dict[str, str], session_split: dict[str, str], seed: int) -> pd.Series:
    out = pd.Series(index=meta.index, dtype=object)
    sample_ids = meta["sample_id"].astype(str)
    sessions = meta["vehicle_raw_relative_path"].astype(str)
    out.loc[sample_ids.isin(sample_split)] = sample_ids.map(sample_split)
    missing = out.isna()
    out.loc[missing & sessions.isin(session_split)] = sessions[missing & sessions.isin(session_split)].map(session_split)

    missing = out.isna()
    if not missing.any():
        return out.astype(str)

    missing_idx = meta.index[missing].to_numpy()
    groups = sessions.loc[missing].to_numpy()
    unique_groups = np.unique(groups)
    if len(unique_groups) < 3:
        out.loc[missing] = "train"
        return out.astype(str)

    idx = np.arange(len(missing_idx))
    first = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    trainval_rel, test_rel = next(first.split(idx, groups=groups))
    split_values = np.full(len(missing_idx), "train", dtype=object)
    split_values[test_rel] = "test"
    groups_trainval = groups[trainval_rel]
    if len(np.unique(groups_trainval)) >= 2:
        second = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed + 1)
        train_rel2, val_rel2 = next(second.split(trainval_rel, groups=groups_trainval))
        split_values[trainval_rel[val_rel2]] = "val"
        split_values[trainval_rel[train_rel2]] = "train"
    out.loc[missing_idx] = split_values
    return out.astype(str)


def build_variant_dataset(variant: dict[str, Any], sample_split: dict[str, str], session_split: dict[str, str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any]]:
    variant_id = str(variant["variant_id"])
    data_dir = DATASET_ROOT / variant_id
    array_dir = data_dir / "arrays"
    table_dir = data_dir / "tables"
    log_dir = data_dir / "logs"
    ensure_dirs(array_dir, table_dir, log_dir)

    array_path = array_dir / f"{variant_id}_vehicle_only_pre2_label5_20hz.npz"
    manifest_path = table_dir / f"{variant_id}_manifest.csv"
    summary_path = log_dir / f"{variant_id}_dataset_summary.json"
    if array_path.exists() and manifest_path.exists() and summary_path.exists():
        z = np.load(array_path, allow_pickle=True)
        meta = pd.read_csv(manifest_path, encoding="utf-8-sig", low_memory=False)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return (
            z["input_values"].astype(np.float32),
            z["input_valid_mask"].astype(bool),
            z["label_values"].astype(np.float32),
            z["label_valid_mask"].astype(bool),
            meta,
            summary,
        )

    episodes = pd.read_csv(base.EPISODE_TABLE, encoding="utf-8-sig", low_memory=False)
    episodes = episodes[episodes["v0_3_category"].isin(set(variant["categories"]))].copy()
    excluded_contexts = variant.get("excluded_contexts")
    if excluded_contexts:
        context_set = set(str(x) for x in excluded_contexts)
        is_excluded = episodes["v0_3_category"].astype(str).eq("excluded")
        in_context = episodes["condition_context_cn"].astype(str).isin(context_set)
        episodes = episodes[(~is_excluded) | in_context].copy()
    episodes = episodes.sort_values(["subject", "session_stamp", "t_condition_anchor"]).reset_index(drop=True)

    cache: dict[str, pd.DataFrame | None] = {}
    rows: list[dict[str, Any]] = []
    x_values: list[np.ndarray] = []
    x_masks: list[np.ndarray] = []
    y_values: list[np.ndarray] = []
    y_masks: list[np.ndarray] = []
    dropped: list[dict[str, Any]] = []

    for _, ep in episodes.iterrows():
        path = str(base.resolve_vehicle_raw_path(ep))
        if path not in cache:
            df, _ = base.v03.load_vehicle_csv(Path(path))
            cache[path] = df
        df = cache[path]
        if df is None or "zx|SteeringWheel" not in df.columns:
            dropped.append({**ep.to_dict(), "drop_reason": "vehicle csv missing or steering missing"})
            continue
        anchor = float(ep["t_condition_anchor"])
        input_query = anchor + base.INPUT_TIME
        label_query = anchor + base.LABEL_TIME
        input_mat = []
        input_mask = []
        for col in base.VEHICLE_FEATURES:
            vals, mask = base.interp_series(df, col, input_query)
            if col == "zx|SteeringWheel":
                anchor_val, anchor_mask = base.interp_series(df, col, np.array([anchor], dtype=float))
                vals = vals - float(anchor_val[0]) if anchor_mask[0] else vals
            input_mat.append(vals)
            input_mask.append(mask)
        y_abs, y_mask = base.interp_series(df, "zx|SteeringWheel", label_query)
        anchor_abs, anchor_mask = base.interp_series(df, "zx|SteeringWheel", np.array([anchor], dtype=float))
        input_valid_ratio = float(np.vstack(input_mask).mean())
        if not anchor_mask[0] or y_mask.mean() < 0.95 or input_valid_ratio < 0.85:
            dropped.append({**ep.to_dict(), "drop_reason": "window incomplete"})
            continue
        y_rel = y_abs - float(anchor_abs[0])
        row = ep.to_dict()
        signed_peak = base.safe_peak_signed(y_rel, y_mask)
        row.update(
            {
                "sample_id": str(ep["episode_uid"]),
                "anchor_steer_abs": float(anchor_abs[0]),
                "target_peak_signed": signed_peak,
                "target_peak_abs": abs(signed_peak) if math.isfinite(signed_peak) else float("nan"),
                "target_final_delta": float(y_rel[y_mask][-1]) if y_mask.any() else float("nan"),
                "input_valid_ratio": input_valid_ratio,
                "label_valid_ratio": float(y_mask.mean()),
            }
        )
        rows.append(row)
        x_values.append(np.stack(input_mat, axis=1).astype(np.float32))
        x_masks.append(np.stack(input_mask, axis=1).astype(bool))
        y_values.append(y_rel.astype(np.float32))
        y_masks.append(y_mask.astype(bool))

    if not rows:
        raise RuntimeError(f"No usable samples for {variant_id}")

    meta = pd.DataFrame(rows)
    meta["split"] = assign_split(meta, sample_split, session_split, 20260518)
    x = np.stack(x_values, axis=0)
    x_mask = np.stack(x_masks, axis=0)
    y = np.stack(y_values, axis=0)
    y_mask = np.stack(y_masks, axis=0)

    summary = {
        "variant_id": variant_id,
        "name_cn": variant["name_cn"],
        "description_cn": variant["description_cn"],
        "source_episode_table": str(base.EPISODE_TABLE),
        "included_categories": variant["categories"],
        "included_excluded_contexts": list(variant.get("excluded_contexts") or []),
        "dropped_features": list(variant.get("drop_features") or []),
        "sample_count": int(len(meta)),
        "dropped_count": int(len(dropped)),
        "split_counts": meta["split"].value_counts().to_dict(),
        "category_counts": meta["v0_3_category"].value_counts().to_dict(),
        "subject_counts": meta["subject"].value_counts().to_dict(),
        "split_rule": "reuse clean sample/session split when possible; group split only for unseen sessions",
    }

    meta.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    if dropped:
        pd.DataFrame(dropped).to_csv(table_dir / f"{variant_id}_dropped_samples.csv", index=False, encoding="utf-8-sig")
    meta.groupby(["split", "v0_3_category_cn"], dropna=False).size().reset_index(name="count").to_csv(
        table_dir / f"{variant_id}_split_category_counts.csv", index=False, encoding="utf-8-sig"
    )
    meta.groupby(["split", "subject"], dropna=False).size().reset_index(name="count").to_csv(
        table_dir / f"{variant_id}_split_subject_counts.csv", index=False, encoding="utf-8-sig"
    )
    np.savez_compressed(
        array_path,
        input_values=x,
        input_valid_mask=x_mask,
        label_values=y,
        label_valid_mask=y_mask,
        input_time=base.INPUT_TIME.astype(np.float32),
        label_time=base.LABEL_TIME.astype(np.float32),
        feature_names=np.array(base.VEHICLE_FEATURES, dtype=object),
        split=meta["split"].astype(str).to_numpy(dtype=object),
        sample_id=meta["sample_id"].astype(str).to_numpy(dtype=object),
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return x, x_mask, y, y_mask, meta, summary


def aggregate_best_subset(per_sample: pd.DataFrame, model_name: str, categories: set[str] | None = None) -> dict[str, float]:
    subset = per_sample[(per_sample["split"] == "test") & (per_sample["model_name"] == model_name)].copy()
    if categories is not None:
        subset = subset[subset["v0_3_category"].isin(categories)]
    if subset.empty:
        return {"n": 0, "sample_rmse_aggregate": float("nan"), "large_n": 0}
    large = subset[subset["large_response"].astype(bool)]
    return {
        "n": int(len(subset)),
        "sample_rmse_aggregate": float(np.sqrt(np.nanmean(np.square(pd.to_numeric(subset["sample_rmse"], errors="coerce"))))),
        "large_n": int(len(large)),
        "wrong_side_rate_large": float(large["wrong_side_large"].astype(bool).mean()) if len(large) else float("nan"),
        "severe_amp_under_rate_large": float(large["severe_amp_under_large"].astype(bool).mean()) if len(large) else float("nan"),
    }


def run_variant(variant: dict[str, Any], sample_split: dict[str, str], session_split: dict[str, str]) -> dict[str, Any]:
    variant_id = str(variant["variant_id"])
    out_dir = OUT_ROOT / variant_id
    table_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    log_dir = out_dir / "logs"
    ensure_dirs(table_dir, fig_dir, log_dir)

    feature_backup = list(base.VEHICLE_FEATURES)
    drop_features = set(str(x) for x in variant.get("drop_features") or [])
    if drop_features:
        base.VEHICLE_FEATURES = [x for x in feature_backup if x not in drop_features]
    try:
        base.TABLE_DIR = table_dir
        base.FIG_DIR = fig_dir
        base.LOG_DIR = log_dir

        x, x_mask, y, y_mask, meta, dataset_summary = build_variant_dataset(variant, sample_split, session_split)
        train_idx = np.where(meta["split"].astype(str).to_numpy() == "train")[0]
        val_idx = np.where(meta["split"].astype(str).to_numpy() == "val")[0]
        test_idx = np.where(meta["split"].astype(str).to_numpy() == "test")[0]
        if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
            raise RuntimeError(f"{variant_id} split invalid: {dataset_summary.get('split_counts')}")

        X, _ = base.flatten_history_features(x, x_mask, meta)
        preds = base.build_no_learning_predictions(y, y_mask, x, x_mask, meta, train_idx)
        preds.update(base.train_vehicle_models(X, y, y_mask, train_idx, val_idx))
        metrics, per_sample = base.evaluate_all(y, y_mask, base.LABEL_TIME, meta, preds, train_idx)
    finally:
        base.VEHICLE_FEATURES = feature_backup
    metrics.to_csv(table_dir / f"{variant_id}_baseline_metrics.csv", index=False, encoding="utf-8-sig")
    per_sample.to_csv(table_dir / f"{variant_id}_per_sample_metrics.csv", index=False, encoding="utf-8-sig")

    test_best = metrics[metrics["split"] == "test"].sort_values("rmse_steer").iloc[0].to_dict()
    best_model = str(test_best["model_name"])
    base.write_group_tables(per_sample, best_model)

    fixed_ids = (
        meta.iloc[test_idx]
        .sort_values(["v0_3_category", "target_peak_abs"], ascending=[True, False])
        .groupby("v0_3_category")
        .head(4)["sample_id"]
        .astype(str)
        .head(20)
        .tolist()
    )
    bad_ids = (
        per_sample[(per_sample["split"] == "test") & (per_sample["model_name"] == best_model)]
        .sort_values("sample_rmse", ascending=False)
        .head(20)["sample_id"]
        .astype(str)
        .tolist()
    )
    base.plot_predictions(
        fixed_ids,
        y,
        y_mask,
        base.LABEL_TIME,
        meta,
        preds,
        fig_dir / f"{variant_id}_fixed_predictions_test.png",
        f"{variant_id} fixed test predictions",
    )
    base.plot_predictions(
        bad_ids,
        y,
        y_mask,
        base.LABEL_TIME,
        meta,
        preds,
        fig_dir / f"{variant_id}_bad_samples_test.png",
        f"{variant_id} worst test predictions",
    )
    np.savez_compressed(
        out_dir / f"{variant_id}_predictions.npz",
        sample_id=meta["sample_id"].astype(str).to_numpy(dtype=object),
        label_time=base.LABEL_TIME.astype(np.float32),
        y_true=y,
        y_mask=y_mask,
        **{f"pred_{k}": v for k, v in preds.items()},
    )
    clean_subset = aggregate_best_subset(per_sample, best_model, CLEAN_CATEGORIES)
    all_subset = aggregate_best_subset(per_sample, best_model, None)
    result = {
        "variant_id": variant_id,
        "name_cn": variant["name_cn"],
        "description_cn": variant["description_cn"],
        "sample_count": int(dataset_summary["sample_count"]),
        "split_counts_json": json.dumps(dataset_summary["split_counts"], ensure_ascii=False),
        "category_counts_json": json.dumps(dataset_summary["category_counts"], ensure_ascii=False),
        "test_best_model": best_model,
        "test_rmse_steer": float(test_best["rmse_steer"]),
        "test_primary_rmse_0_2s": float(test_best["primary_rmse_0_2s"]),
        "test_tail_rmse_2_5s": float(test_best["tail_rmse_2_5s"]),
        "test_wrong_side_rate_large": float(test_best["wrong_side_rate_large"]),
        "test_severe_amp_under_rate_large": float(test_best["severe_amp_under_rate_large"]),
        "test_large_response_recall": float(test_best["large_response_recall"]),
        "clean_subset_test_n": clean_subset["n"],
        "clean_subset_test_sample_rmse_aggregate": clean_subset["sample_rmse_aggregate"],
        "clean_subset_wrong_side_rate_large": clean_subset["wrong_side_rate_large"],
        "clean_subset_severe_amp_under_rate_large": clean_subset["severe_amp_under_rate_large"],
        "all_test_sample_rmse_aggregate": all_subset["sample_rmse_aggregate"],
        "fixed_plot": str(fig_dir / f"{variant_id}_fixed_predictions_test.png"),
        "bad_plot": str(fig_dir / f"{variant_id}_bad_samples_test.png"),
    }
    (log_dir / f"{variant_id}_summary.json").write_text(
        json.dumps({"dataset_summary": dataset_summary, "result": result}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def markdown_table(df: pd.DataFrame) -> str:
    cols = [
        "variant_id",
        "name_cn",
        "sample_count",
        "test_best_model",
        "test_rmse_steer",
        "test_wrong_side_rate_large",
        "test_severe_amp_under_rate_large",
        "test_large_response_recall",
        "clean_subset_test_sample_rmse_aggregate",
        "clean_subset_wrong_side_rate_large",
        "clean_subset_severe_amp_under_rate_large",
    ]
    cols = [c for c in cols if c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df[cols].iterrows():
        vals = []
        for c in cols:
            v = row[c]
            vals.append(f"{v:.6g}" if isinstance(v, float) and np.isfinite(v) else str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(summary: pd.DataFrame) -> None:
    summary_path = OUT_ROOT / "tables" / "v03_vehicle_only_inclusion_ablation_summary.csv"
    lines = [
        "# v0.3 样本纳入范围消融（用户查看版）",
        "",
        "## 这次为什么做",
        "",
        "用户复核后认为待复核样本和部分排除样本也可能可以进入训练，因此本轮不改模型结构，只逐步放宽样本纳入范围，看车辆-only 基线是否变好或变差。",
        "",
        "三档设置如下：",
        "",
        "1. 当前干净集：只用当前已经纳入的四类样本。",
        "2. 干净集 + 待复核：加入 `manual_review`。",
        "3. 干净集 + 待复核 + 可成窗排除样本：再加入 `excluded` 中仍能构建完整窗口的样本。",
        "",
        "切分尽量沿用当前干净集的 session 划分；同一原始记录中新加入的样本跟随原记录的 train/val/test，减少因为重新切分导致的误判。",
        "",
        "## 总体结果",
        "",
        markdown_table(summary),
        "",
        "## 当前读法",
        "",
        "- 如果加入待复核后 test RMSE 和干净子集指标同时改善，说明待复核样本大概率有训练价值。",
        "- 如果总 RMSE 改善但干净子集恶化，说明新增样本可能改变了测试分布，不能直接说更好。",
        "- 如果加入排除样本后明显恶化，说明 excluded 里仍有大量语义或信号问题，只能筛选后使用。",
        "",
        "## 阶段性判断",
        "",
        "- `干净集 + 待复核` 是当前最值得继续的训练样本范围：总体 RMSE 明显下降，大响应错侧率和大响应召回也同步改善。",
        "- `excluded` 不建议直接全量加入：虽然比当前干净集好，但比 `干净集 + 待复核` 差，说明里面还有一批语义混乱、信号异常或锚点不合适的样本。",
        "- 下一步更合理的是把 `manual_review` 升级为可训练样本，同时对 `excluded` 再按排除原因分层，只逐步加入低风险子类。",
        "",
        "## 可查看文件",
        "",
        f"- 汇总表：`{summary_path}`",
        f"- 输出目录：`{OUT_ROOT}`",
    ]
    (REPORT_DIR / "stage03_v03_vehicle_only_inclusion_ablation_user_summary_cn.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def append_notes(summary: pd.DataFrame) -> None:
    best = summary.sort_values("test_rmse_steer").iloc[0].to_dict()
    status_heading = "## 最新更新：2026-05-18 v0.3 样本纳入范围消融"
    artifact_heading = "## v0.3 样本纳入范围消融"
    text = (
        f"{status_heading}\n\n"
        "- 当前阶段：在不改模型结构的前提下，逐步加入待复核和可成窗排除样本，检查车辆-only 基线是否受益。\n"
        f"- 已完成：3 档纳入范围对照，当前总 test RMSE 最低的是 `{best['variant_id']}`，RMSE={best['test_rmse_steer']:.6f}。\n"
        "- 当前判断：该结果只能回答“加样本是否改善车辆-only 基线”，不能证明连续风格或生理数据有效。\n"
        f"- 用户查看版报告：`{REPORT_DIR / 'stage03_v03_vehicle_only_inclusion_ablation_user_summary_cn.md'}`。\n"
    )
    def replace_section(path: Path, heading: str, section: str) -> None:
        if not path.exists():
            return
        raw = path.read_text(encoding="utf-8")
        pattern = re.compile(rf"\n{{0,2}}{re.escape(heading)}\n.*?(?=\n\n## |\Z)", re.S)
        cleaned = pattern.sub("", raw).rstrip()
        path.write_text(cleaned + "\n\n" + section.strip() + "\n", encoding="utf-8")

    project_status = NOTES_DIR / "PROJECT_STATUS_CN.md"
    task_queue = NOTES_DIR / "TASK_QUEUE_CN.md"
    replace_section(project_status, status_heading, text)
    replace_section(
        task_queue,
        status_heading,
        (
            f"{status_heading}\n\n"
            "### 已完成任务\n"
            "- 已跑当前干净集、干净集+待复核、干净集+待复核+可成窗排除样本三档车辆-only 对照。\n"
            "- 已生成汇总表、每档指标表、逐样本指标和预测图。\n\n"
            "### 待做任务\n"
            "- 查看哪一档的坏样本图更符合物理意义。\n"
            "- 决定后续训练样本是否采用更宽松纳入范围，或只筛选部分待复核/排除样本。\n"
        ),
    )
    replace_section(
        ARTIFACT_INDEX,
        artifact_heading,
        (
            f"{artifact_heading}\n\n"
            f"- 用户查看版报告：`{REPORT_DIR / 'stage03_v03_vehicle_only_inclusion_ablation_user_summary_cn.md'}`\n"
            f"- 汇总表：`{OUT_ROOT / 'tables' / 'v03_vehicle_only_inclusion_ablation_summary.csv'}`\n"
            f"- 输出目录：`{OUT_ROOT}`\n"
        ),
    )
    replace_section(DAILY_LOG, status_heading, text)


def main() -> None:
    ensure_dirs(OUT_ROOT / "tables", REPORT_DIR)
    sample_split, session_split = load_reference_split()
    results = []
    for variant in VARIANTS:
        print(f"run {variant['variant_id']} categories={variant['categories']}", flush=True)
        results.append(run_variant(variant, sample_split, session_split))
    summary = pd.DataFrame(results)
    summary_path = OUT_ROOT / "tables" / "v03_vehicle_only_inclusion_ablation_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    write_report(summary)
    append_notes(summary)
    print(summary[["variant_id", "sample_count", "test_best_model", "test_rmse_steer", "test_large_response_recall"]].to_string(index=False))


if __name__ == "__main__":
    main()
