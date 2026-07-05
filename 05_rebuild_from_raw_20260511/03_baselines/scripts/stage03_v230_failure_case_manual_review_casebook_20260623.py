#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v230 failure-case manual review casebook.

本脚本执行 GPTPro 对 v229 的唯一允许 bounded 下一步：失败案例人工复核包 /
论文案例证据包。它只读取 v225、v226、v228、v229 已完成产物，只复制既有图，
不训练模型、不生成新预测、不调 tau / threshold、不新建 gate/router/selector，
不调用 v222b/v223，也不改变 formal headline。

输入：
1. v225 formal route reconstruction evidence pack：正式逐样本评估、失败样本索引和既有 case 图。
2. v226 formal robustness / CI audit：既有聚合诊断图。
3. v228 final paper artifact freeze：最终 formal lock、主结果、CI 与已选图。
4. v229 two-month lessons / failure taxonomy：失败桶、路线复盘、下一步边界。

输出：
1. case 选择索引、人工复核模板、casebook 表、claim 映射、图清单、formal 边界检查。
2. 中文 casebook 报告、导师讨论笔记、论文失败案例小节草稿。
3. 只复制既有 figure 的 selected_casebook_figures。
4. guardrail / forbidden scan / consistency / figure-copy / inventory 日志和 ZIP。
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
REBUILD_ROOT = REPO_ROOT / "05_rebuild_from_raw_20260511"
BASE_DIR = REBUILD_ROOT / "03_baselines"

V225_DIR = BASE_DIR / "v225_formal_route_reconstruction_evidence_pack_20260622"
V226_DIR = BASE_DIR / "v226_formal_robustness_ci_audit_20260622"
V228_DIR = BASE_DIR / "v228_final_paper_artifact_freeze_20260623"
V229_DIR = BASE_DIR / "v229_two_month_lessons_failure_taxonomy_20260623"

OUT_DIR = BASE_DIR / "v230_failure_case_manual_review_casebook_20260623"
TABLE_DIR = OUT_DIR / "tables"
REPORT_DIR = OUT_DIR / "reports"
FIGURE_DIR = OUT_DIR / "figures" / "selected_casebook_figures"
LOG_DIR = OUT_DIR / "logs"
ZIP_NAME = "v230_failure_case_manual_review_casebook_pack.zip"

POOL_DIRS = {
    "loose_main_pool": FIGURE_DIR / "loose_main_pool",
    "strict_main_pool": FIGURE_DIR / "strict_main_pool",
}
CROSS_POOL_FIGURE_DIR = FIGURE_DIR / "cross_pool_repeated_cases"
BASELINE_CONTROL_FIGURE_DIR = FIGURE_DIR / "baseline_sufficient_controls"

FORMAL_MODEL_LOCK = {
    "loose_main_pool": "avg_joint_focus",
    "strict_main_pool": "peak_floor_090",
}

SOURCE_FILES = {
    "v229_failure_taxonomy": V229_DIR / "tables" / "v229_failure_taxonomy_by_pool_event.csv",
    "v229_top_tail_cases": V229_DIR / "tables" / "v229_top_tail_failure_cases.csv",
    "v229_bucket_risk": V229_DIR / "tables" / "v229_bucket_risk_summary.csv",
    "v229_selector_diag": V229_DIR / "tables" / "v229_selector_candidate_diagnosis.csv",
    "v229_decision_matrix": V229_DIR / "tables" / "v229_next_action_decision_matrix.csv",
    "v228_formal_lock": V228_DIR / "tables" / "final_formal_model_lock.csv",
    "v228_main_result": V228_DIR / "tables" / "final_main_result_table.csv",
    "v228_ci": V228_DIR / "tables" / "final_ci_table.csv",
    "v225_failure_case_index": V225_DIR / "tables" / "formal_failure_case_index.csv",
    "v225_per_sample_eval": V225_DIR / "tables" / "per_sample_formal_reconstruction_eval.csv",
}

ALLOWED_FIGURE_ROOTS = [
    V225_DIR / "figures" / "worst_tail_cases",
    V225_DIR / "figures" / "strong_under_cases",
    V225_DIR / "figures" / "baseline_sufficient_cases",
    V226_DIR / "figures" / "tail_error_concentration",
    V226_DIR / "figures" / "underestimation_profile",
    V226_DIR / "figures" / "extreme_peak_cases_summary",
    V228_DIR / "figures" / "selected_main_figures",
    V228_DIR / "figures" / "selected_appendix_figures",
]

REQUIRED_RELATIVE_FILES = [
    "tables/v230_case_selection_index.csv",
    "tables/v230_manual_review_template.csv",
    "tables/v230_failure_casebook_table.csv",
    "tables/v230_bucket_to_claim_mapping.csv",
    "tables/v230_case_figure_inventory.csv",
    "tables/v230_formal_boundary_check.csv",
    "reports/v230_failure_case_manual_review_casebook_cn.md",
    "reports/v230_advisor_discussion_notes_cn.md",
    "reports/v230_paper_failure_case_section_draft_cn.md",
    "logs/run_manifest.json",
    "logs/input_file_hashes.json",
    "logs/guardrail_check.json",
    "logs/forbidden_scan_report.json",
    "logs/file_inventory.json",
    "logs/figure_copy_check.json",
    "logs/consistency_check.json",
    ZIP_NAME,
]

SELECTION_TARGETS = {
    "强反应低估": 5,
    "极端峰值失败": 4,
    "强响应幅值/尾段": 5,
    "反转或多次修正": 3,
    "过零/换向边界": 3,
    "普通曲线可控": 3,
}

MANUAL_REVIEW_COLUMNS = [
    "pool",
    "sample_id",
    "formal_model",
    "scene_type",
    "route_event",
    "failure_bucket_v229",
    "rmse",
    "tail_rmse",
    "under_flag",
    "strong_steer",
    "extreme_peak",
    "reverse",
    "zero_cross",
    "multi_correction",
    "observed_peak_abs",
    "pred_peak_abs",
    "peak_ratio",
    "figure_path",
    "review_status",
    "human_primary_failure_label",
    "human_secondary_failure_label",
    "is_anchor_suspicious",
    "is_prediction_direction_correct",
    "is_tail_lag_visible",
    "is_peak_flattened",
    "is_reverse_missed_or_delayed",
    "is_vehicle_response_mismatch",
    "paper_figure_candidate",
    "advisor_discussion_candidate",
    "human_notes",
]

FORMAL_TABLES_TO_SCAN = [
    "tables/v230_case_selection_index.csv",
    "tables/v230_manual_review_template.csv",
    "tables/v230_failure_casebook_table.csv",
]

FORBIDDEN_FORMAL_PATTERNS = [
    "W3_B4_original_soft",
    "oracle",
    "true_label",
    "true-label",
    "fallback",
    "v222a_bounded_residual",
    "v222a_noharm_gate",
    "oracle_safe_gate",
    "v222b",
    "v223",
]


def rel(path: Path) -> str:
    """返回仓库内相对路径，方便 manifest 和报告稳定引用。"""

    return str(path.relative_to(REPO_ROOT)).replace("\\", "/")


def sha256_file(path: Path) -> str:
    """记录文件哈希，保证复制图和输入表可追溯。"""

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def normalize_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def clean_output_dir() -> None:
    """只清理 v230 固定输出目录。"""

    resolved_out = OUT_DIR.resolve()
    resolved_base = BASE_DIR.resolve()
    if resolved_base not in resolved_out.parents:
        raise RuntimeError(f"Refusing to clean outside baseline dir: {resolved_out}")
    if OUT_DIR.name != "v230_failure_case_manual_review_casebook_20260623":
        raise RuntimeError(f"Unexpected output dir name: {OUT_DIR.name}")
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    for path in [TABLE_DIR, REPORT_DIR, LOG_DIR, *POOL_DIRS.values(), CROSS_POOL_FIGURE_DIR, BASELINE_CONTROL_FIGURE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def ensure_sources_exist() -> None:
    missing = [name for name, path in SOURCE_FILES.items() if not path.exists()]
    missing_fig_roots = [str(path) for path in ALLOWED_FIGURE_ROOTS if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required source files: {missing}")
    if missing_fig_roots:
        raise FileNotFoundError(f"Missing allowed figure roots: {missing_fig_roots}")


def sanitize_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)[:180]


def short_case_filename(rank: int, pool_key: str, sample_id: str) -> str:
    """Windows 路径较长，复制图时使用短哈希文件名，完整 sample_id 保留在 CSV 中。"""

    digest = hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:10]
    subject = sample_id.split("_", 1)[0] if sample_id else "case"
    return f"{rank:03d}_{pool_key}_{sanitize_filename(subject)}_{digest}.png"


def load_case_frame() -> pd.DataFrame:
    """合并 v225 formal case rows 与 v229 failure bucket 标注。"""

    failure = read_csv(SOURCE_FILES["v225_failure_case_index"])
    eval_df = read_csv(SOURCE_FILES["v225_per_sample_eval"])
    v229_top = read_csv(SOURCE_FILES["v229_top_tail_cases"])

    keep_eval = [
        "pool_key",
        "sample_id",
        "formal_model",
        "split",
        "subject",
        "recording",
        "anchor_s",
        "direction_ok",
        "reverse",
        "zero_cross",
        "multi_correction",
        "vehicle_strong",
        "normal_curve",
        "prediction_shape",
        "horizon_length",
    ]
    eval_df = eval_df[[col for col in keep_eval if col in eval_df.columns]].copy()

    df = failure.merge(
        eval_df,
        on=["pool_key", "sample_id", "formal_model", "split"],
        how="left",
        suffixes=("", "_eval"),
    )

    top_map = v229_top[
        ["pool_key", "sample_id", "failure_bucket_v229", "is_top20_tail_v229", "figure_path"]
    ].rename(
        columns={
            "failure_bucket_v229": "failure_bucket_from_v229_top",
            "is_top20_tail_v229": "is_top20_tail_v229",
            "figure_path": "figure_path_from_v229_top",
        }
    )
    df = df.merge(top_map, on=["pool_key", "sample_id"], how="left")

    for col in [
        "rmse",
        "tail_rmse",
        "observed_peak_abs",
        "pred_peak_abs",
        "peak_ratio",
        "anchor_s",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in [
        "under_flag",
        "strong_steer",
        "extreme_peak",
        "high_tail_error",
        "reverse",
        "zero_cross",
        "multi_correction",
        "vehicle_strong",
        "normal_curve",
        "selected_for_figure",
        "worst_tail_case",
        "strong_under_case",
        "baseline_sufficient_case",
        "formal_example_case",
        "is_top20_tail_v229",
    ]:
        if col in df.columns:
            df[col] = df[col].map(normalize_bool)

    df = df[df["split"].astype(str) == "test"].copy()
    df = df[df.apply(lambda r: FORMAL_MODEL_LOCK.get(r["pool_key"]) == r["formal_model"], axis=1)].copy()
    df["failure_bucket_v229"] = df.apply(classify_failure_bucket, axis=1)
    df["selection_bucket"] = df.apply(classify_selection_bucket, axis=1)
    df["source_in_v229_top"] = df["failure_bucket_from_v229_top"].notna()
    df["cross_pool_repeated"] = df["sample_id"].map(df.groupby("sample_id")["pool_key"].nunique()) > 1
    df["figure_source_path"] = df.apply(resolve_existing_figure, axis=1)
    df["figure_exists"] = df["figure_source_path"].astype(str).ne("")
    df["selection_score"] = df.apply(selection_score, axis=1)
    return df


def classify_failure_bucket(row: pd.Series) -> str:
    existing = row.get("failure_bucket_from_v229_top")
    if isinstance(existing, str) and existing.strip():
        return existing
    route_event = str(row.get("route_event", ""))
    under = normalize_bool(row.get("under_flag"))
    strong = normalize_bool(row.get("strong_steer"))
    extreme = normalize_bool(row.get("extreme_peak"))
    high_tail = normalize_bool(row.get("high_tail_error"))
    if route_event == "extreme_peak" or extreme:
        return "极端峰值低估" if under else "极端峰值/尾段难例"
    if under and strong:
        return "强反应低估"
    if route_event in {"reverse", "multi_correction"}:
        return "反转或多次修正"
    if route_event in {"strong_event", "vehicle_strong"} or strong:
        return "强响应幅值/尾段"
    if route_event == "zero_cross":
        return "过零/换向边界"
    if high_tail:
        return "普通样本高尾误差"
    if route_event == "normal_curve":
        return "普通曲线可控"
    return "其他/需人工复核"


def classify_selection_bucket(row: pd.Series) -> str:
    bucket = str(row.get("failure_bucket_v229", ""))
    route_event = str(row.get("route_event", ""))
    primary_group = str(row.get("primary_case_group", ""))
    under = normalize_bool(row.get("under_flag"))
    strong = normalize_bool(row.get("strong_steer"))
    extreme = normalize_bool(row.get("extreme_peak"))
    reverse = normalize_bool(row.get("reverse")) or route_event == "reverse"
    multi = normalize_bool(row.get("multi_correction")) or route_event == "multi_correction"
    zero = normalize_bool(row.get("zero_cross")) or route_event == "zero_cross"

    if bucket == "强反应低估" or (under and strong and not extreme):
        return "强反应低估"
    if bucket in {"极端峰值低估", "极端峰值/尾段难例"} or route_event == "extreme_peak" or extreme:
        return "极端峰值失败"
    if bucket == "强响应幅值/尾段" or route_event in {"strong_event", "vehicle_strong"} or strong:
        return "强响应幅值/尾段"
    if bucket == "反转或多次修正" or reverse or multi:
        return "反转或多次修正"
    if bucket == "过零/换向边界" or zero:
        return "过零/换向边界"
    if bucket == "普通曲线可控" or primary_group == "baseline_sufficient_cases" or route_event == "normal_curve":
        return "普通曲线可控"
    return "其他/需人工复核"


def resolve_existing_figure(row: pd.Series) -> str:
    """只查找既有 figure，不生成新图。"""

    candidates: list[Path] = []
    for col, base_dir in [
        ("figure_path", V225_DIR),
        ("figure_path_from_v229_top", V225_DIR),
    ]:
        value = row.get(col)
        if isinstance(value, str) and value.strip() and value.strip().lower() != "nan":
            path = base_dir / value.strip()
            if path.exists():
                candidates.append(path)

    sample_id = str(row.get("sample_id", ""))
    pool_key = str(row.get("pool_key", ""))
    if sample_id:
        for root in [V225_DIR / "figures", V228_DIR / "figures"]:
            if root.exists():
                for path in root.rglob("*.png"):
                    if sample_id in path.name and (pool_key in str(path) or root == V228_DIR / "figures"):
                        candidates.append(path)

    if not candidates:
        return ""
    # 优先用 v225 case 图，其次用 v228 已选图。
    candidates = sorted(
        set(candidates),
        key=lambda p: (
            0 if V225_DIR in p.parents else 1,
            len(str(p)),
            str(p),
        ),
    )
    return str(candidates[0])


def selection_score(row: pd.Series) -> float:
    """casebook 选择排序：既有图、跨池重复、v229 top、困难程度优先。"""

    score = 0.0
    score += 1000.0 if normalize_bool(row.get("figure_exists")) else 0.0
    score += 250.0 if normalize_bool(row.get("cross_pool_repeated")) else 0.0
    score += 150.0 if normalize_bool(row.get("source_in_v229_top")) else 0.0
    score += 60.0 if normalize_bool(row.get("under_flag")) else 0.0
    score += 50.0 if normalize_bool(row.get("extreme_peak")) else 0.0
    score += 40.0 if normalize_bool(row.get("strong_steer")) else 0.0
    score += 35.0 if normalize_bool(row.get("reverse")) else 0.0
    score += 30.0 if normalize_bool(row.get("multi_correction")) else 0.0
    score += 20.0 if normalize_bool(row.get("zero_cross")) else 0.0
    try:
        score += float(row.get("tail_rmse", 0.0))
    except Exception:
        pass
    return score


def select_cases(case_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按 GPTPro 指定的每池最低数量选择 case，并输出不足项日志。"""

    selected_chunks = []
    shortage_rows = []
    selected_keys: set[tuple[str, str]] = set()

    for pool_key in FORMAL_MODEL_LOCK:
        pool_df = case_df[case_df["pool_key"] == pool_key].copy()
        for bucket, target_n in SELECTION_TARGETS.items():
            eligible = pool_df[pool_df["selection_bucket"] == bucket].copy()
            eligible = eligible[~eligible.apply(lambda r: (r["pool_key"], r["sample_id"]) in selected_keys, axis=1)]
            eligible = eligible.sort_values(
                ["selection_score", "tail_rmse", "rmse"],
                ascending=[False, False, False],
            )
            chosen = eligible.head(target_n).copy()
            for _, row in chosen.iterrows():
                selected_keys.add((row["pool_key"], row["sample_id"]))
            selected_chunks.append(chosen)
            if len(chosen) < target_n:
                shortage_rows.append(
                    {
                        "pool_key": pool_key,
                        "selection_bucket": bucket,
                        "target_n": target_n,
                        "selected_n": len(chosen),
                        "available_n": len(eligible),
                        "status": "insufficient_cases_logged",
                    }
                )

        # 如果某池因极端不足低于 20，则用剩余高尾 formal case 补齐，但仍保持正式锁定模型。
        pool_selected_n = sum(len(chunk[chunk["pool_key"] == pool_key]) for chunk in selected_chunks if not chunk.empty)
        if pool_selected_n < 20:
            fill_n = 20 - pool_selected_n
            remaining = pool_df[~pool_df.apply(lambda r: (r["pool_key"], r["sample_id"]) in selected_keys, axis=1)].copy()
            remaining = remaining.sort_values(["selection_score", "tail_rmse"], ascending=[False, False])
            fill = remaining.head(fill_n).copy()
            fill["selection_bucket"] = fill["selection_bucket"].where(fill["selection_bucket"].ne("其他/需人工复核"), "高尾补充案例")
            for _, row in fill.iterrows():
                selected_keys.add((row["pool_key"], row["sample_id"]))
            selected_chunks.append(fill)

    selected = pd.concat(selected_chunks, ignore_index=True) if selected_chunks else pd.DataFrame()
    if selected.empty:
        raise RuntimeError("No cases selected for v230 casebook")
    selected = selected.sort_values(["pool_key", "selection_bucket", "selection_score"], ascending=[True, True, False]).reset_index(drop=True)
    selected.insert(0, "casebook_rank", range(1, len(selected) + 1))
    selected["case_id"] = selected.apply(lambda r: f"{r['pool_key']}::{r['sample_id']}", axis=1)
    selected["selection_reason_cn"] = selected.apply(selection_reason, axis=1)
    shortage_df = pd.DataFrame(shortage_rows)
    return selected, shortage_df


def selection_reason(row: pd.Series) -> str:
    parts = [f"归入{row['selection_bucket']}"]
    if normalize_bool(row.get("source_in_v229_top")):
        parts.append("v229高尾案例")
    if normalize_bool(row.get("figure_exists")):
        parts.append("已有图")
    if normalize_bool(row.get("cross_pool_repeated")):
        parts.append("跨池重复")
    if normalize_bool(row.get("under_flag")):
        parts.append("低估")
    if normalize_bool(row.get("extreme_peak")):
        parts.append("极端峰值")
    if normalize_bool(row.get("strong_steer")):
        parts.append("强响应")
    return "；".join(parts)


def copy_case_figures(selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """复制 case 图和聚合图；缺图只记录 figure_missing。"""

    inventory_rows = []
    selected = selected.copy()
    copied_paths = []
    figure_statuses = []

    for _, row in selected.iterrows():
        source_text = str(row.get("figure_source_path", "") or "")
        source = Path(source_text) if source_text else None
        bucket = str(row["selection_bucket"])
        pool_key = str(row["pool_key"])
        if bucket == "普通曲线可控":
            dest_dir = BASELINE_CONTROL_FIGURE_DIR
        else:
            dest_dir = POOL_DIRS[pool_key]
        dest_dir.mkdir(parents=True, exist_ok=True)

        if source is None or not source.exists():
            copied_paths.append("figure_missing")
            figure_statuses.append("figure_missing")
            inventory_rows.append(
                {
                    "case_id": row["case_id"],
                    "pool_key": pool_key,
                    "sample_id": row["sample_id"],
                    "figure_type": "case",
                    "source_path": source_text,
                    "copied_path": "figure_missing",
                    "status": "figure_missing",
                    "source_sha256": "",
                    "copied_sha256": "",
                }
            )
            continue

        dest_name = short_case_filename(int(row["casebook_rank"]), pool_key, str(row["sample_id"]))
        dest = dest_dir / dest_name
        shutil.copy2(source, dest)
        copied_paths.append(rel(dest))
        figure_statuses.append("copied")
        inventory_rows.append(
            {
                "case_id": row["case_id"],
                "pool_key": pool_key,
                "sample_id": row["sample_id"],
                "figure_type": "case",
                "source_path": rel(source),
                "copied_path": rel(dest),
                "status": "copied",
                "source_sha256": sha256_file(source),
                "copied_sha256": sha256_file(dest),
            }
        )

    selected["copied_figure_path"] = copied_paths
    selected["figure_status"] = figure_statuses

    repeated_samples = selected.groupby("sample_id")["pool_key"].nunique()
    repeated_samples = set(repeated_samples[repeated_samples > 1].index)
    for sample_id in sorted(repeated_samples):
        sample_rows = selected[selected["sample_id"] == sample_id]
        for _, row in sample_rows.iterrows():
            copied_path = str(row.get("copied_figure_path", ""))
            if not copied_path or copied_path == "figure_missing":
                continue
            source = REPO_ROOT / copied_path
            if not source.exists():
                continue
            digest = hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:10]
            subject = sample_id.split("_", 1)[0] if sample_id else "case"
            dest = CROSS_POOL_FIGURE_DIR / f"cross_pool_{sanitize_filename(subject)}_{digest}_{row['pool_key']}.png"
            shutil.copy2(source, dest)
            inventory_rows.append(
                {
                    "case_id": row["case_id"],
                    "pool_key": row["pool_key"],
                    "sample_id": sample_id,
                    "figure_type": "cross_pool_duplicate",
                    "source_path": rel(source),
                    "copied_path": rel(dest),
                    "status": "copied",
                    "source_sha256": sha256_file(source),
                    "copied_sha256": sha256_file(dest),
                }
            )

    # 复制 v226/v228 的聚合图作为论文/导师讨论背景证据，不作为单个 case 的人工结论。
    aggregate_sources = []
    for root in [
        V226_DIR / "figures" / "tail_error_concentration",
        V226_DIR / "figures" / "underestimation_profile",
        V226_DIR / "figures" / "extreme_peak_cases_summary",
        V228_DIR / "figures" / "selected_main_figures",
        V228_DIR / "figures" / "selected_appendix_figures",
    ]:
        aggregate_sources.extend(sorted(root.glob("*.png")))

    for source in aggregate_sources:
        dest = CROSS_POOL_FIGURE_DIR / f"aggregate_{sanitize_filename(source.name)}"
        if dest.exists():
            continue
        shutil.copy2(source, dest)
        inventory_rows.append(
            {
                "case_id": "",
                "pool_key": "",
                "sample_id": "",
                "figure_type": "aggregate_context",
                "source_path": rel(source),
                "copied_path": rel(dest),
                "status": "copied",
                "source_sha256": sha256_file(source),
                "copied_sha256": sha256_file(dest),
            }
        )

    return selected, pd.DataFrame(inventory_rows)


def make_case_selection_index(selected: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "casebook_rank",
        "case_id",
        "pool_key",
        "pool_name",
        "formal_model",
        "split",
        "sample_id",
        "subject",
        "recording",
        "anchor_s",
        "scene_type",
        "route_event",
        "failure_bucket_v229",
        "selection_bucket",
        "rmse",
        "tail_rmse",
        "under_flag",
        "strong_steer",
        "extreme_peak",
        "reverse",
        "zero_cross",
        "multi_correction",
        "observed_peak_abs",
        "pred_peak_abs",
        "peak_ratio",
        "source_in_v229_top",
        "cross_pool_repeated",
        "figure_source_path",
        "copied_figure_path",
        "figure_status",
        "selection_reason_cn",
    ]
    return selected[[col for col in columns if col in selected.columns]].copy()


def make_manual_review_template(selected: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame()
    df["pool"] = selected["pool_key"]
    for col in [
        "sample_id",
        "formal_model",
        "scene_type",
        "route_event",
        "failure_bucket_v229",
        "rmse",
        "tail_rmse",
        "under_flag",
        "strong_steer",
        "extreme_peak",
        "reverse",
        "zero_cross",
        "multi_correction",
        "observed_peak_abs",
        "pred_peak_abs",
        "peak_ratio",
    ]:
        df[col] = selected[col] if col in selected.columns else ""
    df["figure_path"] = selected["copied_figure_path"]

    # 人工复核字段全部留空，Codex 不填写任何人工判断。
    manual_blank_cols = [
        "review_status",
        "human_primary_failure_label",
        "human_secondary_failure_label",
        "is_anchor_suspicious",
        "is_prediction_direction_correct",
        "is_tail_lag_visible",
        "is_peak_flattened",
        "is_reverse_missed_or_delayed",
        "is_vehicle_response_mismatch",
        "paper_figure_candidate",
        "advisor_discussion_candidate",
        "human_notes",
    ]
    for col in manual_blank_cols:
        df[col] = ""
    return df[MANUAL_REVIEW_COLUMNS].copy()


def make_casebook_table(selected: pd.DataFrame) -> pd.DataFrame:
    df = selected.copy()
    df["metric_summary_cn"] = df.apply(
        lambda r: (
            f"tail RMSE={format_float(r['tail_rmse'], 3)}，RMSE={format_float(r['rmse'], 3)}，"
            f"peak ratio={format_float(r['peak_ratio'], 3)}。"
        ),
        axis=1,
    )
    df["paper_use_cn"] = df["selection_bucket"].map(
        {
            "强反应低估": "用于说明强反应幅值仍有低估风险。",
            "极端峰值失败": "用于说明极端峰值是主要 limitation。",
            "强响应幅值/尾段": "用于说明尾段延续和幅值跟随仍是难点。",
            "反转或多次修正": "用于说明平均 RMSE 不能覆盖反转/多修正失败。",
            "过零/换向边界": "用于说明换向边界仍需人工检查。",
            "普通曲线可控": "作为普通曲线可控的对照样本。",
        }
    ).fillna("作为补充失败案例。")
    keep = [
        "casebook_rank",
        "case_id",
        "pool_key",
        "formal_model",
        "sample_id",
        "scene_type",
        "route_event",
        "selection_bucket",
        "failure_bucket_v229",
        "rmse",
        "tail_rmse",
        "under_flag",
        "strong_steer",
        "extreme_peak",
        "reverse",
        "zero_cross",
        "multi_correction",
        "observed_peak_abs",
        "pred_peak_abs",
        "peak_ratio",
        "copied_figure_path",
        "figure_status",
        "metric_summary_cn",
        "paper_use_cn",
    ]
    return df[[col for col in keep if col in df.columns]].copy()


def make_claim_mapping(selected: pd.DataFrame) -> pd.DataFrame:
    def cases_for(bucket_names: Iterable[str], n: int = 8) -> str:
        subset = selected[selected["selection_bucket"].isin(set(bucket_names))]
        subset = subset.sort_values("tail_rmse", ascending=False).head(n)
        return "; ".join(subset["case_id"].tolist())

    rows = [
        {
            "claim_id": "C1_direction_is_stable_but_amplitude_hard",
            "claim_strength": "main",
            "allowed_wording": "formal 主结果显示方向判断稳定，但幅值和尾段仍是困难部分。",
            "forbidden_wording": "不能写成强反应幅值问题已经解决。",
            "supporting_table": "v228/final_main_result_table.csv; v230_failure_casebook_table.csv",
            "supporting_cases": cases_for(["强反应低估", "强响应幅值/尾段"]),
        },
        {
            "claim_id": "C2_tail_error_concentrates_in_difficult_cases",
            "claim_strength": "limitation",
            "allowed_wording": "tail error 集中在少量困难样本，应作为 limitation 和 case study 呈现。",
            "forbidden_wording": "不能只用平均 RMSE 淡化尾部集中误差。",
            "supporting_table": "v229_failure_taxonomy_by_pool_event.csv; v230_case_selection_index.csv",
            "supporting_cases": cases_for(["强反应低估", "极端峰值失败", "强响应幅值/尾段"]),
        },
        {
            "claim_id": "C3_strong_reaction_underestimation_remains",
            "claim_strength": "limitation",
            "allowed_wording": "强反应样本仍有低估案例，需要人工复核和论文透明呈现。",
            "forbidden_wording": "不能声称峰值保护已经完全消除低估。",
            "supporting_table": "v230_manual_review_template.csv; v230_failure_casebook_table.csv",
            "supporting_cases": cases_for(["强反应低估"]),
        },
        {
            "claim_id": "C4_extreme_peak_cases_are_key_limitation",
            "claim_strength": "limitation",
            "allowed_wording": "极端峰值样本是关键失败桶之一，应放入失败案例小节。",
            "forbidden_wording": "不能用普通样本表现代替极端峰值表现。",
            "supporting_table": "v229_bucket_risk_summary.csv; v230_case_selection_index.csv",
            "supporting_cases": cases_for(["极端峰值失败"]),
        },
        {
            "claim_id": "C5_reverse_multi_correction_are_not_solved_by_average_rmse",
            "claim_strength": "limitation",
            "allowed_wording": "反转/多次修正案例说明平均 RMSE 不能覆盖结构性失败。",
            "forbidden_wording": "不能因为方向准确率高就忽略反转延迟或多修正失败。",
            "supporting_table": "v230_failure_casebook_table.csv",
            "supporting_cases": cases_for(["反转或多次修正"]),
        },
        {
            "claim_id": "C6_normal_curve_cases_are_relatively_controlled",
            "claim_strength": "main",
            "allowed_wording": "普通曲线对照样本相对可控，可作为失败案例的参照。",
            "forbidden_wording": "不能把普通曲线可控扩展成所有复杂样本可控。",
            "supporting_table": "v230_case_selection_index.csv",
            "supporting_cases": cases_for(["普通曲线可控"]),
        },
        {
            "claim_id": "C7_v222a_selector_gap_is_diagnostic_not_formal",
            "claim_strength": "diagnostic_only",
            "allowed_wording": "selector gap 只能作为路线诊断，不能作为 formal 可部署提升。",
            "forbidden_wording": "不能把诊断上限写成 formal model improvement。",
            "supporting_table": "v229_selector_candidate_diagnosis.csv",
            "supporting_cases": "",
        },
    ]
    return pd.DataFrame(rows)


def make_boundary_check(selected: pd.DataFrame, shortage_df: pd.DataFrame) -> pd.DataFrame:
    counts = selected.groupby(["pool_key", "selection_bucket"]).size().to_dict()
    rows = []
    for pool_key, formal_model in FORMAL_MODEL_LOCK.items():
        rows.append(
            {
                "check_name": f"formal_lock_{pool_key}",
                "status": "pass",
                "detail": f"{pool_key}={formal_model}",
            }
        )
        for bucket, target in SELECTION_TARGETS.items():
            selected_n = counts.get((pool_key, bucket), 0)
            rows.append(
                {
                    "check_name": f"minimum_case_count_{pool_key}_{bucket}",
                    "status": "pass" if selected_n >= target else "warn",
                    "detail": f"selected={selected_n}; target={target}",
                }
            )
    rows.extend(
        [
            {"check_name": "no_training_executed", "status": "pass", "detail": "audit-only packaging"},
            {"check_name": "no_new_prediction_generated", "status": "pass", "detail": "only copied existing figures"},
            {"check_name": "no_tau_threshold_created", "status": "pass", "detail": "no config output"},
            {"check_name": "no_gate_router_selector_created", "status": "pass", "detail": "casebook only"},
            {"check_name": "diagnostic_boundary_preserved", "status": "pass", "detail": "formal casebook rows come from v225 formal lock rows"},
            {"check_name": "human_review_fields_blank", "status": "pass", "detail": "manual template blanks are checked in consistency"},
            {"check_name": "shortages_logged", "status": "pass" if shortage_df.empty else "warn", "detail": f"shortage_rows={len(shortage_df)}"},
        ]
    )
    return pd.DataFrame(rows)


def format_float(value: object, digits: int = 3) -> str:
    try:
        if pd.isna(value):
            return ""
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def make_reports(
    selected: pd.DataFrame,
    casebook: pd.DataFrame,
    claim_mapping: pd.DataFrame,
    figure_inventory: pd.DataFrame,
    final_main: pd.DataFrame,
    shortage_df: pd.DataFrame,
) -> tuple[str, str, str]:
    counts = selected.groupby(["pool_key", "selection_bucket"]).size().reset_index(name="n")
    copied = int((figure_inventory["status"] == "copied").sum()) if not figure_inventory.empty else 0
    missing = int((figure_inventory["status"] == "figure_missing").sum()) if not figure_inventory.empty else 0

    lines = [
        "# v230 失败案例人工复核 / 论文案例证据包",
        "",
        f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}",
        "- 范围：audit-only + paper-case packaging；不训练、不新预测、不调阈值、不建 gate/router/selector。",
        "- 当前 formal lock：`loose_main_pool=avg_joint_focus`，`strict_main_pool=peak_floor_090`。",
        f"- 选入 case 数：{len(selected)}；复制图数：{copied}；缺图记录：{missing}。",
        "",
        "## 正式主结果边界",
        "",
        "| pool | model | test n | RMSE | tail RMSE | direction acc | under rate |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in final_main.iterrows():
        lines.append(
            f"| {row['pool_key']} | {row['formal_model']} | {int(row['n'])} | "
            f"{format_float(row['rmse'], 6)} | {format_float(row['tail_rmse'], 6)} | "
            f"{format_float(row['direction_acc'], 6)} | {format_float(row['under_rate'], 6)} |"
        )
    lines.extend(["", "## case 选择分布", "", "| pool | bucket | n |", "|---|---|---:|"])
    for _, row in counts.iterrows():
        lines.append(f"| {row['pool_key']} | {row['selection_bucket']} | {int(row['n'])} |")
    if not shortage_df.empty:
        lines.extend(["", "## 数量不足记录", ""])
        for _, row in shortage_df.iterrows():
            lines.append(
                f"- `{row['pool_key']} / {row['selection_bucket']}`：selected={row['selected_n']}，target={row['target_n']}。"
            )
    lines.extend(
        [
            "",
            "## 人工复核说明",
            "",
            "`v230_manual_review_template.csv` 中的人工复核字段已全部留空。后续需要人工逐图填写，"
            "Codex 没有自动判断锚点是否可疑、方向是否正确、尾段是否滞后或峰值是否压平。",
            "",
            "## 论文使用边界",
            "",
            "- 可以写：方向和普通响应相对稳定，但困难样本中的幅值、尾段、极端峰值和反转仍是 limitation。",
            "- 可以写：casebook 用于失败模式展示和人工复核，不是新的模型提升。",
            "- 不可以写：v230 改进了 RMSE、训练了新模型或证明 selector/gate 已可部署。",
        ]
    )
    report = "\n".join(lines)

    advisor_lines = [
        "# v230 导师讨论笔记",
        "",
        "## 建议先讲的结论",
        "",
        "- v230 是失败案例证据包，不是新实验。",
        "- 主模型已经冻结；本轮只把最典型失败样本、普通曲线对照和聚合诊断图整理出来。",
        "- 讨论重点应放在论文怎么诚实呈现 limitation，以及人工复核这些样本是否有锚点/样本定义问题。",
        "",
        "## 讨论顺序",
        "",
        "1. 先看普通曲线可控样本，建立模型不是完全失败的参照。",
        "2. 再看强反应低估和极端峰值，解释为什么平均 RMSE 不够。",
        "3. 再看反转/多次修正，解释结构性困难。",
        "4. 最后决定哪些图进入论文主文，哪些留附录。",
        "",
        "## 不建议讨论成",
        "",
        "- 不建议讨论成下一轮模型训练计划。",
        "- 不建议用 casebook 推导 aggregate improvement。",
        "- 不建议把诊断上限写成正式可部署结论。",
    ]
    advisor = "\n".join(advisor_lines)

    paper_lines = [
        "# 论文失败案例小节草稿",
        "",
        "在最终冻结的 formal 设置下，模型在方向判断和普通响应样本上表现较稳定，但失败案例显示，"
        "高尾误差主要集中在少数困难事件。为避免仅用平均 RMSE 掩盖结构性风险，我们构建了一个"
        "人工复核 casebook，覆盖强反应低估、极端峰值失败、强响应尾段、反转/多次修正、过零/换向边界，"
        "并加入普通曲线可控样本作为对照。",
        "",
        "这些案例不构成新的模型训练结果，也不改变 formal headline。它们的作用是支持 limitation："
        "当前方法可以稳定捕捉多数样本的趋势和方向，但在强幅值与复杂尾段动态上仍存在可见误差。"
        "后续人工复核将进一步判断这些失败来自模型保守性、锚点不确定性、输入窗口不可判别，还是车辆响应与驾驶员动作之间的错配。",
        "",
        "写作时应避免将 casebook 解释为 aggregate improvement。casebook 只提供失败模式证据和可视化样例，"
        "正式定量结论仍以 v228 主结果、CI 和 limitation 表为准。",
    ]
    paper = "\n".join(paper_lines)
    return report, advisor, paper


def scan_forbidden() -> dict:
    hits = []
    for rel_path in FORMAL_TABLES_TO_SCAN:
        path = OUT_DIR / rel_path
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8-sig", errors="ignore")
        for pattern in FORBIDDEN_FORMAL_PATTERNS:
            if pattern.lower() in text.lower():
                hits.append({"file": rel_path, "pattern": pattern})
    return {
        "pass": len(hits) == 0,
        "scanned_files": FORMAL_TABLES_TO_SCAN,
        "pattern_count": len(FORBIDDEN_FORMAL_PATTERNS),
        "hits": hits,
    }


def make_guardrail_check(formal_lock: pd.DataFrame, selected: pd.DataFrame, forbidden_report: dict) -> dict:
    lock_exact = True
    for pool_key, formal_model in FORMAL_MODEL_LOCK.items():
        match = formal_lock[
            (formal_lock["pool_key"] == pool_key)
            & (formal_lock["formal_model"] == formal_model)
        ]
        lock_exact = lock_exact and not match.empty
    selected_lock_exact = selected.apply(
        lambda r: FORMAL_MODEL_LOCK.get(r["pool_key"]) == r["formal_model"],
        axis=1,
    ).all()
    return {
        "pass": bool(lock_exact and selected_lock_exact and forbidden_report["pass"]),
        "formal_lock_exact": bool(lock_exact),
        "selected_cases_follow_formal_lock": bool(selected_lock_exact),
        "no_training_executed": True,
        "no_new_prediction_arrays_created": True,
        "no_new_tau_threshold_config_created": True,
        "no_new_gate_router_selector_created": True,
        "v222b_v223_not_called": True,
        "manual_review_fields_left_blank": True,
        "diagnostic_only_rows_excluded_from_formal_casebook": True,
        "forbidden_scan_pass": bool(forbidden_report["pass"]),
    }


def make_consistency_check(
    selected: pd.DataFrame,
    manual_template: pd.DataFrame,
    figure_inventory: pd.DataFrame,
    shortage_df: pd.DataFrame,
    required_missing: list[str],
    guardrail: dict,
) -> dict:
    manual_cols = [
        "review_status",
        "human_primary_failure_label",
        "human_secondary_failure_label",
        "is_anchor_suspicious",
        "is_prediction_direction_correct",
        "is_tail_lag_visible",
        "is_peak_flattened",
        "is_reverse_missed_or_delayed",
        "is_vehicle_response_mismatch",
        "paper_figure_candidate",
        "advisor_discussion_candidate",
        "human_notes",
    ]
    manual_blank = all(manual_template[col].fillna("").astype(str).eq("").all() for col in manual_cols)
    valid_bucket = selected["selection_bucket"].fillna("").astype(str).ne("").all()
    linked_to_v225 = selected["array_index"].notna().all()
    figure_bad = figure_inventory[~figure_inventory["status"].isin(["copied", "figure_missing"])]
    per_pool_counts = selected.groupby("pool_key")["sample_id"].count().to_dict()
    selected_case_count_ok = len(selected) >= 40
    consistency = {
        "pass": bool(
            guardrail["pass"]
            and not required_missing
            and selected_case_count_ok
            and valid_bucket
            and linked_to_v225
            and manual_blank
            and figure_bad.empty
        ),
        "required_files_missing": required_missing,
        "selected_case_count": int(len(selected)),
        "selected_case_count_ok": selected_case_count_ok,
        "per_pool_counts": {str(k): int(v) for k, v in per_pool_counts.items()},
        "shortage_rows": int(len(shortage_df)),
        "each_selected_case_has_valid_bucket": bool(valid_bucket),
        "each_selected_case_links_to_existing_evidence_row": bool(linked_to_v225),
        "manual_review_fields_blank": bool(manual_blank),
        "figure_copy_bad_rows": int(len(figure_bad)),
        "figure_missing_rows_logged": int((figure_inventory["status"] == "figure_missing").sum()) if not figure_inventory.empty else 0,
    }
    return consistency


def write_input_hashes() -> None:
    rows = []
    for name, path in SOURCE_FILES.items():
        rows.append(
            {
                "source_name": name,
                "relative_path": rel(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    write_json(LOG_DIR / "input_file_hashes.json", rows)
    write_csv(LOG_DIR / "input_file_hashes.csv", pd.DataFrame(rows))


def collect_file_inventory() -> pd.DataFrame:
    rows = []
    for path in sorted(OUT_DIR.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "relative_path": rel(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return pd.DataFrame(rows)


def missing_required_files() -> list[str]:
    return [item for item in REQUIRED_RELATIVE_FILES if not (OUT_DIR / item).exists()]


def zip_output() -> Path:
    zip_path = OUT_DIR / ZIP_NAME
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(OUT_DIR.rglob("*")):
            if path.is_file() and path != zip_path:
                zf.write(path, path.relative_to(OUT_DIR))
    with zipfile.ZipFile(zip_path, "r") as zf:
        bad_file = zf.testzip()
    if bad_file is not None:
        raise RuntimeError(f"ZIP validation failed at {bad_file}")
    return zip_path


def main() -> None:
    ensure_sources_exist()
    clean_output_dir()

    formal_lock = read_csv(SOURCE_FILES["v228_formal_lock"])
    final_main = read_csv(SOURCE_FILES["v228_main_result"])
    v229_failure_taxonomy = read_csv(SOURCE_FILES["v229_failure_taxonomy"])

    case_df = load_case_frame()
    selected, shortage_df = select_cases(case_df)
    selected, figure_inventory = copy_case_figures(selected)

    selection_index = make_case_selection_index(selected)
    manual_template = make_manual_review_template(selected)
    casebook = make_casebook_table(selected)
    claim_mapping = make_claim_mapping(selected)
    boundary_check = make_boundary_check(selected, shortage_df)

    write_csv(TABLE_DIR / "v230_case_selection_index.csv", selection_index)
    write_csv(TABLE_DIR / "v230_manual_review_template.csv", manual_template)
    write_csv(TABLE_DIR / "v230_failure_casebook_table.csv", casebook)
    write_csv(TABLE_DIR / "v230_bucket_to_claim_mapping.csv", claim_mapping)
    write_csv(TABLE_DIR / "v230_case_figure_inventory.csv", figure_inventory)
    write_csv(TABLE_DIR / "v230_formal_boundary_check.csv", boundary_check)
    if not shortage_df.empty:
        write_csv(TABLE_DIR / "v230_case_selection_shortage_log.csv", shortage_df)

    report, advisor, paper = make_reports(
        selected=selected,
        casebook=casebook,
        claim_mapping=claim_mapping,
        figure_inventory=figure_inventory,
        final_main=final_main,
        shortage_df=shortage_df,
    )
    (REPORT_DIR / "v230_failure_case_manual_review_casebook_cn.md").write_text(report, encoding="utf-8")
    (REPORT_DIR / "v230_advisor_discussion_notes_cn.md").write_text(advisor, encoding="utf-8")
    (REPORT_DIR / "v230_paper_failure_case_section_draft_cn.md").write_text(paper, encoding="utf-8")

    write_input_hashes()
    forbidden_report = scan_forbidden()
    write_json(LOG_DIR / "forbidden_scan_report.json", forbidden_report)
    guardrail = make_guardrail_check(formal_lock, selected, forbidden_report)
    write_json(LOG_DIR / "guardrail_check.json", guardrail)

    figure_copy_check = {
        "pass": bool(figure_inventory["status"].isin(["copied", "figure_missing"]).all()),
        "copied_count": int((figure_inventory["status"] == "copied").sum()),
        "figure_missing_count": int((figure_inventory["status"] == "figure_missing").sum()),
        "failed_count": int((~figure_inventory["status"].isin(["copied", "figure_missing"])).sum()),
        "case_figure_missing_count": int(
            ((figure_inventory["figure_type"] == "case") & (figure_inventory["status"] == "figure_missing")).sum()
        ),
    }
    write_json(LOG_DIR / "figure_copy_check.json", figure_copy_check)

    run_manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "script": rel(Path(__file__).resolve()),
        "output_dir": rel(OUT_DIR),
        "scope": "audit_only_paper_case_packaging",
        "formal_model_lock": FORMAL_MODEL_LOCK,
        "source_dirs": {
            "v225": rel(V225_DIR),
            "v226": rel(V226_DIR),
            "v228": rel(V228_DIR),
            "v229": rel(V229_DIR),
        },
        "selected_case_count": int(len(selected)),
        "v229_failure_taxonomy_rows": int(len(v229_failure_taxonomy)),
        "guardrail_pass": guardrail["pass"],
        "forbidden_scan_pass": forbidden_report["pass"],
    }
    write_json(LOG_DIR / "run_manifest.json", run_manifest)

    pre_inventory = collect_file_inventory()
    write_csv(LOG_DIR / "file_inventory.csv", pre_inventory)
    write_json(LOG_DIR / "file_inventory.json", pre_inventory.to_dict(orient="records"))

    # ZIP 前先检查一次必需文件；ZIP 本身随后补入。
    late_written_files = {ZIP_NAME, "logs/consistency_check.json"}
    missing_before_zip = [
        item for item in REQUIRED_RELATIVE_FILES if item not in late_written_files and not (OUT_DIR / item).exists()
    ]
    if missing_before_zip:
        raise RuntimeError(f"Missing required outputs before ZIP: {missing_before_zip}")

    zip_path = zip_output()
    required_missing = [
        item for item in missing_required_files() if item != "logs/consistency_check.json"
    ]

    consistency = make_consistency_check(
        selected=selected,
        manual_template=manual_template,
        figure_inventory=figure_inventory,
        shortage_df=shortage_df,
        required_missing=required_missing,
        guardrail=guardrail,
    )
    write_json(LOG_DIR / "consistency_check.json", consistency)

    # consistency 写入后刷新 ZIP 和 inventory，确保 ZIP 内包含最终 consistency_check。
    zip_path = zip_output()
    final_inventory = collect_file_inventory()
    write_csv(LOG_DIR / "file_inventory.csv", final_inventory)
    write_json(LOG_DIR / "file_inventory.json", final_inventory.to_dict(orient="records"))

    required_missing = missing_required_files()
    if required_missing:
        raise RuntimeError(f"Missing required outputs after ZIP: {required_missing}")
    if not guardrail["pass"]:
        raise RuntimeError("guardrail_check.pass is false")
    if not consistency["pass"]:
        raise RuntimeError("consistency_check.pass is false")
    if not forbidden_report["pass"]:
        raise RuntimeError("forbidden_scan_report.pass is false")

    print(
        json.dumps(
            {
                "output_dir": str(OUT_DIR),
                "zip": str(zip_path),
                "selected_case_count": int(len(selected)),
                "guardrail_pass": guardrail["pass"],
                "consistency_pass": consistency["pass"],
                "forbidden_hits": forbidden_report["hits"],
                "required_files_missing": required_missing,
                "figure_copy_check": figure_copy_check,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
