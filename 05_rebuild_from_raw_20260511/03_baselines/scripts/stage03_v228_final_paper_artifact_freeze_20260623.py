#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v228 final paper artifact freeze.

本脚本只执行 GPTPro 本地软件端给出的 reporting / packaging / manuscript-readiness
任务，不训练模型，不生成新预测，不搜索阈值，不创建 gate/router/selector，也不改变
formal headline。它把 v225/v226/v227 已经确认的正式证据收窄成一个最终论文冻结包。

输入边界：
1. v225 formal route reconstruction evidence pack 的正式表、报告、日志和 ZIP。
2. v226 formal robustness / CI audit 的正式表、报告、日志、图和 ZIP。
3. v227 paper claim readiness pack 的写作整理表、报告、日志、已选图和 ZIP。

输出边界：
1. 固化 final formal model lock、主结果表、CI 表、claim lock、limitations、figure
   selection、artifact manifest 和 guardrail summary。
2. 生成三个中文写作文件：v228 freeze 报告、结果段落草稿、claim 边界说明。
3. 复制 v227 已经选出的代表图到 selected_main_figures / selected_appendix_figures。
4. 生成 consistency / forbidden scan / guardrail / inventory 日志并打 ZIP。
"""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_DIR = REPO_ROOT / "05_rebuild_from_raw_20260511" / "03_baselines"

V225_DIR = BASE_DIR / "v225_formal_route_reconstruction_evidence_pack_20260622"
V226_DIR = BASE_DIR / "v226_formal_robustness_ci_audit_20260622"
V227_DIR = BASE_DIR / "v227_paper_claim_readiness_pack_20260622"

OUT_DIR = BASE_DIR / "v228_final_paper_artifact_freeze_20260623"
TABLE_DIR = OUT_DIR / "tables"
REPORT_DIR = OUT_DIR / "reports"
FIGURE_DIR = OUT_DIR / "figures"
MAIN_FIGURE_DIR = FIGURE_DIR / "selected_main_figures"
APPENDIX_FIGURE_DIR = FIGURE_DIR / "selected_appendix_figures"
LOG_DIR = OUT_DIR / "logs"

ZIP_NAME = "v228_final_paper_artifact_freeze_pack.zip"

FORMAL_MODEL_LOCK = {
    "loose_main_pool": "avg_joint_focus",
    "strict_main_pool": "peak_floor_090",
}

EXPECTED_TEST_METRICS = {
    "loose_main_pool": {
        "formal_model": "avg_joint_focus",
        "rmse": 0.5448840970647589,
        "tail_rmse": 0.6297521592665997,
        "n": 184,
    },
    "strict_main_pool": {
        "formal_model": "peak_floor_090",
        "rmse": 0.571769914574812,
        "tail_rmse": 0.6583063251135349,
        "n": 174,
    },
}

METRIC_TOLERANCE = 1e-5

# 只列入 v228 允许读取的正式来源文件。这里不读取任何诊断-only 模型输出、
# oracle/true-label/fallback 行、新预测缓存或新路由配置。
SOURCE_FILES = {
    "v225_formal_model_lock": V225_DIR / "tables" / "formal_model_lock.csv",
    "v225_pool_metrics": V225_DIR / "tables" / "formal_reconstruction_metrics_by_pool.csv",
    "v225_report": V225_DIR / "reports" / "v225_formal_route_reconstruction_evidence_cn.md",
    "v225_zip": V225_DIR / "v225_formal_route_reconstruction_evidence_pack.zip",
    "v225_metric_reproduction": V225_DIR / "logs" / "metric_reproduction_check.json",
    "v225_leakage_guard": V225_DIR / "logs" / "leakage_guard_report.json",
    "v225_forbidden_scan": V225_DIR / "logs" / "forbidden_scan_report.json",
    "v225_table_alignment": V225_DIR / "logs" / "table_alignment_check.json",
    "v226_model_lock_recheck": V226_DIR / "tables" / "formal_model_lock_recheck.csv",
    "v226_sample_ci": V226_DIR / "tables" / "formal_metric_ci_sample_bootstrap.csv",
    "v226_subject_ci": V226_DIR / "tables" / "formal_metric_ci_subject_block_bootstrap.csv",
    "v226_tail_concentration": V226_DIR / "tables" / "formal_tail_error_concentration.csv",
    "v226_readiness": V226_DIR / "tables" / "formal_readiness_decision.csv",
    "v226_report": V226_DIR / "reports" / "v226_formal_robustness_ci_audit_cn.md",
    "v226_zip": V226_DIR / "v226_formal_robustness_ci_audit_pack.zip",
    "v226_metric_reproduction": V226_DIR / "logs" / "metric_reproduction_check.json",
    "v226_leakage_guard": V226_DIR / "logs" / "leakage_guard_report.json",
    "v226_forbidden_scan": V226_DIR / "logs" / "forbidden_scan_report.json",
    "v226_table_alignment": V226_DIR / "logs" / "table_alignment_check.json",
    "v226_file_inventory": V226_DIR / "logs" / "file_inventory.json",
    "v227_main_result": V227_DIR / "tables" / "paper_main_result_table.csv",
    "v227_claim_matrix": V227_DIR / "tables" / "paper_claim_support_matrix.csv",
    "v227_limitations": V227_DIR / "tables" / "paper_limitation_table.csv",
    "v227_guardrail_summary": V227_DIR / "tables" / "formal_guardrail_summary.csv",
    "v227_figure_index": V227_DIR / "tables" / "figure_selection_index.csv",
    "v227_report": V227_DIR / "reports" / "v227_paper_claim_readiness_cn.md",
    "v227_no_model_change_guard": V227_DIR / "logs" / "no_model_change_guard.json",
    "v227_source_artifact_checks": V227_DIR / "logs" / "source_artifact_checks.json",
    "v227_file_inventory": V227_DIR / "logs" / "file_inventory.json",
    "v227_zip": V227_DIR / "v227_paper_claim_readiness_pack.zip",
}

REQUIRED_RELATIVE_FILES = [
    "tables/final_formal_model_lock.csv",
    "tables/final_main_result_table.csv",
    "tables/final_ci_table.csv",
    "tables/final_claim_lock_table.csv",
    "tables/final_limitations_table.csv",
    "tables/final_figure_selection_table.csv",
    "tables/final_artifact_manifest.csv",
    "tables/final_guardrail_summary.csv",
    "reports/v228_final_paper_artifact_freeze_cn.md",
    "reports/manuscript_results_section_draft_cn.md",
    "reports/manuscript_claim_boundary_notes_cn.md",
    "logs/run_manifest.json",
    "logs/input_file_hashes.json",
    "logs/consistency_check.json",
    "logs/forbidden_scan_report.json",
    "logs/guardrail_check.json",
    "logs/file_inventory.json",
]

# 禁用模式只用于扫描最终 formal 表。为了让 forbidden_scan_report 本身不制造文本命中，
# 日志只记录 pattern_count 和 hits，不回写这些字面量。
FORBIDDEN_PATTERNS = [
    "W3_B4_original_soft",
    "oracle_model",
    "oracle/",
    "true_label",
    "true-label",
    "true labels",
    "fallback row",
    "true-label fallback",
    "v222b",
    "v223",
    "new tau",
    "new gate",
    "new router",
    "new selector",
]


def rel(path: Path) -> str:
    """把绝对路径转换成仓库内稳定相对路径，方便报告和 manifest 复核。"""

    return str(path.relative_to(REPO_ROOT)).replace("\\", "/")


def sha256_file(path: Path) -> str:
    """计算文件 sha256，用于记录输入和输出的可追溯性。"""

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def test_zip(path: Path) -> str | None:
    with zipfile.ZipFile(path, "r") as zf:
        return zf.testzip()


def clean_output_dir() -> None:
    """只清理本轮固定输出目录，避免误删 03_baselines 下其他版本。"""

    resolved_out = OUT_DIR.resolve()
    resolved_base = BASE_DIR.resolve()
    if resolved_base not in resolved_out.parents or OUT_DIR.name != "v228_final_paper_artifact_freeze_20260623":
        raise RuntimeError(f"Refusing to clean unexpected output directory: {resolved_out}")
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    for path in [TABLE_DIR, REPORT_DIR, MAIN_FIGURE_DIR, APPENDIX_FIGURE_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def assert_sources_exist() -> None:
    missing = [name for name, path in SOURCE_FILES.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing allowed source files: {missing}")


def normalize_bool(value: object) -> bool:
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y"}


def top_level_pass(payload: object) -> bool:
    """兼容 v225/v226/v227 各种 JSON 结构，抽取最上层 pass 状态。"""

    if isinstance(payload, dict) and "pass" in payload:
        return bool(payload["pass"])
    if isinstance(payload, dict) and isinstance(payload.get("checks"), list):
        return all(bool(item.get("pass", False)) for item in payload["checks"])
    return False


def assert_formal_lock(v225_lock: pd.DataFrame, v226_lock: pd.DataFrame, v227_main: pd.DataFrame) -> None:
    """确认 v225/v226/v227 三层都没有改变 formal headline。"""

    v225_pairs = dict(zip(v225_lock["pool"], v225_lock["formal_model"]))
    v226_pairs = dict(zip(v226_lock["pool_key"], v226_lock["formal_model"]))
    v227_pairs = dict(zip(v227_main["pool_key"], v227_main["formal_model"]))
    if v225_pairs != FORMAL_MODEL_LOCK:
        raise AssertionError(f"v225 formal lock mismatch: {v225_pairs}")
    if v226_pairs != FORMAL_MODEL_LOCK:
        raise AssertionError(f"v226 formal lock mismatch: {v226_pairs}")
    if v227_pairs != FORMAL_MODEL_LOCK:
        raise AssertionError(f"v227 formal lock mismatch: {v227_pairs}")
    if "formal_lock_pass" in v226_lock and not v226_lock["formal_lock_pass"].map(normalize_bool).all():
        raise AssertionError("v226 formal_lock_pass is not all true")


def assert_metrics_match(main_result: pd.DataFrame) -> List[Dict[str, object]]:
    """用 v225/v226 锁定数值检查 v227/v228 主结果是否完全复现。"""

    checks: List[Dict[str, object]] = []
    for pool_key, expected in EXPECTED_TEST_METRICS.items():
        rows = main_result[(main_result["pool_key"] == pool_key) & (main_result["split"] == "test")]
        if len(rows) != 1:
            raise AssertionError(f"Expected one test row for {pool_key}, got {len(rows)}")
        row = rows.iloc[0]
        if row["formal_model"] != expected["formal_model"]:
            raise AssertionError(f"Unexpected formal model for {pool_key}: {row['formal_model']}")
        for metric in ["rmse", "tail_rmse"]:
            actual = float(row[metric])
            diff = abs(actual - float(expected[metric]))
            checks.append(
                {
                    "check": f"{pool_key}_{metric}_match",
                    "actual": actual,
                    "expected": float(expected[metric]),
                    "absolute_diff": diff,
                    "tolerance": METRIC_TOLERANCE,
                    "pass": diff <= METRIC_TOLERANCE,
                }
            )
        n_actual = int(row["n"])
        checks.append(
            {
                "check": f"{pool_key}_n_match",
                "actual": n_actual,
                "expected": int(expected["n"]),
                "absolute_diff": abs(n_actual - int(expected["n"])),
                "tolerance": 0,
                "pass": n_actual == int(expected["n"]),
            }
        )
    failed = [item for item in checks if not item["pass"]]
    if failed:
        raise AssertionError(f"Main metric consistency failed: {failed}")
    return checks


def assert_readiness(readiness: pd.DataFrame) -> None:
    """v228 只允许在 v226 readiness 已明确不需要新模型和新路由时生成。"""

    required = {
        "accepted_for_paper_main_result": True,
        "needs_new_model": False,
        "needs_gate_or_router": False,
    }
    for col, expected in required.items():
        if col not in readiness.columns:
            raise AssertionError(f"v226 readiness missing column: {col}")
        observed = readiness[col].map(normalize_bool).tolist()
        if any(value != expected for value in observed):
            raise AssertionError(f"v226 readiness {col} expected all {expected}, got {observed}")


def build_final_model_lock() -> pd.DataFrame:
    """GPTPro 要求该表只保留两个 pool->formal_model 锁定项。"""

    return pd.DataFrame(
        [
            {"pool_key": "loose_main_pool", "formal_model": "avg_joint_focus"},
            {"pool_key": "strict_main_pool", "formal_model": "peak_floor_090"},
        ]
    )


def build_final_main_result(v227_main: pd.DataFrame) -> pd.DataFrame:
    """从 v227 主结果收窄成最终论文主结果表，不添加任何新实验结论。"""

    keep_cols = [
        "pool_key",
        "pool_name",
        "formal_model",
        "split",
        "n",
        "rmse",
        "rmse_sample_ci_lower",
        "rmse_sample_ci_upper",
        "rmse_subject_block_ci_lower",
        "rmse_subject_block_ci_upper",
        "tail_rmse",
        "tail_rmse_sample_ci_lower",
        "tail_rmse_sample_ci_upper",
        "tail_rmse_subject_block_ci_lower",
        "tail_rmse_subject_block_ci_upper",
        "mean_sample_rmse",
        "median_sample_rmse",
        "p90_sample_rmse",
        "under_rate",
        "direction_acc",
        "strong_steer_rate",
        "extreme_peak_rate",
        "tail_top20pct_share",
        "tail_gini_proxy",
        "accepted_for_paper_main_result",
        "needs_new_model",
        "needs_gate_or_router",
    ]
    missing = [col for col in keep_cols if col not in v227_main.columns]
    if missing:
        raise AssertionError(f"v227 main result missing columns: {missing}")
    out = v227_main[keep_cols].copy()
    out["v228_freeze_status"] = "locked_from_v225_v226_v227"
    out["selection_basis"] = "pre_v228_formal_lock_only"
    out["test_used_for_v228_selection"] = False
    return out


def build_final_ci(sample_ci: pd.DataFrame, subject_ci: pd.DataFrame) -> pd.DataFrame:
    """合并 v226 sample bootstrap 与 subject-block bootstrap CI，数值不重新计算。"""

    sample = sample_ci.copy()
    sample["ci_method"] = "sample_bootstrap_from_v226"
    subject = subject_ci.copy()
    subject["ci_method"] = "subject_block_bootstrap_from_v226"
    ci = pd.concat([sample, subject], ignore_index=True, sort=False)
    ci["v228_action"] = "copied_without_recalculation"
    return ci


def build_final_claim_lock(v227_claims: pd.DataFrame) -> pd.DataFrame:
    """保留 v227 科学 claim，移除旧的过程性 bridge-blocked claim，属于收窄而非扩展。"""

    keep = v227_claims[v227_claims["claim_level"].astype(str) != "process"].copy()
    keep_cols = [
        "claim_id",
        "claim_level",
        "claim_text_cn",
        "supporting_numbers",
        "evidence_files",
        "allowed_wording_cn",
        "status",
    ]
    missing = [col for col in keep_cols if col not in keep.columns]
    if missing:
        raise AssertionError(f"v227 claim matrix missing columns: {missing}")
    out = keep[keep_cols].copy()
    out["v228_claim_action"] = "preserved_or_narrowed_from_v227"
    return out


def build_final_limitations(v227_limitations: pd.DataFrame) -> pd.DataFrame:
    """保留正式模型相关 limitations，移除旧的 GPTPro 通道 pending 过程项。"""

    keep = v227_limitations[v227_limitations["pool_key"].astype(str) != "process"].copy()
    keep_cols = [
        "limitation_id",
        "pool_key",
        "formal_model",
        "evidence",
        "impact_cn",
        "allowed_reporting_cn",
    ]
    missing = [col for col in keep_cols if col not in keep.columns]
    if missing:
        raise AssertionError(f"v227 limitation table missing columns: {missing}")
    out = keep[keep_cols].copy()
    out["v228_limitation_action"] = "preserved_from_v227_formal_model_scope"
    return out


def copy_selected_figures(v227_figures: pd.DataFrame) -> pd.DataFrame:
    """复制 v227 已筛选图，不重新作图，保证 selected figure 文件存在且非空。"""

    rows: List[Dict[str, object]] = []
    for idx, row in v227_figures.iterrows():
        src = REPO_ROOT / str(row["packaged_path"]).replace("/", "\\")
        if not src.exists():
            raise FileNotFoundError(f"Selected v227 figure missing: {src}")
        role = str(row.get("role_cn", ""))
        is_main = "ci_forest_by_pool" in role or "tail_error_concentration" in role
        dest_root = MAIN_FIGURE_DIR if is_main else APPENDIX_FIGURE_DIR
        dest = dest_root / f"{int(idx) + 1:03d}_{src.name}"
        shutil.copy2(src, dest)
        if dest.stat().st_size <= 0:
            raise AssertionError(f"Copied figure is empty: {dest}")
        rows.append(
            {
                "figure_id": row.get("figure_id", f"F{int(idx) + 1:03d}"),
                "figure_set": "main" if is_main else "appendix",
                "source_v227_path": rel(src),
                "v228_path": rel(dest),
                "role_cn": role,
                "caption_cn": row.get("caption_cn", ""),
                "bytes": dest.stat().st_size,
                "sha256": sha256_file(dest),
            }
        )
    if not rows:
        raise AssertionError("No selected figures copied")
    return pd.DataFrame(rows)


def build_final_guardrail_summary(
    final_lock: pd.DataFrame,
    final_main: pd.DataFrame,
    final_claims: pd.DataFrame,
    final_limits: pd.DataFrame,
    final_figures: pd.DataFrame,
) -> pd.DataFrame:
    """生成面向论文冻结包的高层 guardrail 汇总，不把诊断模型文本写进 formal 表。"""

    source_checks = [
        ("v225_metric_reproduction", SOURCE_FILES["v225_metric_reproduction"]),
        ("v225_leakage_guard", SOURCE_FILES["v225_leakage_guard"]),
        ("v225_forbidden_scan", SOURCE_FILES["v225_forbidden_scan"]),
        ("v225_table_alignment", SOURCE_FILES["v225_table_alignment"]),
        ("v226_metric_reproduction", SOURCE_FILES["v226_metric_reproduction"]),
        ("v226_leakage_guard", SOURCE_FILES["v226_leakage_guard"]),
        ("v226_forbidden_scan", SOURCE_FILES["v226_forbidden_scan"]),
        ("v226_table_alignment", SOURCE_FILES["v226_table_alignment"]),
        ("v227_no_model_change_guard", SOURCE_FILES["v227_no_model_change_guard"]),
        ("v227_source_artifact_checks", SOURCE_FILES["v227_source_artifact_checks"]),
    ]
    rows = []
    for check_id, path in source_checks:
        rows.append(
            {
                "check_id": check_id,
                "source": rel(path),
                "pass": top_level_pass(read_json(path)),
                "detail_cn": "来源日志顶层通过，v228 未重新计算或重写来源结论。",
            }
        )
    rows.extend(
        [
            {
                "check_id": "v228_formal_model_lock_exact",
                "source": "v228",
                "pass": dict(zip(final_lock["pool_key"], final_lock["formal_model"])) == FORMAL_MODEL_LOCK,
                "detail_cn": "最终锁定模型只包含两个正式主池。",
            },
            {
                "check_id": "v228_main_result_rows",
                "source": "v228",
                "pass": len(final_main) == 2,
                "detail_cn": "最终主结果表只包含 loose 与 strict 两行 test 主结果。",
            },
            {
                "check_id": "v228_claims_narrowed",
                "source": "v228",
                "pass": set(final_claims["claim_id"]).issubset({"C1_formal_lock", "C2_test_main_metrics", "C3_ci_robustness", "C4_tail_limitation", "C5_no_new_model_needed"}),
                "detail_cn": "最终 claim lock 未新增科学 claim，只保留 v227 中可写入论文的正式 claim。",
            },
            {
                "check_id": "v228_limitations_formal_scope",
                "source": "v228",
                "pass": not (final_limits["pool_key"].astype(str) == "process").any(),
                "detail_cn": "最终 limitations 只保留正式模型相关限制。",
            },
            {
                "check_id": "v228_selected_figures_nonempty",
                "source": "v228",
                "pass": bool((final_figures["bytes"].astype(int) > 0).all()),
                "detail_cn": "所有复制图文件均存在且非空。",
            },
        ]
    )
    out = pd.DataFrame(rows)
    if not out["pass"].astype(bool).all():
        failed = out[~out["pass"].astype(bool)].to_dict(orient="records")
        raise AssertionError(f"Guardrail summary has failed rows: {failed}")
    return out


def build_artifact_manifest(input_hashes: Dict[str, Dict[str, object]]) -> pd.DataFrame:
    """先记录输入 manifest，输出文件生成后会追加 v228 文件清单。"""

    rows = []
    for artifact_id, item in input_hashes.items():
        rows.append(
            {
                "artifact_id": artifact_id,
                "artifact_scope": "allowed_source_input",
                "path": item["path"],
                "bytes": item["bytes"],
                "sha256": item["sha256"],
                "role": infer_role(artifact_id),
            }
        )
    return pd.DataFrame(rows)


def infer_role(artifact_id: str) -> str:
    if "zip" in artifact_id:
        return "source_package"
    if "lock" in artifact_id:
        return "formal_lock"
    if "ci" in artifact_id:
        return "confidence_interval"
    if "guard" in artifact_id or "scan" in artifact_id or "alignment" in artifact_id:
        return "guardrail_log"
    if "report" in artifact_id:
        return "source_report"
    if "figure" in artifact_id:
        return "figure_manifest"
    return "source_table"


def build_input_hashes() -> Dict[str, Dict[str, object]]:
    return {
        name: {
            "path": rel(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for name, path in SOURCE_FILES.items()
    }


def scan_forbidden_outputs() -> Dict[str, object]:
    """扫描最终 formal 表，确认没有把禁用模型/标签/诊断行写入正式表。"""

    scan_files = [
        TABLE_DIR / "final_formal_model_lock.csv",
        TABLE_DIR / "final_main_result_table.csv",
        TABLE_DIR / "final_ci_table.csv",
        TABLE_DIR / "final_claim_lock_table.csv",
        TABLE_DIR / "final_limitations_table.csv",
        TABLE_DIR / "final_figure_selection_table.csv",
        TABLE_DIR / "final_artifact_manifest.csv",
    ]
    hits: List[Dict[str, object]] = []
    for path in scan_files:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        lowered = text.lower()
        for pattern in FORBIDDEN_PATTERNS:
            if pattern.lower() in lowered:
                hits.append({"file": rel(path), "pattern_id": f"pattern_{len(hits) + 1:03d}"})
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "pass": len(hits) == 0,
        "scan_scope": [rel(path) for path in scan_files],
        "pattern_count": len(FORBIDDEN_PATTERNS),
        "hits": hits,
    }


def build_consistency_check(
    metric_checks: List[Dict[str, object]],
    v227_claims: pd.DataFrame,
    final_claims: pd.DataFrame,
    v227_limitations: pd.DataFrame,
    final_limits: pd.DataFrame,
    final_figures: pd.DataFrame,
) -> Dict[str, object]:
    """集中记录 v228 没有扩展 claim、没有改变指标、没有扩大 limitations。"""

    claim_subset = set(final_claims["claim_id"]).issubset(set(v227_claims["claim_id"]))
    claim_not_expanded = len(final_claims) <= len(v227_claims)
    limitation_subset = set(final_limits["limitation_id"]).issubset(set(v227_limitations["limitation_id"]))
    limitation_not_expanded = len(final_limits) <= len(v227_limitations)
    figure_nonempty = bool((final_figures["bytes"].astype(int) > 0).all()) and len(final_figures) > 0
    checks = [
        {"check": "main_test_metrics_match_locked_values", "pass": all(item["pass"] for item in metric_checks)},
        {"check": "v227_claim_matrix_preserved_or_narrowed", "pass": claim_subset and claim_not_expanded},
        {"check": "v227_limitations_preserved_or_narrowed", "pass": limitation_subset and limitation_not_expanded},
        {"check": "selected_figure_files_exist_and_nonempty", "pass": figure_nonempty},
    ]
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "pass": all(item["pass"] for item in checks),
        "metric_checks": metric_checks,
        "checks": checks,
        "final_claim_ids": final_claims["claim_id"].tolist(),
        "final_limitation_ids": final_limits["limitation_id"].tolist(),
        "selected_figure_count": int(len(final_figures)),
    }
    if not payload["pass"]:
        raise AssertionError(f"Consistency check failed: {payload}")
    return payload


def build_guardrail_check(forbidden_report: Dict[str, object], consistency: Dict[str, object]) -> Dict[str, object]:
    """v228 自身的停止条件检查。所有 check 失败时只允许停止报错，不允许 repair。"""

    checks = [
        {
            "check": "reporting_packaging_only",
            "pass": True,
            "detail": "v228 only reads existing formal/reporting artifacts and writes final package files.",
        },
        {
            "check": "no_model_work_executed",
            "pass": True,
            "detail": "No fit/train/predict/calibration/search branch exists in this script.",
        },
        {
            "check": "formal_headline_unchanged",
            "pass": True,
            "detail": "Final formal lock remains loose=avg_joint_focus and strict=peak_floor_090.",
        },
        {
            "check": "forbidden_scan_clean",
            "pass": bool(forbidden_report["pass"]),
            "detail": f"final formal table hits={len(forbidden_report['hits'])}",
        },
        {
            "check": "consistency_check_clean",
            "pass": bool(consistency["pass"]),
            "detail": "metrics, claims, limitations, and figures match the frozen source boundary.",
        },
    ]
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "pass": all(item["pass"] for item in checks),
        "checks": checks,
    }


def write_reports(final_main: pd.DataFrame, final_claims: pd.DataFrame, final_limits: pd.DataFrame, final_figures: pd.DataFrame) -> None:
    """写出三份中文报告，供论文/组会/下一轮 GPTPro 回报直接引用。"""

    loose = final_main[final_main["pool_key"] == "loose_main_pool"].iloc[0]
    strict = final_main[final_main["pool_key"] == "strict_main_pool"].iloc[0]
    main_fig_count = int((final_figures["figure_set"] == "main").sum())
    appendix_fig_count = int((final_figures["figure_set"] == "appendix").sum())

    freeze_report = f"""# v228 最终论文产物冻结报告

## 结论

v228 按本地 GPTPro 软件端的有效回复执行：接受 v227 作为 reporting-only closeout，并生成最终论文产物冻结包。该版本没有训练模型、没有生成新预测、没有搜索阈值、没有创建新路由/选择器，也没有改变 formal headline。

最终 formal model lock 仍为：

| pool | formal model |
|---|---|
| loose_main_pool | avg_joint_focus |
| strict_main_pool | peak_floor_090 |

## 主结果冻结

| pool | formal model | test n | RMSE | tail RMSE | sample RMSE CI | subject-block RMSE CI |
|---|---|---:|---:|---:|---|---|
| loose_main_pool | avg_joint_focus | {int(loose['n'])} | {float(loose['rmse']):.6f} | {float(loose['tail_rmse']):.6f} | {float(loose['rmse_sample_ci_lower']):.6f}-{float(loose['rmse_sample_ci_upper']):.6f} | {float(loose['rmse_subject_block_ci_lower']):.6f}-{float(loose['rmse_subject_block_ci_upper']):.6f} |
| strict_main_pool | peak_floor_090 | {int(strict['n'])} | {float(strict['rmse']):.6f} | {float(strict['tail_rmse']):.6f} | {float(strict['rmse_sample_ci_lower']):.6f}-{float(strict['rmse_sample_ci_upper']):.6f} | {float(strict['rmse_subject_block_ci_lower']):.6f}-{float(strict['rmse_subject_block_ci_upper']):.6f} |

## claim 与 limitation 边界

- final claim lock 保留 {len(final_claims)} 条 v227 已有正式 claim，移除了旧的过程性通道阻塞 claim。
- final limitations 保留 {len(final_limits)} 条正式模型相关 limitation，移除了旧的过程性 pending 项。
- 图文件从 v227 已选图复制而来：主图 {main_fig_count} 张，附录图 {appendix_fig_count} 张。

## 输出文件

- `tables/final_formal_model_lock.csv`
- `tables/final_main_result_table.csv`
- `tables/final_ci_table.csv`
- `tables/final_claim_lock_table.csv`
- `tables/final_limitations_table.csv`
- `tables/final_figure_selection_table.csv`
- `tables/final_artifact_manifest.csv`
- `tables/final_guardrail_summary.csv`
- `reports/manuscript_results_section_draft_cn.md`
- `reports/manuscript_claim_boundary_notes_cn.md`
- `logs/consistency_check.json`
- `logs/forbidden_scan_report.json`
- `logs/guardrail_check.json`
- `{ZIP_NAME}`
"""

    results_draft = f"""# 论文结果段落草稿

在最终 formal reconstruction 设置下，主结果锁定为两个预先确认的主池配置。loose_main_pool 使用 `avg_joint_focus`，在 test split 上获得 RMSE={float(loose['rmse']):.6f}、tail RMSE={float(loose['tail_rmse']):.6f}，sample bootstrap RMSE 95% CI 为 {float(loose['rmse_sample_ci_lower']):.6f}-{float(loose['rmse_sample_ci_upper']):.6f}，subject-block bootstrap RMSE 95% CI 为 {float(loose['rmse_subject_block_ci_lower']):.6f}-{float(loose['rmse_subject_block_ci_upper']):.6f}。

strict_main_pool 使用 `peak_floor_090`，在 test split 上获得 RMSE={float(strict['rmse']):.6f}、tail RMSE={float(strict['tail_rmse']):.6f}，sample bootstrap RMSE 95% CI 为 {float(strict['rmse_sample_ci_lower']):.6f}-{float(strict['rmse_sample_ci_upper']):.6f}，subject-block bootstrap RMSE 95% CI 为 {float(strict['rmse_subject_block_ci_lower']):.6f}-{float(strict['rmse_subject_block_ci_upper']):.6f}。

这些结果来自 v225 的 formal route reconstruction evidence pack 与 v226 的 robustness / CI audit；v228 只做最终表格、图件、claim 和日志冻结，不引入新的模型结果。
"""

    claim_notes = """# manuscript claim boundary notes

## 可以写入正文的边界

- 可以写：formal headline 已锁定为 loose=avg_joint_focus 与 strict=peak_floor_090。
- 可以写：主结果数值与 v225/v226 完全复现，并附带 sample bootstrap 与 subject-block bootstrap 区间。
- 可以写：tail error 集中度和 underestimation 是 limitation，需要在论文中透明呈现。
- 可以写：v228 是最终论文产物冻结，不是新实验。

## 不可以写的边界

- 不可以声称 v228 训练了新模型或改进了模型。
- 不可以把 reporting-only 表格整理解释成新的 leaderboard 提升。
- 不可以把诊断行、人工标签、oracle 类标签或回退行写入 formal evidence。
- 不可以基于 test split 重新选择模型、阈值、样本或 claim。

## 本轮 GPTPro 指令状态

本轮通过本地 ChatGPT 软件端取得有效 GPTPro 回复。GPTPro 接受 v227 作为 reporting-only closeout，并要求 v228 只做论文产物冻结；任何锁定指标、模型名、CI 值、guardrail 状态或 claim 与 v225/v226/v227 冲突时，都应停止并输出 failure report，而不是修补模型或扩展实验边界。
"""

    (REPORT_DIR / "v228_final_paper_artifact_freeze_cn.md").write_text(freeze_report, encoding="utf-8")
    (REPORT_DIR / "manuscript_results_section_draft_cn.md").write_text(results_draft, encoding="utf-8")
    (REPORT_DIR / "manuscript_claim_boundary_notes_cn.md").write_text(claim_notes, encoding="utf-8")


def list_output_files() -> List[Dict[str, object]]:
    files = []
    for path in sorted(OUT_DIR.rglob("*")):
        if path.is_file() and path.name != ZIP_NAME:
            files.append(
                {
                    "path": str(path.relative_to(OUT_DIR)).replace("\\", "/"),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return files


def append_outputs_to_manifest(input_manifest: pd.DataFrame) -> pd.DataFrame:
    rows = input_manifest.to_dict(orient="records")
    for item in list_output_files():
        rows.append(
            {
                "artifact_id": "v228_" + item["path"].replace("/", "_"),
                "artifact_scope": "v228_output",
                "path": rel(OUT_DIR / item["path"]),
                "bytes": item["bytes"],
                "sha256": item["sha256"],
                "role": "final_output",
            }
        )
    return pd.DataFrame(rows)


def write_file_inventory(zip_bad_file: str | None) -> Dict[str, object]:
    existing = {item["path"] for item in list_output_files()}
    inventory = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "file_count_excluding_zip": len(existing),
        "required_files_missing": sorted(set(REQUIRED_RELATIVE_FILES) - existing),
        "zip_bad_file": zip_bad_file,
        "files": list_output_files(),
    }
    write_json(LOG_DIR / "file_inventory.json", inventory)
    return inventory


def create_zip() -> Path:
    zip_path = OUT_DIR / ZIP_NAME
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(OUT_DIR.rglob("*")):
            if path.is_file() and path != zip_path:
                zf.write(path, path.relative_to(OUT_DIR))
    return zip_path


def main() -> None:
    clean_output_dir()
    assert_sources_exist()

    v225_lock = pd.read_csv(SOURCE_FILES["v225_formal_model_lock"])
    v226_lock = pd.read_csv(SOURCE_FILES["v226_model_lock_recheck"])
    v227_main = pd.read_csv(SOURCE_FILES["v227_main_result"])
    v226_sample_ci = pd.read_csv(SOURCE_FILES["v226_sample_ci"])
    v226_subject_ci = pd.read_csv(SOURCE_FILES["v226_subject_ci"])
    v226_readiness = pd.read_csv(SOURCE_FILES["v226_readiness"])
    v227_claims = pd.read_csv(SOURCE_FILES["v227_claim_matrix"])
    v227_limitations = pd.read_csv(SOURCE_FILES["v227_limitations"])
    v227_figures = pd.read_csv(SOURCE_FILES["v227_figure_index"])

    assert_formal_lock(v225_lock, v226_lock, v227_main)
    assert_readiness(v226_readiness)

    final_lock = build_final_model_lock()
    final_main = build_final_main_result(v227_main)
    metric_checks = assert_metrics_match(final_main)
    final_ci = build_final_ci(v226_sample_ci, v226_subject_ci)
    final_claims = build_final_claim_lock(v227_claims)
    final_limits = build_final_limitations(v227_limitations)
    final_figures = copy_selected_figures(v227_figures)
    input_hashes = build_input_hashes()
    input_manifest = build_artifact_manifest(input_hashes)
    final_guardrails = build_final_guardrail_summary(final_lock, final_main, final_claims, final_limits, final_figures)

    write_csv(final_lock, TABLE_DIR / "final_formal_model_lock.csv")
    write_csv(final_main, TABLE_DIR / "final_main_result_table.csv")
    write_csv(final_ci, TABLE_DIR / "final_ci_table.csv")
    write_csv(final_claims, TABLE_DIR / "final_claim_lock_table.csv")
    write_csv(final_limits, TABLE_DIR / "final_limitations_table.csv")
    write_csv(final_figures, TABLE_DIR / "final_figure_selection_table.csv")
    write_csv(final_guardrails, TABLE_DIR / "final_guardrail_summary.csv")
    write_reports(final_main, final_claims, final_limits, final_figures)

    # 先写初版 manifest，随后在输出文件生成后追加 v228 自身输出清单。
    write_csv(input_manifest, TABLE_DIR / "final_artifact_manifest.csv")

    forbidden_report = scan_forbidden_outputs()
    if not forbidden_report["pass"]:
        raise AssertionError(f"Forbidden formal table scan failed: {forbidden_report['hits']}")
    consistency = build_consistency_check(metric_checks, v227_claims, final_claims, v227_limitations, final_limits, final_figures)
    guardrail_check = build_guardrail_check(forbidden_report, consistency)
    if not guardrail_check["pass"]:
        raise AssertionError(f"Guardrail check failed: {guardrail_check}")

    write_json(
        LOG_DIR / "run_manifest.json",
        {
            "version": "v228_final_paper_artifact_freeze_20260623",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "scope": "final_paper_artifact_freeze_reporting_packaging_only",
            "gptpro_instruction": "local_chatgpt_desktop_clean_ascii_reply",
            "source_versions": ["v225", "v226", "v227"],
            "formal_model_lock": FORMAL_MODEL_LOCK,
            "no_model_training": True,
            "no_prediction_generation": True,
            "no_threshold_search": True,
            "no_formal_headline_change": True,
            "output_dir": rel(OUT_DIR),
        },
    )
    write_json(LOG_DIR / "input_file_hashes.json", input_hashes)
    write_json(LOG_DIR / "forbidden_scan_report.json", forbidden_report)
    write_json(LOG_DIR / "consistency_check.json", consistency)
    write_json(LOG_DIR / "guardrail_check.json", guardrail_check)

    final_manifest = append_outputs_to_manifest(input_manifest)
    write_csv(final_manifest, TABLE_DIR / "final_artifact_manifest.csv")

    # inventory 需要包含最终 manifest，所以在 manifest 更新后生成。
    write_file_inventory(zip_bad_file=None)
    zip_path = create_zip()
    bad_file = test_zip(zip_path)
    if bad_file is not None:
        raise AssertionError(f"ZIP integrity failed at {bad_file}")

    inventory = write_file_inventory(zip_bad_file=bad_file)
    if inventory["required_files_missing"]:
        raise AssertionError(f"Required files missing: {inventory['required_files_missing']}")

    # 重新打包一次，让 ZIP 内部的 file_inventory 也包含最终 zip_bad_file 状态。
    zip_path = create_zip()
    final_bad_file = test_zip(zip_path)
    if final_bad_file is not None:
        raise AssertionError(f"Final ZIP integrity failed at {final_bad_file}")

    print(
        json.dumps(
            {
                "ok": True,
                "output_dir": rel(OUT_DIR),
                "zip_path": rel(zip_path),
                "zip_bad_file": final_bad_file,
                "required_files_missing": inventory["required_files_missing"],
                "main_result_rows": len(final_main),
                "claim_rows": len(final_claims),
                "limitation_rows": len(final_limits),
                "selected_main_figures": int((final_figures["figure_set"] == "main").sum()),
                "selected_appendix_figures": int((final_figures["figure_set"] == "appendix").sum()),
                "forbidden_scan_hits": len(forbidden_report["hits"]),
                "guardrail_pass": guardrail_check["pass"],
                "consistency_pass": consistency["pass"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
