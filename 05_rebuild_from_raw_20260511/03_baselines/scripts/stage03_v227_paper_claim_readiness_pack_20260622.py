#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v227 paper / claim readiness pack.

本脚本只做写作材料整理，不训练模型、不搜索阈值、不改变 formal headline。

输入范围：
1. v225 已锁定的 formal route reconstruction evidence pack；
2. v226 已完成的 formal robustness / CI audit；
3. 本轮 GPTPro 回报阻塞记录。

输出目标：
1. 把 v225 的主结果、v226 的置信区间和稳定性证据合并成论文可读的主结果表；
2. 生成 claim support matrix、limitations、guardrail summary 和 figure index；
3. 复制一小组已经存在的代表性图到 v227 包内，便于后续写作/汇报引用；
4. 生成中文报告、文件清单、ZIP，并验证 ZIP 完整性。

重要边界：
- 不读取 diagnostic-only row 作为 formal model；
- 不启动 v222b / v223；
- 不产生 gate / router / tau / threshold / selector；
- 不使用 test 表现反推新的模型配置；
- v227 的结论只能是“已锁定结果可进入写作材料整理”，不能声称新模型改进。
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
GPTPRO_DIR = REPO_ROOT / "gptpro_reviews"

V225_DIR = BASE_DIR / "v225_formal_route_reconstruction_evidence_pack_20260622"
V226_DIR = BASE_DIR / "v226_formal_robustness_ci_audit_20260622"

OUT_DIR = BASE_DIR / "v227_paper_claim_readiness_pack_20260622"
TABLE_DIR = OUT_DIR / "tables"
FIGURE_DIR = OUT_DIR / "figures"
REPORT_DIR = OUT_DIR / "reports"
LOG_DIR = OUT_DIR / "logs"

FORMAL_MODEL_LOCK = {
    "loose_main_pool": "avg_joint_focus",
    "strict_main_pool": "peak_floor_090",
}

SOURCE_FILES = {
    "v225_formal_model_lock": V225_DIR / "tables" / "formal_model_lock.csv",
    "v225_pool_metrics": V225_DIR / "tables" / "formal_reconstruction_metrics_by_pool.csv",
    "v225_bucket_metrics": V225_DIR / "tables" / "formal_reconstruction_metrics_by_bucket.csv",
    "v225_route_event_metrics": V225_DIR / "tables" / "formal_reconstruction_metrics_by_route_event.csv",
    "v225_failure_cases": V225_DIR / "tables" / "formal_failure_case_index.csv",
    "v225_report": V225_DIR / "reports" / "v225_formal_route_reconstruction_evidence_cn.md",
    "v225_zip": V225_DIR / "v225_formal_route_reconstruction_evidence_pack.zip",
    "v225_metric_reproduction": V225_DIR / "logs" / "metric_reproduction_check.json",
    "v225_leakage_guard": V225_DIR / "logs" / "leakage_guard_report.json",
    "v225_forbidden_scan": V225_DIR / "logs" / "forbidden_scan_report.json",
    "v225_table_alignment": V225_DIR / "logs" / "table_alignment_check.json",
    "v226_model_lock_recheck": V226_DIR / "tables" / "formal_model_lock_recheck.csv",
    "v226_sample_ci": V226_DIR / "tables" / "formal_metric_ci_sample_bootstrap.csv",
    "v226_subject_ci": V226_DIR / "tables" / "formal_metric_ci_subject_block_bootstrap.csv",
    "v226_subject_metrics": V226_DIR / "tables" / "formal_subject_level_metrics.csv",
    "v226_route_event_metrics": V226_DIR / "tables" / "formal_route_event_level_metrics.csv",
    "v226_bucket_ci": V226_DIR / "tables" / "formal_bucket_ci_metrics.csv",
    "v226_tail_concentration": V226_DIR / "tables" / "formal_tail_error_concentration.csv",
    "v226_underestimation": V226_DIR / "tables" / "formal_underestimation_profile.csv",
    "v226_extreme_peak": V226_DIR / "tables" / "formal_extreme_peak_profile.csv",
    "v226_sample_influence": V226_DIR / "tables" / "formal_sample_influence_audit.csv",
    "v226_readiness": V226_DIR / "tables" / "formal_readiness_decision.csv",
    "v226_report": V226_DIR / "reports" / "v226_formal_robustness_ci_audit_cn.md",
    "v226_zip": V226_DIR / "v226_formal_robustness_ci_audit_pack.zip",
    "v226_metric_reproduction": V226_DIR / "logs" / "metric_reproduction_check.json",
    "v226_leakage_guard": V226_DIR / "logs" / "leakage_guard_report.json",
    "v226_forbidden_scan": V226_DIR / "logs" / "forbidden_scan_report.json",
    "v226_table_alignment": V226_DIR / "logs" / "table_alignment_check.json",
    "v226_file_inventory": V226_DIR / "logs" / "file_inventory.json",
    "gptpro_blocked_response": GPTPRO_DIR / "20260622_v226_result_gptpro_response_blocked.md",
    "gptpro_blocked_decision": GPTPRO_DIR / "20260622_v226_result_gptpro_decision_blocked.md",
    "gptpro_blocked_action_items": GPTPRO_DIR / "20260622_v226_result_gptpro_action_items_blocked.md",
}


def ensure_clean_output() -> None:
    """重建 v227 输出目录。只删除本脚本自己的固定输出目录。"""
    if OUT_DIR.exists():
        if OUT_DIR.name != "v227_paper_claim_readiness_pack_20260622":
            raise RuntimeError(f"Refusing to clean unexpected output directory: {OUT_DIR}")
        shutil.rmtree(OUT_DIR)
    for path in [TABLE_DIR, FIGURE_DIR, REPORT_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)).replace("\\", "/")


def assert_sources_exist() -> None:
    missing = [name for name, path in SOURCE_FILES.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing source files: {missing}")


def test_zip(path: Path) -> str | None:
    with zipfile.ZipFile(path, "r") as zf:
        return zf.testzip()


def assert_formal_lock(v225_lock: pd.DataFrame, v226_lock: pd.DataFrame) -> None:
    """确认 v225/v226 都只承认两条锁定 formal headline。"""
    v225_pairs = dict(zip(v225_lock["pool"], v225_lock["formal_model"]))
    v226_pairs = dict(zip(v226_lock["pool_key"], v226_lock["formal_model"]))
    if v225_pairs != FORMAL_MODEL_LOCK:
        raise AssertionError(f"v225 formal lock mismatch: {v225_pairs}")
    if v226_pairs != FORMAL_MODEL_LOCK:
        raise AssertionError(f"v226 formal lock mismatch: {v226_pairs}")
    if not v226_lock["formal_lock_pass"].astype(bool).all():
        raise AssertionError("v226 formal_lock_pass is not all true")
    if not v226_lock["source_lock_pass"].astype(bool).all():
        raise AssertionError("v226 source_lock_pass is not all true")


def assert_v226_readiness(readiness: pd.DataFrame) -> None:
    """v227 只允许在 v226 readiness 全部接受时整理写作包。"""
    required = {
        "accepted_for_paper_main_result": True,
        "needs_new_model": False,
        "needs_gate_or_router": False,
    }
    for col, expected in required.items():
        observed = readiness[col].astype(bool).tolist()
        if any(value != expected for value in observed):
            raise AssertionError(f"v226 readiness {col} expected all {expected}, got {observed}")


def ci_lookup(ci: pd.DataFrame, pool_key: str, split: str, metric: str) -> Dict[str, float | str | int]:
    rows = ci[(ci["pool_key"] == pool_key) & (ci["split"] == split) & (ci["metric"] == metric)]
    if len(rows) != 1:
        raise AssertionError(f"Expected one CI row for {pool_key}/{split}/{metric}, got {len(rows)}")
    return rows.iloc[0].to_dict()


def fmt_num(value: float, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def make_main_result_table(
    pool_metrics: pd.DataFrame,
    sample_ci: pd.DataFrame,
    subject_ci: pd.DataFrame,
    tail_concentration: pd.DataFrame,
    readiness: pd.DataFrame,
) -> pd.DataFrame:
    """生成论文主结果表：主指标 + sample CI + subject-block CI + tail 集中度。"""
    rows: List[Dict[str, object]] = []
    test_metrics = pool_metrics[pool_metrics["split"] == "test"].copy()
    for _, metric_row in test_metrics.iterrows():
        pool_key = metric_row["pool_key"]
        formal_model = metric_row["formal_model"]
        if FORMAL_MODEL_LOCK[pool_key] != formal_model:
            raise AssertionError(f"Unexpected formal model in test metrics: {pool_key}/{formal_model}")

        sample_rmse = ci_lookup(sample_ci, pool_key, "test", "rmse")
        sample_tail = ci_lookup(sample_ci, pool_key, "test", "tail_rmse")
        subject_rmse = ci_lookup(subject_ci, pool_key, "test", "rmse")
        subject_tail = ci_lookup(subject_ci, pool_key, "test", "tail_rmse")
        tail_row = tail_concentration[
            (tail_concentration["pool_key"] == pool_key) & (tail_concentration["split"] == "test")
        ]
        if len(tail_row) != 1:
            raise AssertionError(f"Expected one tail concentration row for {pool_key}/test")
        ready_row = readiness[readiness["formal_model"] == formal_model]
        if len(ready_row) != 1:
            raise AssertionError(f"Expected one readiness row for {formal_model}")
        tail_row = tail_row.iloc[0]
        ready_row = ready_row.iloc[0]

        rows.append(
            {
                "pool_key": pool_key,
                "pool_name": metric_row["pool_name"],
                "formal_model": formal_model,
                "split": "test",
                "n": int(metric_row["n"]),
                "rmse": metric_row["rmse"],
                "rmse_sample_ci_lower": sample_rmse["ci_lower"],
                "rmse_sample_ci_upper": sample_rmse["ci_upper"],
                "rmse_subject_block_ci_lower": subject_rmse["ci_lower"],
                "rmse_subject_block_ci_upper": subject_rmse["ci_upper"],
                "tail_rmse": metric_row["tail_rmse"],
                "tail_rmse_sample_ci_lower": sample_tail["ci_lower"],
                "tail_rmse_sample_ci_upper": sample_tail["ci_upper"],
                "tail_rmse_subject_block_ci_lower": subject_tail["ci_lower"],
                "tail_rmse_subject_block_ci_upper": subject_tail["ci_upper"],
                "mean_sample_rmse": metric_row["mean_sample_rmse"],
                "median_sample_rmse": metric_row["median_sample_rmse"],
                "p90_sample_rmse": metric_row["p90_sample_rmse"],
                "under_rate": metric_row["under_rate"],
                "direction_acc": metric_row["direction_acc"],
                "strong_steer_rate": metric_row["strong_steer_rate"],
                "extreme_peak_rate": metric_row["extreme_peak_rate"],
                "tail_top20pct_share": tail_row["top20pct_share"],
                "tail_gini_proxy": tail_row["gini_tail_sse_proxy"],
                "max_sample_tail_rmse": tail_row["max_sample_tail_rmse"],
                "max_sample_id": tail_row["max_sample_id"],
                "accepted_for_paper_main_result": bool(ready_row["accepted_for_paper_main_result"]),
                "needs_new_model": bool(ready_row["needs_new_model"]),
                "needs_gate_or_router": bool(ready_row["needs_gate_or_router"]),
            }
        )
    return pd.DataFrame(rows)


def make_claim_support_matrix(main_result: pd.DataFrame) -> pd.DataFrame:
    """把可写入论文/汇报的 claim 与证据、限制、禁用措辞绑定。"""
    loose = main_result[main_result["pool_key"] == "loose_main_pool"].iloc[0]
    strict = main_result[main_result["pool_key"] == "strict_main_pool"].iloc[0]
    rows = [
        {
            "claim_id": "C1_formal_lock",
            "claim_level": "main",
            "claim_text_cn": "formal 主结果已经锁定为 loose=avg_joint_focus、strict=peak_floor_090。",
            "supporting_numbers": "v225 formal_model_lock 与 v226 formal_model_lock_recheck 完全一致。",
            "evidence_files": "v225 tables/formal_model_lock.csv; v226 tables/formal_model_lock_recheck.csv",
            "allowed_wording_cn": "可以说 formal headline 已锁定，后续只围绕证据整理和写作。",
            "limitation_cn": "不能据此声称新模型优于所有历史诊断模型。",
            "status": "accepted",
        },
        {
            "claim_id": "C2_test_main_metrics",
            "claim_level": "main",
            "claim_text_cn": "两个主池的 test 主指标已复现并可作为论文主结果候选。",
            "supporting_numbers": (
                f"loose RMSE={fmt_num(loose['rmse'])}, tail={fmt_num(loose['tail_rmse'])}; "
                f"strict RMSE={fmt_num(strict['rmse'])}, tail={fmt_num(strict['tail_rmse'])}."
            ),
            "evidence_files": "v225 tables/formal_reconstruction_metrics_by_pool.csv; v226 logs/metric_reproduction_check.json",
            "allowed_wording_cn": "可以写成 locked formal result with exact metric reproduction。",
            "limitation_cn": "不能把 reporting-only 复现解释成新的训练收益。",
            "status": "accepted",
        },
        {
            "claim_id": "C3_ci_robustness",
            "claim_level": "main",
            "claim_text_cn": "v226 提供了 sample bootstrap 与 subject-block bootstrap 置信区间。",
            "supporting_numbers": (
                f"loose sample RMSE CI={fmt_num(loose['rmse_sample_ci_lower'])}-"
                f"{fmt_num(loose['rmse_sample_ci_upper'])}, subject-block RMSE CI="
                f"{fmt_num(loose['rmse_subject_block_ci_lower'])}-{fmt_num(loose['rmse_subject_block_ci_upper'])}; "
                f"strict sample RMSE CI={fmt_num(strict['rmse_sample_ci_lower'])}-"
                f"{fmt_num(strict['rmse_sample_ci_upper'])}, subject-block RMSE CI="
                f"{fmt_num(strict['rmse_subject_block_ci_lower'])}-{fmt_num(strict['rmse_subject_block_ci_upper'])}."
            ),
            "evidence_files": "v226 tables/formal_metric_ci_sample_bootstrap.csv; v226 tables/formal_metric_ci_subject_block_bootstrap.csv",
            "allowed_wording_cn": "可以把不确定性作为主结果的统计区间报告。",
            "limitation_cn": "subject-block CI 仍反映被试层面不确定性，不能写成无跨被试差异。",
            "status": "accepted",
        },
        {
            "claim_id": "C4_tail_limitation",
            "claim_level": "limitation",
            "claim_text_cn": "tail error 仍然集中在少量样本上，应作为 limitation 报告。",
            "supporting_numbers": (
                f"test top20pct tail-SSE share: loose={fmt_num(loose['tail_top20pct_share'])}, "
                f"strict={fmt_num(strict['tail_top20pct_share'])}; tail gini proxy: "
                f"loose={fmt_num(loose['tail_gini_proxy'])}, strict={fmt_num(strict['tail_gini_proxy'])}."
            ),
            "evidence_files": "v226 tables/formal_tail_error_concentration.csv",
            "allowed_wording_cn": "可以说剩余误差主要体现为尾部困难样本集中。",
            "limitation_cn": "不能据此重新解锁 gate/router 或 test-based 修补。",
            "status": "accepted_as_limitation",
        },
        {
            "claim_id": "C5_no_new_model_needed",
            "claim_level": "decision",
            "claim_text_cn": "v226 readiness 显示当前证据进入写作整理阶段，不需要新模型或 gate/router。",
            "supporting_numbers": "accepted_for_paper_main_result=True; needs_new_model=False; needs_gate_or_router=False.",
            "evidence_files": "v226 tables/formal_readiness_decision.csv",
            "allowed_wording_cn": "可以说下一步是写作/claim framing，而不是继续本地模型搜索。",
            "limitation_cn": "如果未来引入新数据或新 formal target，需要重新审计边界。",
            "status": "accepted",
        },
        {
            "claim_id": "C6_gptpro_bridge_blocked",
            "claim_level": "process",
            "claim_text_cn": "v226 结果已尝试回报 GPTPro，但当前外部桥接没有有效回复。",
            "supporting_numbers": "Desktop returned empty stopped-thinking outputs; Chrome profile required login.",
            "evidence_files": "gptpro_reviews/20260622_v226_result_gptpro_response_blocked.md",
            "allowed_wording_cn": "可以说明 v227 是在无新 GPTPro 指令下的 reporting-only 安全整理。",
            "limitation_cn": "不能把 v227 当成 GPTPro 新批准的实验方向。",
            "status": "process_note",
        },
    ]
    return pd.DataFrame(rows)


def make_limitation_table(main_result: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for _, row in main_result.iterrows():
        rows.extend(
            [
                {
                    "limitation_id": f"{row['pool_key']}_tail_concentration",
                    "pool_key": row["pool_key"],
                    "formal_model": row["formal_model"],
                    "evidence": f"top20pct tail share={fmt_num(row['tail_top20pct_share'])}, tail gini={fmt_num(row['tail_gini_proxy'])}",
                    "impact_cn": "尾部误差不是均匀分布，论文中需要呈现集中度和代表性失败样本。",
                    "allowed_reporting_cn": "作为 limitation 和 future work 说明。",
                    "not_allowed_cn": "不得以此为理由重启 test-tuned gate/router。",
                },
                {
                    "limitation_id": f"{row['pool_key']}_underestimation",
                    "pool_key": row["pool_key"],
                    "formal_model": row["formal_model"],
                    "evidence": f"under_rate={fmt_num(row['under_rate'])}",
                    "impact_cn": "仍存在一定比例低估，尤其需要结合 underestimation profile 表解释。",
                    "allowed_reporting_cn": "作为失败模式/误差模式描述。",
                    "not_allowed_cn": "不得把 true under label 当作推理特征。",
                },
                {
                    "limitation_id": f"{row['pool_key']}_sample_size",
                    "pool_key": row["pool_key"],
                    "formal_model": row["formal_model"],
                    "evidence": f"test n={int(row['n'])}",
                    "impact_cn": "test 样本量有限，subject-block CI 应优先作为跨被试稳健性提示。",
                    "allowed_reporting_cn": "同时报告点估计、sample CI 和 subject-block CI。",
                    "not_allowed_cn": "不得删除 test 样本来改善 CI。",
                },
            ]
        )
    rows.append(
        {
            "limitation_id": "gptpro_bridge_pending",
            "pool_key": "process",
            "formal_model": "none",
            "evidence": "new GPTPro instruction not obtained after v226 report attempt",
            "impact_cn": "v227 只能作为本地 reporting-only fallback，等待 GPTPro 恢复后再回报。",
            "allowed_reporting_cn": "透明记录外部复核暂时阻塞。",
            "not_allowed_cn": "不得声称 v227 是 GPTPro 批准的新实验。",
        }
    )
    return pd.DataFrame(rows)


def flatten_guard_checks(source: str, payload: object) -> List[Dict[str, object]]:
    """把 v225/v226 的 JSON guard 输出摊平成 CSV。"""
    rows: List[Dict[str, object]] = []
    if isinstance(payload, dict) and "checks" in payload and isinstance(payload["checks"], list):
        for item in payload["checks"]:
            rows.append(
                {
                    "source": source,
                    "check": item.get("check", "unknown"),
                    "pass": bool(item.get("pass", False)),
                    "detail": item.get("detail", ""),
                }
            )
    elif isinstance(payload, dict):
        rows.append(
            {
                "source": source,
                "check": "top_level_pass",
                "pass": bool(payload.get("pass", False)),
                "detail": json.dumps(payload, ensure_ascii=False)[:1000],
            }
        )
    return rows


def make_guardrail_summary() -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for source_name in [
        "v225_metric_reproduction",
        "v225_leakage_guard",
        "v225_forbidden_scan",
        "v225_table_alignment",
        "v226_metric_reproduction",
        "v226_leakage_guard",
        "v226_forbidden_scan",
        "v226_table_alignment",
        "v226_file_inventory",
    ]:
        payload = read_json(SOURCE_FILES[source_name])
        rows.extend(flatten_guard_checks(source_name, payload))
    rows.extend(
        [
            {
                "source": "v227_local_guard",
                "check": "no_model_training",
                "pass": True,
                "detail": "v227 reads v225/v226 tables and logs only; no fit/train/predict branch.",
            },
            {
                "source": "v227_local_guard",
                "check": "no_formal_headline_change",
                "pass": True,
                "detail": "formal_model_lock remains loose=avg_joint_focus, strict=peak_floor_090.",
            },
            {
                "source": "v227_local_guard",
                "check": "reporting_only_fallback",
                "pass": True,
                "detail": "new GPTPro instruction is blocked, so v227 only packages writing evidence.",
            },
        ]
    )
    return pd.DataFrame(rows)


def make_artifact_manifest() -> pd.DataFrame:
    rows = []
    for artifact_id, path in SOURCE_FILES.items():
        rows.append(
            {
                "artifact_id": artifact_id,
                "source_version": artifact_id.split("_")[0],
                "path": rel(path),
                "role": infer_artifact_role(artifact_id),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "included_in_v227_zip": False,
            }
        )
    return pd.DataFrame(rows)


def infer_artifact_role(artifact_id: str) -> str:
    if "gptpro_blocked" in artifact_id:
        return "bridge_status"
    if "zip" in artifact_id:
        return "source_package"
    if "lock" in artifact_id:
        return "formal_lock"
    if "ci" in artifact_id:
        return "confidence_interval"
    if "guard" in artifact_id or "scan" in artifact_id or "alignment" in artifact_id:
        return "guardrail"
    if "report" in artifact_id:
        return "source_report"
    return "source_table"


def select_existing_figures() -> List[Tuple[Path, str, str]]:
    """选择少量已经生成的图，复制到 v227 包中。"""
    selected: List[Tuple[Path, str, str]] = []

    # v226 稳健性图全部体积较小，直接复制，保证 writing pack 可独立查看。
    for subdir in [
        "ci_forest_by_pool",
        "subject_level_metric_distribution",
        "tail_error_concentration",
        "underestimation_profile",
        "extreme_peak_cases_summary",
    ]:
        for path in sorted((V226_DIR / "figures" / subdir).glob("*.png")):
            selected.append((path, f"v226_robustness/{subdir}/{path.name}", f"v226 {subdir}"))

    # v225 示例图只取每类每池第一张，避免 v227 包变成重复大包。
    for case_group in ["formal_examples", "worst_tail_cases", "strong_under_cases"]:
        for pool_key in FORMAL_MODEL_LOCK:
            paths = sorted((V225_DIR / "figures" / case_group / pool_key).glob("*.png"))
            if paths:
                selected.append(
                    (
                        paths[0],
                        f"v225_examples/{case_group}/{pool_key}/{paths[0].name}",
                        f"v225 representative {case_group} / {pool_key}",
                    )
                )
    return selected


def copy_figures() -> pd.DataFrame:
    rows = []
    for idx, (src, rel_dest, role) in enumerate(select_existing_figures(), start=1):
        dest = FIGURE_DIR / rel_dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        rows.append(
            {
                "figure_id": f"F{idx:03d}",
                "source_path": rel(src),
                "packaged_path": rel(dest),
                "role_cn": role,
                "caption_cn": build_caption(role, src.name),
                "bytes": dest.stat().st_size,
                "sha256": sha256_file(dest),
            }
        )
    return pd.DataFrame(rows)


def build_caption(role: str, filename: str) -> str:
    if "ci_forest" in role:
        return f"{filename}: formal test metric bootstrap confidence interval."
    if "subject_level" in role:
        return f"{filename}: subject-level distribution for locked formal model."
    if "tail_error" in role:
        return f"{filename}: test tail-error concentration of locked formal model."
    if "underestimation" in role:
        return f"{filename}: route-event underestimation profile for writing limitation."
    if "extreme_peak" in role:
        return f"{filename}: extreme-peak sample summary for formal limitation."
    return f"{filename}: representative v225 formal reconstruction case."


def make_gptpro_bridge_status_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "status_item": "desktop_original_prompt",
                "status": "rejected_as_valid_reply",
                "detail": "Chinese parts displayed as mojibake; not treated as valid GPTPro instruction.",
            },
            {
                "status_item": "desktop_ascii_prompt",
                "status": "no_valid_reply",
                "detail": "Desktop entered Pro thinking and then stopped with no visible answer body.",
            },
            {
                "status_item": "desktop_short_followup",
                "status": "no_valid_reply",
                "detail": "Short ASCII six-item follow-up also stopped with no answer body.",
            },
            {
                "status_item": "chrome_bridge",
                "status": "blocked_by_login",
                "detail": "Chrome snapshot showed ChatGPT login/signup page; user action required.",
            },
            {
                "status_item": "local_fallback",
                "status": "reporting_only",
                "detail": "v227 is limited to paper/claim readiness packaging from existing v225+v226 evidence.",
            },
        ]
    )


def write_report(
    main_result: pd.DataFrame,
    claim_matrix: pd.DataFrame,
    limitations: pd.DataFrame,
    figure_index: pd.DataFrame,
) -> None:
    loose = main_result[main_result["pool_key"] == "loose_main_pool"].iloc[0]
    strict = main_result[main_result["pool_key"] == "strict_main_pool"].iloc[0]
    report = f"""# v227 写作 / claim readiness 整理报告

## 结论

v227 不是新实验，也不是 GPTPro 新指令解锁的模型路线。由于 v226 结果回报 GPTPro 时桌面端连续空停、Chrome 端需要登录，当前没有新的 GPTPro 正文指令。为了不让本地工作卡死，本轮只做 reporting-only 的写作材料整理：把 v225 锁定 formal 结果和 v226 稳健性 / CI 审计整理成可写入论文或组会材料的 claim/readiness 包。

本轮没有训练模型、没有搜索阈值、没有创建 gate/router、没有运行 v222b/v223、没有改变 formal headline。

## formal 主结果仍然锁定

| pool | formal model | test n | RMSE | tail RMSE | sample RMSE CI | subject-block RMSE CI |
|---|---|---:|---:|---:|---|---|
| loose_main_pool | avg_joint_focus | {int(loose['n'])} | {fmt_num(loose['rmse'])} | {fmt_num(loose['tail_rmse'])} | {fmt_num(loose['rmse_sample_ci_lower'])}-{fmt_num(loose['rmse_sample_ci_upper'])} | {fmt_num(loose['rmse_subject_block_ci_lower'])}-{fmt_num(loose['rmse_subject_block_ci_upper'])} |
| strict_main_pool | peak_floor_090 | {int(strict['n'])} | {fmt_num(strict['rmse'])} | {fmt_num(strict['tail_rmse'])} | {fmt_num(strict['rmse_sample_ci_lower'])}-{fmt_num(strict['rmse_sample_ci_upper'])} | {fmt_num(strict['rmse_subject_block_ci_lower'])}-{fmt_num(strict['rmse_subject_block_ci_upper'])} |

## 可以写入论文的表述边界

- 可以写：v225/v226 共同支持 locked formal result，且指标复现、泄漏检查、forbidden scan、table alignment 和 ZIP 完整性均通过。
- 可以写：v226 给出了 sample bootstrap 与 subject-block bootstrap 的不确定性区间。
- 可以写：tail error 仍集中在少量困难样本上，这是 limitation，不是继续本地模型搜索的解锁条件。
- 不可以写：v227 发现了新模型提升。
- 不可以写：v227 或 GPTPro 解锁了 v222b/v223、new tau、gate/router 或 test retuning。
- 不可以写：诊断-only 行可以进入 formal leaderboard。

## 主要 limitation

- loose test top-20% tail-SSE share = {fmt_num(loose['tail_top20pct_share'])}，strict = {fmt_num(strict['tail_top20pct_share'])}，说明尾部误差仍集中。
- loose under_rate = {fmt_num(loose['under_rate'])}，strict under_rate = {fmt_num(strict['under_rate'])}，仍需在论文里解释低估模式。
- 当前 GPTPro 回报通道暂时没有有效回复，因此 v227 只能作为本地写作整理包，后续仍需在 GPTPro 可用时回报。

## 输出文件

- `tables/paper_main_result_table.csv`
- `tables/paper_claim_support_matrix.csv`
- `tables/paper_limitation_table.csv`
- `tables/formal_guardrail_summary.csv`
- `tables/formal_artifact_manifest.csv`
- `tables/figure_selection_index.csv`
- `tables/gptpro_bridge_status.csv`
- `reports/v227_paper_claim_readiness_cn.md`
- `logs/run_manifest.json`
- `logs/input_file_hashes.json`
- `logs/source_artifact_checks.json`
- `logs/no_model_change_guard.json`
- `logs/file_inventory.json`
- `logs/zip_integrity_check.json`
- `v227_paper_claim_readiness_pack.zip`

## 表和图

- claim support rows: {len(claim_matrix)}
- limitation rows: {len(limitations)}
- copied figure rows: {len(figure_index)}

## 下一步

当 GPTPro 通道恢复时，应把 v226+v227 的执行结果一起回报 GPTPro，请它只给 bounded writing/claim/reporting 下一步，继续禁止模型训练、new tau、gate/router、v222b/v223 和 test-based retuning。
"""
    (REPORT_DIR / "v227_paper_claim_readiness_cn.md").write_text(report, encoding="utf-8")


def write_next_gptpro_prompt(main_result: pd.DataFrame) -> None:
    loose = main_result[main_result["pool_key"] == "loose_main_pool"].iloc[0]
    strict = main_result[main_result["pool_key"] == "strict_main_pool"].iloc[0]
    prompt = f"""# GPTPro review request: v226 completed, v227 reporting-only fallback completed

Please review the local status and provide one bounded next instruction.

Important: the previous attempt to report v226 to GPTPro failed because Desktop
produced empty stopped-thinking outputs and Chrome required login. Codex did not
start a new model task. It only created a reporting-only v227 paper/claim
readiness package from existing v225+v226 outputs.

Local facts:

- v225 formal headline remains locked:
  - loose_main_pool: avg_joint_focus
  - strict_main_pool: peak_floor_090
- v226 formal robustness / CI audit completed and passed all checks.
- v227 paper/claim readiness package completed as reporting-only fallback.
- No model was trained.
- No threshold/tau was searched.
- No gate/router was created.
- No v222b/v223 was run.
- No formal headline changed.

Key test results:

- loose avg_joint_focus:
  - n={int(loose['n'])}
  - RMSE={fmt_num(loose['rmse'])}
  - tail RMSE={fmt_num(loose['tail_rmse'])}
  - sample RMSE CI={fmt_num(loose['rmse_sample_ci_lower'])}-{fmt_num(loose['rmse_sample_ci_upper'])}
  - subject-block RMSE CI={fmt_num(loose['rmse_subject_block_ci_lower'])}-{fmt_num(loose['rmse_subject_block_ci_upper'])}
  - top20pct tail share={fmt_num(loose['tail_top20pct_share'])}
- strict peak_floor_090:
  - n={int(strict['n'])}
  - RMSE={fmt_num(strict['rmse'])}
  - tail RMSE={fmt_num(strict['tail_rmse'])}
  - sample RMSE CI={fmt_num(strict['rmse_sample_ci_lower'])}-{fmt_num(strict['rmse_sample_ci_upper'])}
  - subject-block RMSE CI={fmt_num(strict['rmse_subject_block_ci_lower'])}-{fmt_num(strict['rmse_subject_block_ci_upper'])}
  - top20pct tail share={fmt_num(strict['tail_top20pct_share'])}

Please answer with:

1. Accept/reject v227 as a valid reporting-only fallback and exact reason.
2. The next local task/version.
3. Allowed input files.
4. Required output directory and required files.
5. Exact stop condition.
6. Validation checks before reporting back.

Do not request model training, v222b/v223, new tau, new gate/router, or
test-based retuning unless you explicitly overturn the current stop condition
and provide leakage/test-discipline guardrails.
"""
    (REPORT_DIR / "v227_next_gptpro_prompt_ascii.md").write_text(prompt, encoding="utf-8")


def build_source_checks(v225_lock: pd.DataFrame, v226_lock: pd.DataFrame, readiness: pd.DataFrame) -> Dict[str, object]:
    v225_zip_bad = test_zip(SOURCE_FILES["v225_zip"])
    v226_zip_bad = test_zip(SOURCE_FILES["v226_zip"])
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_files_missing": [],
        "v225_zip_bad_file": v225_zip_bad,
        "v226_zip_bad_file": v226_zip_bad,
        "formal_lock_exact": dict(zip(v225_lock["pool"], v225_lock["formal_model"])) == FORMAL_MODEL_LOCK
        and dict(zip(v226_lock["pool_key"], v226_lock["formal_model"])) == FORMAL_MODEL_LOCK,
        "v226_readiness_all_accepted": bool(readiness["accepted_for_paper_main_result"].astype(bool).all()),
        "v226_needs_new_model_any": bool(readiness["needs_new_model"].astype(bool).any()),
        "v226_needs_gate_or_router_any": bool(readiness["needs_gate_or_router"].astype(bool).any()),
        "gptpro_blocked_records_exist": all(
            SOURCE_FILES[name].exists()
            for name in ["gptpro_blocked_response", "gptpro_blocked_decision", "gptpro_blocked_action_items"]
        ),
        "pass": v225_zip_bad is None and v226_zip_bad is None,
    }


def build_no_model_guard() -> Dict[str, object]:
    checks = [
        ("no_training_executed", True, "v227 uses only pandas table reads and report/zip writing."),
        ("no_new_prediction_generated", True, "No prediction arrays are created or changed."),
        ("no_threshold_or_tau_search", True, "No search grid or threshold selection exists."),
        ("no_gate_or_router_created", True, "No deployable selector/gate/router output is produced."),
        ("no_v222b_or_v223", True, "No v222b/v223 path is read or written."),
        ("formal_headline_unchanged", True, "Only avg_joint_focus and peak_floor_090 appear in formal main result."),
        ("gptpro_fallback_reporting_only", True, "Bridge failure only authorizes writing/package fallback."),
    ]
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "pass": all(item[1] for item in checks),
        "checks": [{"check": name, "pass": passed, "detail": detail} for name, passed, detail in checks],
    }


def list_output_files() -> List[Dict[str, object]]:
    files = []
    for path in sorted(OUT_DIR.rglob("*")):
        if path.is_file() and path.name != "v227_paper_claim_readiness_pack.zip":
            files.append(
                {
                    "path": str(path.relative_to(OUT_DIR)).replace("\\", "/"),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return files


def write_file_inventory(required_files: Iterable[str], zip_bad_file: str | None) -> Dict[str, object]:
    required_files = list(required_files)
    existing = {item["path"] for item in list_output_files()}
    inventory = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "file_count_excluding_zip": len(existing),
        "required_files_missing": sorted(set(required_files) - existing),
        "zip_bad_file": zip_bad_file,
        "files": list_output_files(),
    }
    write_json(LOG_DIR / "file_inventory.json", inventory)
    return inventory


def create_zip() -> Path:
    zip_path = OUT_DIR / "v227_paper_claim_readiness_pack.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(OUT_DIR.rglob("*")):
            if path.is_file() and path != zip_path:
                zf.write(path, path.relative_to(OUT_DIR))
    return zip_path


def main() -> None:
    ensure_clean_output()
    assert_sources_exist()

    v225_lock = pd.read_csv(SOURCE_FILES["v225_formal_model_lock"])
    v226_lock = pd.read_csv(SOURCE_FILES["v226_model_lock_recheck"])
    assert_formal_lock(v225_lock, v226_lock)

    pool_metrics = pd.read_csv(SOURCE_FILES["v225_pool_metrics"])
    sample_ci = pd.read_csv(SOURCE_FILES["v226_sample_ci"])
    subject_ci = pd.read_csv(SOURCE_FILES["v226_subject_ci"])
    tail_concentration = pd.read_csv(SOURCE_FILES["v226_tail_concentration"])
    readiness = pd.read_csv(SOURCE_FILES["v226_readiness"])
    assert_v226_readiness(readiness)

    main_result = make_main_result_table(pool_metrics, sample_ci, subject_ci, tail_concentration, readiness)
    claim_matrix = make_claim_support_matrix(main_result)
    limitations = make_limitation_table(main_result)
    guardrail_summary = make_guardrail_summary()
    artifact_manifest = make_artifact_manifest()
    bridge_status = make_gptpro_bridge_status_table()
    figure_index = copy_figures()

    main_result.to_csv(TABLE_DIR / "paper_main_result_table.csv", index=False, encoding="utf-8-sig")
    claim_matrix.to_csv(TABLE_DIR / "paper_claim_support_matrix.csv", index=False, encoding="utf-8-sig")
    limitations.to_csv(TABLE_DIR / "paper_limitation_table.csv", index=False, encoding="utf-8-sig")
    guardrail_summary.to_csv(TABLE_DIR / "formal_guardrail_summary.csv", index=False, encoding="utf-8-sig")
    artifact_manifest.to_csv(TABLE_DIR / "formal_artifact_manifest.csv", index=False, encoding="utf-8-sig")
    figure_index.to_csv(TABLE_DIR / "figure_selection_index.csv", index=False, encoding="utf-8-sig")
    bridge_status.to_csv(TABLE_DIR / "gptpro_bridge_status.csv", index=False, encoding="utf-8-sig")

    write_report(main_result, claim_matrix, limitations, figure_index)
    write_next_gptpro_prompt(main_result)

    write_json(
        LOG_DIR / "run_manifest.json",
        {
            "version": "v227_paper_claim_readiness_pack_20260622",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "scope": "reporting_only_paper_claim_readiness",
            "source_versions": ["v225", "v226"],
            "gptpro_new_instruction_obtained": False,
            "fallback_reason": "GPTPro report channel blocked after v226 result handoff attempts.",
            "formal_model_lock": FORMAL_MODEL_LOCK,
            "no_model_training": True,
            "no_threshold_search": True,
            "no_gate_or_router": True,
            "no_v222b_or_v223": True,
            "output_dir": rel(OUT_DIR),
        },
    )
    write_json(
        LOG_DIR / "input_file_hashes.json",
        {
            name: {"path": rel(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for name, path in SOURCE_FILES.items()
        },
    )
    write_json(LOG_DIR / "source_artifact_checks.json", build_source_checks(v225_lock, v226_lock, readiness))
    write_json(LOG_DIR / "no_model_change_guard.json", build_no_model_guard())

    required_files = [
        "tables/paper_main_result_table.csv",
        "tables/paper_claim_support_matrix.csv",
        "tables/paper_limitation_table.csv",
        "tables/formal_guardrail_summary.csv",
        "tables/formal_artifact_manifest.csv",
        "tables/figure_selection_index.csv",
        "tables/gptpro_bridge_status.csv",
        "reports/v227_paper_claim_readiness_cn.md",
        "reports/v227_next_gptpro_prompt_ascii.md",
        "logs/run_manifest.json",
        "logs/input_file_hashes.json",
        "logs/source_artifact_checks.json",
        "logs/no_model_change_guard.json",
        "logs/file_inventory.json",
        "logs/zip_integrity_check.json",
    ]

    # 先写入稳定的 zip_integrity_check，再生成 inventory 和 ZIP。
    write_json(
        LOG_DIR / "zip_integrity_check.json",
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "zip_path": rel(OUT_DIR / "v227_paper_claim_readiness_pack.zip"),
            "zip_bad_file": None,
            "pass": True,
        },
    )
    write_file_inventory(required_files, zip_bad_file=None)
    zip_path = create_zip()
    bad_file = test_zip(zip_path)
    if bad_file is not None:
        raise AssertionError(f"ZIP integrity failed at {bad_file}")

    # 用最终 ZIP 检查结果更新日志，再重新打包一次，确保 ZIP 内部日志也是最终状态。
    write_json(
        LOG_DIR / "zip_integrity_check.json",
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "zip_path": rel(zip_path),
            "zip_bad_file": bad_file,
            "pass": bad_file is None,
        },
    )
    inventory = write_file_inventory(required_files, zip_bad_file=bad_file)
    if inventory["required_files_missing"]:
        raise AssertionError(f"Required files missing: {inventory['required_files_missing']}")
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
                "required_files_missing": inventory["required_files_missing"],
                "zip_bad_file": final_bad_file,
                "main_result_rows": len(main_result),
                "claim_rows": len(claim_matrix),
                "limitation_rows": len(limitations),
                "copied_figures": len(figure_index),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
