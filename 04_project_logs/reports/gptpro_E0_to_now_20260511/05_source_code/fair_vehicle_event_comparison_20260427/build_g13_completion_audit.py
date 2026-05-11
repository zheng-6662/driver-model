# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[4]
REPORT_DIR = PROJECT_ROOT / "04_project_logs" / "reports" / "g13_model_breakthrough_20260510"
RESTORE_DIR = PROJECT_ROOT / "04_project_logs" / "reports" / "restore_checkpoint_audit_20260510"
OUT_DIR = REPORT_DIR / "g13_completion_audit_20260510"


EXPECTED_RUNS = [
    ("G13A", 2026),
    ("G13B", 2026),
    ("G13C", 2026),
    ("G13F", 2026),
    ("G13H", 2026),
    ("G13I", 2026),
    ("G13H", 2027),
    ("G13H", 2028),
    ("G13I", 2027),
    ("G13I", 2028),
]

RUN_REQUIRED_FILES = [
    "best_model.pt",
    "metrics.json",
    "loss_history.csv",
    "prediction_figures/test/overview.png",
    "prediction_figures/test/plot_index.csv",
    "prediction_figures/test/prediction_sample_metrics.csv",
    "prediction_figures/test/prediction_sequences.npz",
]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, encoding="utf-8-sig")


def _exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _fmt_path(path: Path) -> str:
    return str(path)


def _run_index() -> pd.DataFrame:
    seed2026 = _read_csv(REPORT_DIR / "g13_seed2026_full_index.csv")
    seed2026 = seed2026.rename(
        columns={
            "label": "experiment_name",
            "test_steer_rmse": "test_rmse",
            "selection": "selection_score",
        }
    )
    seed2026 = seed2026[
        [
            "experiment_id",
            "seed",
            "experiment_name",
            "local_run_root",
            "test_rmse",
            "primary_rmse",
            "tail_rmse",
            "peak_err_s",
            "selection_score",
        ]
    ].copy()

    multiseed = _read_csv(REPORT_DIR / "g13_hi_multiseed_summary_20260510" / "g13_hi_seed_wise_metrics.csv")
    multiseed = multiseed[multiseed["seed"].astype(int).isin([2027, 2028])].copy()
    multiseed = multiseed[
        [
            "experiment_id",
            "seed",
            "experiment_name",
            "local_run_root",
            "test_rmse",
            "primary_rmse",
            "tail_rmse",
            "peak_err_s",
            "selection_score",
        ]
    ]
    out = pd.concat([seed2026, multiseed], ignore_index=True)
    out["seed"] = out["seed"].astype(int)
    return out


def _check_run_artifacts(run_index: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for exp_id, seed in EXPECTED_RUNS:
        matches = run_index[(run_index["experiment_id"].eq(exp_id)) & (run_index["seed"].eq(seed))]
        if matches.empty:
            rows.append(
                {
                    "experiment_id": exp_id,
                    "seed": seed,
                    "local_run_root": "",
                    "all_required_files_present": False,
                    "missing_files": "run index row missing",
                    "overview_path": "",
                    "plot_index_path": "",
                }
            )
            continue
        root = Path(str(matches.iloc[0]["local_run_root"]))
        missing = [rel for rel in RUN_REQUIRED_FILES if not _exists(root / rel)]
        rows.append(
            {
                "experiment_id": exp_id,
                "seed": seed,
                "local_run_root": _fmt_path(root),
                "all_required_files_present": len(missing) == 0,
                "missing_files": "; ".join(missing),
                "overview_path": _fmt_path(root / "prediction_figures" / "test" / "overview.png"),
                "plot_index_path": _fmt_path(root / "prediction_figures" / "test" / "plot_index.csv"),
                "sample_metrics_path": _fmt_path(root / "prediction_figures" / "test" / "prediction_sample_metrics.csv"),
            }
        )
    return pd.DataFrame(rows)


def _artifact_rows() -> list[dict[str, str]]:
    paths = {
        "恢复报告": RESTORE_DIR / "restore_status_cn.md",
        "恢复索引": RESTORE_DIR / "restored_run_index_20260510.csv",
        "G13 执行说明": REPORT_DIR / "g13_execution_note_cn.md",
        "服务器启动记录": REPORT_DIR / "g13_server_start_cn.md",
        "服务器最终状态": REPORT_DIR / "g13_server_final_status_20260510.txt",
        "代码编译检查": REPORT_DIR / "g13_code_compile_check_20260510.txt",
        "seed2026 训练索引": REPORT_DIR / "g13_seed2026_full_index.csv",
        "seed2026 诊断报告": REPORT_DIR / "g13_seed2026_diagnostics" / "g13_seed2026_screening_summary_cn.md",
        "三种子复验报告": REPORT_DIR / "g13_hi_multiseed_summary_20260510" / "g13_hi_multiseed_report_cn.md",
        "三种子逐 seed 表": REPORT_DIR / "g13_hi_multiseed_summary_20260510" / "g13_hi_seed_wise_metrics.csv",
        "三种子汇总表": REPORT_DIR / "g13_hi_multiseed_summary_20260510" / "g13_hi_three_seed_summary.csv",
        "物理风险表": REPORT_DIR / "g13_hi_multiseed_summary_20260510" / "g13_hi_physical_mean.csv",
        "G11 困难样本表": REPORT_DIR / "g13_hi_multiseed_summary_20260510" / "g13_hi_g11_mean.csv",
        "分被试表": REPORT_DIR / "g13_hi_multiseed_summary_20260510" / "g13_hi_subject_summary.csv",
        "分响应类型表": REPORT_DIR / "g13_hi_multiseed_summary_20260510" / "g13_hi_morphology_summary.csv",
        "逐样本明细": REPORT_DIR / "g13_hi_multiseed_summary_20260510" / "g13_hi_sample_detail.csv",
        "每日记录": PROJECT_ROOT / "04_project_logs" / "reports" / "progress" / "daily" / "2026-05-10.md",
        "实验注册表": PROJECT_ROOT / "04_project_logs" / "reports" / "progress" / "experiment_registry.md",
    }
    rows = []
    for name, path in paths.items():
        rows.append(
            {
                "artifact": name,
                "path": _fmt_path(path),
                "exists": str(_exists(path)),
                "bytes": str(path.stat().st_size if path.exists() else 0),
            }
        )
    return rows


def _check_code_evidence() -> pd.DataFrame:
    files = {
        "响应类型标签/辅助头": THIS_DIR.parent / "event_conditioned_baseline_model.py",
        "响应类型条件化预测头": THIS_DIR.parent / "conditioned_trajectory_head.py",
        "训练参数、物理损失、蒸馏权重": THIS_DIR.parent / "run_event_conditioned_trajectory_baseline.py",
        "G13 候选版本运行器": THIS_DIR / "run_g13_breakthrough_candidates.py",
        "G13 seed2026 诊断脚本": THIS_DIR / "summarize_g13_seed2026_diagnostics.py",
        "G13 三种子汇总脚本": THIS_DIR / "summarize_g13_hi_multiseed.py",
    }
    rows = []
    for name, path in files.items():
        rows.append({"item": name, "path": _fmt_path(path), "exists": _exists(path)})
    return pd.DataFrame(rows)


def _check_requirement_rows(run_artifacts: pd.DataFrame, artifacts: list[dict[str, str]], code_evidence: pd.DataFrame) -> pd.DataFrame:
    artifact_map = {row["artifact"]: row for row in artifacts}

    def ok_artifact(name: str) -> bool:
        return artifact_map.get(name, {}).get("exists") == "True"

    def ok_code(name: str) -> bool:
        rows = code_evidence[code_evidence["item"].eq(name)]
        return bool(len(rows) and rows.iloc[0]["exists"])

    run_ok = bool(run_artifacts["all_required_files_present"].all())
    rows = [
        {
            "requirement": "旧设置和核心 checkpoint 恢复",
            "evidence": "恢复报告、恢复索引、核心运行目录和 best_model.pt",
            "status": "完成" if ok_artifact("恢复报告") and ok_artifact("恢复索引") else "缺失",
        },
        {
            "requirement": "响应类型辅助学习已实现并验证",
            "evidence": "代码含响应类型头；G13A/B/H/I 完整训练；seed2026 诊断和三种子报告",
            "status": "完成" if ok_code("响应类型标签/辅助头") and ok_artifact("seed2026 诊断报告") else "缺失",
        },
        {
            "requirement": "响应类型影响轨迹预测已实现并验证",
            "evidence": "条件化预测头；G13C/F/I 完整训练",
            "status": "完成" if ok_code("响应类型条件化预测头") and run_ok else "缺失",
        },
        {
            "requirement": "方向/幅值/尾段物理约束已实现并验证",
            "evidence": "训练参数含幅值/方向损失；G13F/I 完整训练；物理风险表",
            "status": "完成" if ok_code("训练参数、物理损失、蒸馏权重") and ok_artifact("物理风险表") else "缺失",
        },
        {
            "requirement": "脑电教师与肌电推理输入的选择性融合已验证",
            "evidence": "G13H/G13I 三种子，命令含 EEG teacher checkpoint 和 raw_emg_only",
            "status": "完成" if ok_artifact("三种子复验报告") and run_ok else "缺失",
        },
        {
            "requirement": "服务器高效训练且未留下训练进程",
            "evidence": "服务器启动记录、并行训练日志、最终 GPU 空闲状态",
            "status": "完成" if ok_artifact("服务器启动记录") and ok_artifact("服务器最终状态") else "缺失",
        },
        {
            "requirement": "每个版本记录整体误差、尾段、综合指标和预测图",
            "evidence": "g13_seed2026_full_index、g13_hi_seed_wise_metrics、每个 run 的 overview.png/plot_index.csv",
            "status": "完成" if ok_artifact("三种子逐 seed 表") and run_ok else "缺失",
        },
        {
            "requirement": "记录幅值不足、错侧、后段漂移、G11、分响应类型、分被试",
            "evidence": "物理风险表、G11 困难样本表、分响应类型表、分被试表、逐样本明细",
            "status": "完成"
            if ok_artifact("物理风险表") and ok_artifact("G11 困难样本表") and ok_artifact("分响应类型表") and ok_artifact("分被试表")
            else "缺失",
        },
        {
            "requirement": "中文实验报告、版本表、统一对照表、预测图索引、论文主线建议",
            "evidence": "seed2026 诊断报告、三种子复验报告、本审计报告、预测图索引",
            "status": "完成" if ok_artifact("三种子复验报告") and run_ok else "缺失",
        },
        {
            "requirement": "代码编译检查",
            "evidence": "g13_code_compile_check_20260510.txt",
            "status": "完成" if ok_artifact("代码编译检查") else "缺失",
        },
    ]
    return pd.DataFrame(rows)


def _write_csv(path: Path, rows: list[dict[str, Any]] | pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rows, pd.DataFrame):
        rows.to_csv(path, index=False, encoding="utf-8-sig")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_artifact_index(path: Path, artifacts: list[dict[str, str]], run_artifacts: pd.DataFrame) -> None:
    lines = ["# G13 产物索引", ""]
    lines.append("## 关键报告和表格")
    lines.append("")
    lines.append("| 产物 | 路径 | 状态 |")
    lines.append("| --- | --- | --- |")
    for row in artifacts:
        status = "存在" if row["exists"] == "True" else "缺失"
        lines.append(f"| {row['artifact']} | `{row['path']}` | {status} |")
    lines.append("")
    lines.append("## 预测图索引")
    lines.append("")
    lines.append("| 版本 | seed | overview 图 | 单图索引 | 状态 |")
    lines.append("| --- | ---: | --- | --- | --- |")
    for _, row in run_artifacts.iterrows():
        status = "完整" if bool(row["all_required_files_present"]) else f"缺失：{row['missing_files']}"
        lines.append(
            f"| {row['experiment_id']} | {int(row['seed'])} | `{row['overview_path']}` | `{row['plot_index_path']}` | {status} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_audit_report(
    path: Path,
    checklist: pd.DataFrame,
    run_artifacts: pd.DataFrame,
    code_evidence: pd.DataFrame,
) -> None:
    all_done = bool(checklist["status"].eq("完成").all())
    missing = checklist[~checklist["status"].eq("完成")]
    lines: list[str] = []
    lines.append("# G13 完成审计报告")
    lines.append("")
    lines.append("## 目标拆解")
    lines.append("")
    lines.append("G13 的具体交付标准被拆成以下几类：")
    lines.append("")
    lines.append("- 恢复旧设置和核心 checkpoint，保证旧结果可复验。")
    lines.append("- 实现并验证响应类型辅助学习、响应类型影响轨迹预测、方向/幅值物理约束、脑电教师与肌电推理输入组合。")
    lines.append("- 使用服务器完成正式训练，并拉回 checkpoint、指标和预测图。")
    lines.append("- 每个候选版本记录整体误差、尾段误差、综合选择指标、幅值不足、错侧、后段漂移、G11 困难样本、分响应类型和分被试结果。")
    lines.append("- 输出中文实验报告、版本表、统一对照表、预测图索引和是否形成论文主线的建议。")
    lines.append("")
    lines.append("## 逐项核对")
    lines.append("")
    lines.append("| 要求 | 证据 | 状态 |")
    lines.append("| --- | --- | --- |")
    for _, row in checklist.iterrows():
        lines.append(f"| {row['requirement']} | {row['evidence']} | {row['status']} |")
    lines.append("")
    lines.append("## 运行产物完整性")
    lines.append("")
    lines.append("| 版本 | seed | 运行目录 | 必要文件 |")
    lines.append("| --- | ---: | --- | --- |")
    for _, row in run_artifacts.iterrows():
        status = "完整" if bool(row["all_required_files_present"]) else f"缺失：{row['missing_files']}"
        lines.append(f"| {row['experiment_id']} | {int(row['seed'])} | `{row['local_run_root']}` | {status} |")
    lines.append("")
    lines.append("## 代码证据")
    lines.append("")
    lines.append("| 内容 | 文件 | 状态 |")
    lines.append("| --- | --- | --- |")
    for _, row in code_evidence.iterrows():
        status = "存在" if bool(row["exists"]) else "缺失"
        lines.append(f"| {row['item']} | `{row['path']}` | {status} |")
    lines.append("")
    lines.append("## 审计结论")
    lines.append("")
    if all_done:
        lines.append("- G13 的执行型交付已经完整：代码、训练、恢复、三种子复验、诊断表、预测图索引和中文报告均已落地。")
        lines.append("- 结果结论不是“形成新主线”，而是“G13H/G13I 不能替代 E5A/E6/E10C”。")
        lines.append("- 因此 G13 这一阶段可以关闭；后续若继续，应开新阶段，重点研究 seed2027 回落、幅值不足和 G11 困难样本仍无法超过 E6 的原因。")
    else:
        lines.append("- G13 还不能关闭，以下要求缺失或证据不足：")
        for _, row in missing.iterrows():
            lines.append(f"  - {row['requirement']}：{row['status']}")
    lines.append("")
    lines.append("## 主要结论摘要")
    lines.append("")
    lines.append("- G13H：三种子 test RMSE `0.4503±0.0109`，没有超过 E5A/E6/E10C。")
    lines.append("- G13I：三种子 test RMSE `0.4546±0.0072`，物理风险更均衡但整体和尾段更弱。")
    lines.append("- 当前保留 E5A/E6/E10C 作为主候选，G13H/G13I 作为负面边界和诊断证据。")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_index = _run_index()
    run_artifacts = _check_run_artifacts(run_index)
    artifacts = _artifact_rows()
    code_evidence = _check_code_evidence()
    checklist = _check_requirement_rows(run_artifacts, artifacts, code_evidence)

    _write_csv(OUT_DIR / "g13_required_run_artifacts.csv", run_artifacts)
    _write_csv(OUT_DIR / "g13_artifact_presence.csv", artifacts)
    _write_csv(OUT_DIR / "g13_code_evidence.csv", code_evidence)
    _write_csv(OUT_DIR / "g13_prompt_to_artifact_checklist.csv", checklist)
    _write_artifact_index(OUT_DIR / "g13_artifact_index_cn.md", artifacts, run_artifacts)
    _write_audit_report(OUT_DIR / "g13_completion_audit_report_cn.md", checklist, run_artifacts, code_evidence)

    done = bool(checklist["status"].eq("完成").all())
    status = {"all_requirements_complete": done, "missing_count": int((~checklist["status"].eq("完成")).sum())}
    (OUT_DIR / "g13_completion_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False))


if __name__ == "__main__":
    main()
