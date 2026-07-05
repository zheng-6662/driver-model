#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v229 two-month lessons and failure taxonomy pack.

本脚本只做路线复盘、失败分类和 GPTPro 中文交接稿整理，不训练模型、不生成新预测、
不重新选择 formal headline，也不基于 test split 调参。输入只来自已经完成并验证过的
v220/v225/v228 产物：

1. v220 两个月路线重建包：用于提炼阶段经验和反复出现的路线教训。
2. v225 formal route reconstruction evidence pack：用于读取正式失败样本、bucket 指标、
   route-event 指标和 diagnostic-only selector/oracle gap。
3. v228 final paper artifact freeze：用于锁定最终正式指标、CI、claim 与 limitation 边界。

输出是一个可给 GPTPro 的中文复盘包，目标是让下一轮讨论先围绕“失败类型和路线边界”，
而不是直接进入新的大模型、gate/router、tau/threshold 或 test-based retuning。
"""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
REBUILD_ROOT = REPO_ROOT / "05_rebuild_from_raw_20260511"
BASE_DIR = REBUILD_ROOT / "03_baselines"

V220_DIR = BASE_DIR / "v220_two_month_route_reconstruction_20260622"
V225_DIR = BASE_DIR / "v225_formal_route_reconstruction_evidence_pack_20260622"
V228_DIR = BASE_DIR / "v228_final_paper_artifact_freeze_20260623"

OUT_DIR = BASE_DIR / "v229_two_month_lessons_failure_taxonomy_20260623"
TABLE_DIR = OUT_DIR / "tables"
REPORT_DIR = OUT_DIR / "reports"
LOG_DIR = OUT_DIR / "logs"
ZIP_NAME = "v229_two_month_lessons_failure_taxonomy_pack.zip"

FORMAL_MODEL_LOCK = {
    "loose_main_pool": "avg_joint_focus",
    "strict_main_pool": "peak_floor_090",
}

SOURCE_FILES = {
    "v220_phase_summary": V220_DIR / "tables" / "v220_phase_summary.csv",
    "v220_attempt_timeline": V220_DIR / "tables" / "v220_attempt_timeline.csv",
    "v220_run_summary": V220_DIR / "v220_run_summary.json",
    "v225_bucket_metrics": V225_DIR / "tables" / "formal_reconstruction_metrics_by_bucket.csv",
    "v225_route_event_metrics": V225_DIR / "tables" / "formal_reconstruction_metrics_by_route_event.csv",
    "v225_failure_case_index": V225_DIR / "tables" / "formal_failure_case_index.csv",
    "v225_diagnostic_closeout": V225_DIR / "tables" / "diagnostic_only_v222a_closeout_summary.csv",
    "v225_formal_model_lock": V225_DIR / "tables" / "formal_model_lock.csv",
    "v228_final_main_result": V228_DIR / "tables" / "final_main_result_table.csv",
    "v228_final_ci": V228_DIR / "tables" / "final_ci_table.csv",
    "v228_final_claim_lock": V228_DIR / "tables" / "final_claim_lock_table.csv",
    "v228_final_limitations": V228_DIR / "tables" / "final_limitations_table.csv",
    "v228_guardrail": V228_DIR / "logs" / "guardrail_check.json",
    "v228_consistency": V228_DIR / "logs" / "consistency_check.json",
}

REQUIRED_RELATIVE_FILES = [
    "tables/v229_phase_lessons_table.csv",
    "tables/v229_failure_taxonomy_by_pool_event.csv",
    "tables/v229_top_tail_failure_cases.csv",
    "tables/v229_bucket_risk_summary.csv",
    "tables/v229_selector_candidate_diagnosis.csv",
    "tables/v229_next_action_decision_matrix.csv",
    "reports/v229_two_month_lessons_failure_taxonomy_cn.md",
    "reports/v229_gptpro_next_prompt_cn.md",
    "logs/run_manifest.json",
    "logs/input_file_hashes.json",
    "logs/guardrail_check.json",
    "logs/file_inventory.json",
    ZIP_NAME,
]


def rel(path: Path) -> str:
    """把绝对路径转换成仓库内相对路径，方便报告和日志复核。"""

    return str(path.relative_to(REPO_ROOT)).replace("\\", "/")


def sha256_file(path: Path) -> str:
    """记录输入文件哈希，保证复盘包能追溯到固定源表。"""

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


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def clean_output_dir() -> None:
    """只允许清理本轮固定输出目录，避免误删 03_baselines 下其他版本。"""

    resolved_out = OUT_DIR.resolve()
    resolved_base = BASE_DIR.resolve()
    if resolved_base not in resolved_out.parents:
        raise RuntimeError(f"Refusing to clean outside base dir: {resolved_out}")
    if OUT_DIR.name != "v229_two_month_lessons_failure_taxonomy_20260623":
        raise RuntimeError(f"Unexpected output dir name: {OUT_DIR.name}")
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    for path in [TABLE_DIR, REPORT_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def ensure_sources_exist() -> None:
    missing = [name for name, path in SOURCE_FILES.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required source files: {missing}")


def normalize_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y"}


def make_phase_lessons_table(phase_df: pd.DataFrame, attempt_summary: dict) -> pd.DataFrame:
    """保留 v220 阶段经验，并加上 v229 对后续动作的解释。"""

    rows = []
    for _, row in phase_df.iterrows():
        lesson = str(row.get("沉淀经验", "")).strip()
        phase = str(row.get("阶段", "")).strip()
        if "端到端" in phase:
            implication = "不要再把单一完整曲线回归当作唯一主线；必须单独审计强反应、尾段和反转。"
        elif "生理" in phase or "多候选" in phase:
            implication = "候选和外部信号有诊断价值，但下一步应先问选择是否可部署，而不是直接融合更多输入。"
        elif "原始数据" in phase:
            implication = "模型失败需要先排除锚点、弱响应、道路干扰和样本口径污染。"
        elif "W2/W3" in phase:
            implication = "oracle 好不等于 deployable router 好；同空间 current-window router 调参应保持关闭。"
        elif "Gold-V2" in phase:
            implication = "优先守住事件、锚点、正反例和低维目标定义，避免回到旧数据口径。"
        elif "道路预瞄" in phase:
            implication = "高频/道路/风格可作为诊断输入，但不能绕过强幅值低估问题。"
        elif "滚动预测" in phase:
            implication = "关键点/控制点有上限价值，但输入到曲线的还原链条仍需按失败桶缩小范围。"
        elif "机制优先" in phase:
            implication = "组合框架比继续堆大模型更稳；下一步应围绕失败分类决定是否扩展。"
        else:
            implication = "保留为追溯材料，不作为新实验解锁依据。"

        rows.append(
            {
                "phase": phase,
                "date_range": row.get("时间范围", ""),
                "attempt_count_in_phase": row.get("记录条数", ""),
                "main_work": row.get("主要做法", ""),
                "main_result": row.get("主要结果", ""),
                "lesson": lesson,
                "v229_implication": implication,
            }
        )

    df = pd.DataFrame(rows)
    df["total_attempt_count_from_v220"] = attempt_summary.get("attempt_count")
    df["phase_count_from_v220"] = attempt_summary.get("phase_count")
    return df


def make_failure_taxonomy(failure_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """基于 v225 逐样本表生成 test split 的失败桶统计和高尾误差案例表。"""

    df = failure_df.copy()
    numeric_cols = [
        "rmse",
        "tail_rmse",
        "observed_peak_abs",
        "pred_peak_abs",
        "peak_ratio",
        "tail_p90_within_pool_split",
        "tail_median_within_pool_split",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["under_flag", "strong_steer", "extreme_peak", "high_tail_error"]:
        if col in df.columns:
            df[col] = df[col].map(normalize_bool)

    test_df = df[df["split"].astype(str) == "test"].copy()
    if test_df.empty:
        raise RuntimeError("v225 failure_case_index has no test rows")

    test_df["failure_bucket_v229"] = test_df.apply(classify_failure_row, axis=1)
    test_df["is_top20_tail_v229"] = False
    for pool_key, idx in test_df.groupby("pool_key").groups.items():
        threshold = test_df.loc[idx, "tail_rmse"].quantile(0.8)
        test_df.loc[idx, "is_top20_tail_v229"] = test_df.loc[idx, "tail_rmse"] >= threshold

    grouped = (
        test_df.groupby(["pool_key", "formal_model", "failure_bucket_v229"], dropna=False)
        .agg(
            n=("sample_id", "count"),
            avg_rmse=("rmse", "mean"),
            avg_tail_rmse=("tail_rmse", "mean"),
            max_tail_rmse=("tail_rmse", "max"),
            under_count=("under_flag", "sum"),
            high_tail_count=("high_tail_error", "sum"),
            top20_tail_count=("is_top20_tail_v229", "sum"),
            avg_peak_ratio=("peak_ratio", "mean"),
        )
        .reset_index()
    )
    pool_counts = test_df.groupby("pool_key")["sample_id"].count().rename("pool_test_n")
    grouped = grouped.merge(pool_counts, on="pool_key", how="left")
    grouped["bucket_share_in_pool"] = grouped["n"] / grouped["pool_test_n"]
    grouped["under_rate_in_bucket"] = grouped["under_count"] / grouped["n"]
    grouped["top20_tail_rate_in_bucket"] = grouped["top20_tail_count"] / grouped["n"]
    grouped = grouped.sort_values(["pool_key", "avg_tail_rmse"], ascending=[True, False])

    top_cases = test_df.sort_values(["pool_key", "tail_rmse"], ascending=[True, False]).copy()
    top_cases = top_cases.groupby("pool_key", group_keys=False).head(25)
    top_cases = top_cases[
        [
            "pool_key",
            "formal_model",
            "sample_id",
            "scene_type",
            "route_event",
            "failure_bucket_v229",
            "rmse",
            "tail_rmse",
            "under_flag",
            "strong_steer",
            "extreme_peak",
            "observed_peak_abs",
            "pred_peak_abs",
            "peak_ratio",
            "is_top20_tail_v229",
            "figure_path",
        ]
    ]
    return grouped, top_cases


def classify_failure_row(row: pd.Series) -> str:
    """把正式失败样本归到更适合路线决策的粗桶。"""

    route_event = str(row.get("route_event", "unknown"))
    under = normalize_bool(row.get("under_flag", False))
    strong = normalize_bool(row.get("strong_steer", False))
    extreme = normalize_bool(row.get("extreme_peak", False))
    high_tail = normalize_bool(row.get("high_tail_error", False))

    if route_event == "extreme_peak" or extreme:
        if under:
            return "极端峰值低估"
        return "极端峰值/尾段难例"
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


def make_bucket_risk_summary(bucket_df: pd.DataFrame) -> pd.DataFrame:
    """提取 v225 by-bucket test 指标中和当前经典问题最相关的风险桶。"""

    df = bucket_df.copy()
    df = df[df["split"].astype(str) == "test"].copy()
    for col in ["n", "rmse", "tail_rmse"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    wanted = {
        "strong_steer",
        "extreme_peak",
        "reverse",
        "zero_cross",
        "multi_correction",
        "vehicle_strong",
        "normal_curve",
        "high_tail_error",
        "under_flag",
    }
    df = df[df["bucket"].astype(str).isin(wanted)].copy()
    df["risk_note_cn"] = df.apply(bucket_risk_note, axis=1)
    return df.sort_values(["pool_key", "tail_rmse"], ascending=[True, False])


def bucket_risk_note(row: pd.Series) -> str:
    bucket = str(row.get("bucket", ""))
    if bucket == "extreme_peak":
        return "极端峰值是最大尾部风险，应优先进入失败图谱而不是直接调全局模型。"
    if bucket == "under_flag":
        return "低估样本直接对应经典幅值不足问题，需和强反应/极端峰值分开看。"
    if bucket == "high_tail_error":
        return "尾部误差集中，说明平均指标不能代表失败风险。"
    if bucket == "reverse":
        return "反转样本仍然会放大尾段误差，不能只看方向准确率。"
    if bucket == "strong_steer":
        return "强方向盘动作是主难点之一，容易被保守预测压平。"
    if bucket == "normal_curve":
        return "普通曲线相对可控，可作为不要过度牺牲的稳定区。"
    return "保留为正式失败桶对照。"


def make_selector_candidate_diagnosis(diagnostic_df: pd.DataFrame) -> pd.DataFrame:
    """把 diagnostic-only oracle/selector 结论整理成不能误写为 formal 的诊断表。"""

    rows = []
    for _, row in diagnostic_df.iterrows():
        summary = str(row.get("summary", ""))
        diag_name = str(row.get("diagnostic_name", ""))
        if diag_name not in {"oracle_safe_gate", "future_route_decision"}:
            continue
        item = {
            "pool": row.get("pool", ""),
            "diagnostic_name": diag_name,
            "usage": row.get("usage", "diagnostic_only"),
            "allowed_in_formal": row.get("allowed_in_formal", False),
            "split": row.get("split", ""),
            "rmse": row.get("rmse", ""),
            "tail_rmse": row.get("tail_rmse", ""),
            "summary": summary,
            "v229_interpretation_cn": "",
        }
        if "selector_failed_rate" in summary:
            item["v229_interpretation_cn"] = (
                "候选池常有上限，但 learned selector 在 locked test 上不稳；"
                "这支持先做失败分类，而不是直接训练更大 gate/router。"
            )
        elif "v222b_allowed=False" in summary:
            item["v229_interpretation_cn"] = (
                "当前证据没有解锁 v222b/v223；下一步应先让 GPTPro 审阅路线复盘和失败桶。"
            )
        elif diag_name == "oracle_safe_gate":
            item["v229_interpretation_cn"] = (
                "oracle 只能作为上限诊断，不能写入 formal headline 或可部署结论。"
            )
        rows.append(item)

    return pd.DataFrame(rows)


def make_decision_matrix() -> pd.DataFrame:
    """把 v229 复盘后的下一步选择写成显式决策矩阵。"""

    rows = [
        {
            "candidate_next_step": "直接训练 v222b/v223 或更大 gate/router",
            "decision": "不建议",
            "reason_cn": "v222a closeout 已显示 learned selector 在 locked test 不稳；继续扩大同类 selector 容易陷入局部过拟合。",
            "required_before_reopen_cn": "除非 GPTPro 给出新的 bounded scope，且失败分类证明 candidate_missing 是主因而非 selector_failed。",
        },
        {
            "candidate_next_step": "新增 tau/threshold 或基于 test 重新调 headline",
            "decision": "禁止",
            "reason_cn": "违反当前 formal lock 和 test discipline；v228 已冻结最终主结果。",
            "required_before_reopen_cn": "不应重开。",
        },
        {
            "candidate_next_step": "继续写作/论文材料整理",
            "decision": "可行",
            "reason_cn": "v228 已提供正式主表、CI、claim 和 limitation；结果边界清楚。",
            "required_before_reopen_cn": "保留 tail concentration、underestimation、样本量和 subject-block CI 限制说明。",
        },
        {
            "candidate_next_step": "失败样本 taxonomy + 人工复核少量高尾案例",
            "decision": "推荐",
            "reason_cn": "两个月经验表明必须先区分 candidate_missing、selector_failed、强峰值低估、反转/多修正和 input-indeterminate。",
            "required_before_reopen_cn": "先产出每类占比和代表图，再决定是否允许一个窄范围机制实验。",
        },
        {
            "candidate_next_step": "让 GPTPro 审阅 v229 复盘后给一个 bounded 下一步",
            "decision": "推荐",
            "reason_cn": "可避免 GPTPro 只基于最新指标给出局部调参建议，并把讨论约束到路线经验和失败分类。",
            "required_before_reopen_cn": "发送中文 prompt，要求明确 stop condition、是否只写作、是否允许失败分类审计。",
        },
    ]
    return pd.DataFrame(rows)


def format_float(value: object, digits: int = 6) -> str:
    try:
        if pd.isna(value):
            return ""
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def make_report(
    phase_lessons: pd.DataFrame,
    failure_taxonomy: pd.DataFrame,
    top_cases: pd.DataFrame,
    bucket_risk: pd.DataFrame,
    selector_diag: pd.DataFrame,
    decision_matrix: pd.DataFrame,
    final_main: pd.DataFrame,
    limitations: pd.DataFrame,
) -> str:
    """生成中文路线复盘报告。"""

    lines: list[str] = []
    lines.append("# v229 两个月路线经验复盘与失败分类报告")
    lines.append("")
    lines.append(f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}")
    lines.append("- 范围：只读复盘 v220/v225/v228；不训练模型，不生成新预测，不重选 formal headline。")
    lines.append("- 当前 formal lock：`loose_main_pool=avg_joint_focus`，`strict_main_pool=peak_floor_090`。")
    lines.append("")

    lines.append("## 一句话结论")
    lines.append("")
    lines.append(
        "这两个月的核心经验不是“模型还没堆够”，而是同一个瓶颈被多条路线反复证明："
        "模型能较稳定抓住方向和普通响应，但强反应幅值、极端峰值、尾段、反转/多次修正仍是主要失败区；"
        "候选池经常存在更好上限，真正困难是当前可部署输入下的可靠选择。"
    )
    lines.append("")

    lines.append("## v228 最终正式效果")
    lines.append("")
    lines.append("| pool | formal model | test n | RMSE | tail RMSE | direction acc | under rate | top20 tail share |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for _, row in final_main.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["pool_key"]),
                    str(row["formal_model"]),
                    str(row["n"]),
                    format_float(row["rmse"]),
                    format_float(row["tail_rmse"]),
                    format_float(row["direction_acc"]),
                    format_float(row["under_rate"]),
                    format_float(row["tail_top20pct_share"]),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append(
        "解读：方向准确率已经很高，但 top20 tail-SSE share 约 0.66-0.67，说明误差集中在少数难例，"
        "不能只用平均 RMSE 宣称问题已经解决。"
    )
    lines.append("")

    lines.append("## 两个月路线经验")
    lines.append("")
    lines.append("| 阶段 | 记录数 | 沉淀经验 | v229 对下一步的含义 |")
    lines.append("|---|---:|---|---|")
    for _, row in phase_lessons.iterrows():
        lines.append(
            f"| {row['phase']} | {row['attempt_count_in_phase']} | "
            f"{row['lesson']} | {row['v229_implication']} |"
        )
    lines.append("")

    lines.append("## test split 失败桶")
    lines.append("")
    lines.append("| pool | failure bucket | n | share | avg tail RMSE | max tail RMSE | under rate | top20 tail rate |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for _, row in failure_taxonomy.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["pool_key"]),
                    str(row["failure_bucket_v229"]),
                    str(int(row["n"])),
                    format_float(row["bucket_share_in_pool"], 3),
                    format_float(row["avg_tail_rmse"], 3),
                    format_float(row["max_tail_rmse"], 3),
                    format_float(row["under_rate_in_bucket"], 3),
                    format_float(row["top20_tail_rate_in_bucket"], 3),
                ]
            )
            + " |"
        )
    lines.append("")

    lines.append("## 最严重尾段案例的共同特征")
    lines.append("")
    worst = top_cases.sort_values("tail_rmse", ascending=False).head(12)
    lines.append("| pool | sample | event | bucket | tail RMSE | under | peak ratio |")
    lines.append("|---|---|---|---|---:|---|---:|")
    for _, row in worst.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["pool_key"]),
                    str(row["sample_id"]),
                    str(row["route_event"]),
                    str(row["failure_bucket_v229"]),
                    format_float(row["tail_rmse"], 3),
                    str(row["under_flag"]),
                    format_float(row["peak_ratio"], 3),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append(
        "这些最坏案例主要落在极端峰值、强事件和少量反转/车辆强响应上。它们解释了为什么普通曲线看起来还可以，"
        "但一看预测图就会出现“经典幅值压平/尾段跟不上”的感觉。"
    )
    lines.append("")

    lines.append("## selector / candidate 诊断边界")
    lines.append("")
    for _, row in selector_diag.iterrows():
        if str(row.get("v229_interpretation_cn", "")).strip():
            lines.append(
                f"- `{row['pool']} / {row['diagnostic_name']}`：{row['summary']}。"
                f"{row['v229_interpretation_cn']}"
            )
    lines.append("")
    lines.append(
        "这部分必须保持 diagnostic-only。oracle 或 selector gap 可以帮助解释路线，但不能写成正式可部署提升，"
        "也不能据此直接解锁 v222b/v223。"
    )
    lines.append("")

    lines.append("## v228 limitation 与论文边界")
    lines.append("")
    for _, row in limitations.iterrows():
        lines.append(
            f"- `{row['limitation_id']}`：证据 `{row['evidence']}`；影响：{row['impact_cn']}"
        )
    lines.append("")

    lines.append("## 下一步决策矩阵")
    lines.append("")
    lines.append("| 候选动作 | 决策 | 原因 | 重开条件 |")
    lines.append("|---|---|---|---|")
    for _, row in decision_matrix.iterrows():
        lines.append(
            f"| {row['candidate_next_step']} | {row['decision']} | "
            f"{row['reason_cn']} | {row['required_before_reopen_cn']} |"
        )
    lines.append("")

    lines.append("## 给 GPTPro 的建议提问方式")
    lines.append("")
    lines.append(
        "不要只问“下一步训练什么模型”。应把 v229 报告发给 GPTPro，请它先确认："
        "当前是否进入写作整理；如果继续实验，是否只允许失败样本 taxonomy/人工复核；"
        "是否明确禁止 v222b/v223、大 gate/router、新 tau/threshold 和 test-based retuning；"
        "如果允许新实验，必须给出单一窄范围目标和 stop condition。"
    )
    lines.append("")
    return "\n".join(lines)


def make_gptpro_prompt(report_path: Path, tables: dict[str, Path], final_main: pd.DataFrame) -> str:
    """生成可直接复制到本地 GPTPro 软件的中文 prompt。"""

    loose = final_main[final_main["pool_key"] == "loose_main_pool"].iloc[0]
    strict = final_main[final_main["pool_key"] == "strict_main_pool"].iloc[0]

    return f"""# 给 GPTPro 的中文复盘请求：v229 两个月经验与失败分类

请先阅读这个本地复盘包，然后只给一个 bounded 下一步建议，不要直接要求训练更大模型。

## 本地输出包

- v229 报告：`{report_path}`
- 失败桶统计：`{tables['failure_taxonomy']}`
- 高尾失败案例：`{tables['top_cases']}`
- selector/candidate 诊断：`{tables['selector_diag']}`
- 下一步决策矩阵：`{tables['decision_matrix']}`

## 当前正式锁定结果

- loose_main_pool = avg_joint_focus
  - test n = {int(loose['n'])}
  - RMSE = {format_float(loose['rmse'])}
  - tail RMSE = {format_float(loose['tail_rmse'])}
  - direction_acc = {format_float(loose['direction_acc'])}
  - under_rate = {format_float(loose['under_rate'])}
  - top20 tail-SSE share = {format_float(loose['tail_top20pct_share'])}
- strict_main_pool = peak_floor_090
  - test n = {int(strict['n'])}
  - RMSE = {format_float(strict['rmse'])}
  - tail RMSE = {format_float(strict['tail_rmse'])}
  - direction_acc = {format_float(strict['direction_acc'])}
  - under_rate = {format_float(strict['under_rate'])}
  - top20 tail-SSE share = {format_float(strict['tail_top20pct_share'])}

## 我希望你重点判断的问题

1. 这两个月的证据是否支持：当前应进入论文写作/结果整理，而不是继续模型搜索？
2. 如果还允许继续推进，是否应先做失败样本 taxonomy 和人工复核，而不是 v222b/v223、新 gate/router、新 tau/threshold？
3. 当前经典问题是否应表述为：方向和普通响应可预测，但强反应幅值、极端峰值、尾段、反转/多次修正仍是主要限制？
4. v225 diagnostic-only 结果显示 oracle/candidate 上限存在，但 learned selector 不稳。这个结论是否足以继续禁止同空间 current-window selector 扩大化？
5. 如果你认为必须继续实验，请只给一个窄范围任务，必须包含：
   - 允许输入与禁止输入；
   - validation-only 选择规则；
   - test reporting-only 规则；
   - 明确 stop condition；
   - 输出文件和验收命令。

## 本地边界

- 不允许 test-based retuning。
- 不允许把 oracle、true label、fallback 或 diagnostic-only 行写入 formal headline。
- 不允许把 W3_B4_original_soft 写入 formal leaderboard、formal gate、formal oracle、usage table 或 selected config。
- 不允许在没有明确 stop condition 的情况下启动 v222b/v223、大 gate/router、新 tau/threshold 或新模型训练。
- 请用中文回答，并优先给路线判断，不要只列模型名。
"""


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


def check_guardrails(final_main: pd.DataFrame, report_text: str) -> dict:
    """确认 v229 只做复盘，不改动 formal lock 或解锁新模型。"""

    lock_exact = True
    for pool_key, formal_model in FORMAL_MODEL_LOCK.items():
        matched = final_main[
            (final_main["pool_key"] == pool_key)
            & (final_main["formal_model"] == formal_model)
            & (final_main["test_used_for_v228_selection"] == False)
        ]
        lock_exact = lock_exact and not matched.empty

    banned_unlock_phrases = [
        "v222b_allowed=True",
        "v223_allowed=True",
        "test-based retuning allowed",
        "允许基于 test 调参",
    ]
    unlock_hits = [phrase for phrase in banned_unlock_phrases if phrase in report_text]

    return {
        "pass": lock_exact and not unlock_hits,
        "no_training_executed": True,
        "no_new_prediction_generated": True,
        "formal_model_lock_exact": lock_exact,
        "test_used_for_new_selection": False,
        "new_gate_or_router_created": False,
        "new_tau_or_threshold_created": False,
        "v222b_or_v223_unlocked": False,
        "unlock_phrase_hits": unlock_hits,
        "diagnostic_only_boundary_preserved": True,
    }


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
    write_csv(LOG_DIR / "input_file_hashes.csv", pd.DataFrame(rows))
    write_json(LOG_DIR / "input_file_hashes.json", rows)


def missing_required_files() -> list[str]:
    return [item for item in REQUIRED_RELATIVE_FILES if not (OUT_DIR / item).exists()]


def main() -> None:
    ensure_sources_exist()
    clean_output_dir()

    phase_df = read_csv(SOURCE_FILES["v220_phase_summary"])
    attempt_summary = read_json(SOURCE_FILES["v220_run_summary"])
    bucket_df = read_csv(SOURCE_FILES["v225_bucket_metrics"])
    failure_df = read_csv(SOURCE_FILES["v225_failure_case_index"])
    diagnostic_df = read_csv(SOURCE_FILES["v225_diagnostic_closeout"])
    final_main = read_csv(SOURCE_FILES["v228_final_main_result"])
    limitations = read_csv(SOURCE_FILES["v228_final_limitations"])
    claims = read_csv(SOURCE_FILES["v228_final_claim_lock"])

    phase_lessons = make_phase_lessons_table(phase_df, attempt_summary)
    failure_taxonomy, top_cases = make_failure_taxonomy(failure_df)
    bucket_risk = make_bucket_risk_summary(bucket_df)
    selector_diag = make_selector_candidate_diagnosis(diagnostic_df)
    decision_matrix = make_decision_matrix()

    table_paths = {
        "phase_lessons": TABLE_DIR / "v229_phase_lessons_table.csv",
        "failure_taxonomy": TABLE_DIR / "v229_failure_taxonomy_by_pool_event.csv",
        "top_cases": TABLE_DIR / "v229_top_tail_failure_cases.csv",
        "bucket_risk": TABLE_DIR / "v229_bucket_risk_summary.csv",
        "selector_diag": TABLE_DIR / "v229_selector_candidate_diagnosis.csv",
        "decision_matrix": TABLE_DIR / "v229_next_action_decision_matrix.csv",
        "claims_copy": TABLE_DIR / "v229_v228_claim_lock_reference.csv",
    }
    write_csv(table_paths["phase_lessons"], phase_lessons)
    write_csv(table_paths["failure_taxonomy"], failure_taxonomy)
    write_csv(table_paths["top_cases"], top_cases)
    write_csv(table_paths["bucket_risk"], bucket_risk)
    write_csv(table_paths["selector_diag"], selector_diag)
    write_csv(table_paths["decision_matrix"], decision_matrix)
    write_csv(table_paths["claims_copy"], claims)

    report_text = make_report(
        phase_lessons=phase_lessons,
        failure_taxonomy=failure_taxonomy,
        top_cases=top_cases,
        bucket_risk=bucket_risk,
        selector_diag=selector_diag,
        decision_matrix=decision_matrix,
        final_main=final_main,
        limitations=limitations,
    )
    report_path = REPORT_DIR / "v229_two_month_lessons_failure_taxonomy_cn.md"
    report_path.write_text(report_text, encoding="utf-8")

    gptpro_prompt = make_gptpro_prompt(
        report_path=report_path,
        tables=table_paths,
        final_main=final_main,
    )
    prompt_path = REPORT_DIR / "v229_gptpro_next_prompt_cn.md"
    prompt_path.write_text(gptpro_prompt, encoding="utf-8")

    write_input_hashes()

    guardrail = check_guardrails(final_main, report_text + "\n" + gptpro_prompt)
    write_json(LOG_DIR / "guardrail_check.json", guardrail)

    run_manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "script": rel(Path(__file__).resolve()),
        "output_dir": rel(OUT_DIR),
        "scope": "retrospective_only_no_training_no_new_prediction",
        "source_dirs": {
            "v220": rel(V220_DIR),
            "v225": rel(V225_DIR),
            "v228": rel(V228_DIR),
        },
        "formal_model_lock": FORMAL_MODEL_LOCK,
        "main_report": rel(report_path),
        "gptpro_prompt": rel(prompt_path),
        "tables": {k: rel(v) for k, v in table_paths.items()},
        "guardrail_pass": guardrail["pass"],
    }
    write_json(LOG_DIR / "run_manifest.json", run_manifest)

    file_inventory_pre_zip = collect_file_inventory()
    write_csv(LOG_DIR / "file_inventory.csv", file_inventory_pre_zip)
    write_json(LOG_DIR / "file_inventory.json", file_inventory_pre_zip.to_dict(orient="records"))

    zip_path = zip_output()
    missing = missing_required_files()
    if missing:
        raise RuntimeError(f"Missing required output files: {missing}")

    # ZIP 生成后刷新一次 inventory，让压缩包本身也进入索引。
    file_inventory = collect_file_inventory()
    write_csv(LOG_DIR / "file_inventory.csv", file_inventory)
    write_json(LOG_DIR / "file_inventory.json", file_inventory.to_dict(orient="records"))

    print(json.dumps(
        {
            "output_dir": str(OUT_DIR),
            "report": str(report_path),
            "gptpro_prompt": str(prompt_path),
            "zip": str(zip_path),
            "guardrail_pass": guardrail["pass"],
            "missing_required_files": missing,
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
