#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v315 rapid steering filter / reanchor plan.

目的：
- 承接 v314 的方向盘快速转动来源审计；
- 将 84 个“当前窗口快转证据不足或来源错位”事件转成可执行的数据处理策略；
- 输出过滤训练表、重锚定候选表和统计图，为下一轮模型训练前的数据边界修正做准备。

边界：
- 本脚本不训练模型；
- 本脚本不直接改原始样本清单；
- 重锚定时间只是候选建议，后续需要专门脚本重切窗口并重新生成目标曲线；
- 不用测试误差选择训练候选，测试误差信息只作为诊断字段保留。
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASELINES = ROOT / "03_baselines"
V314_DIR = BASELINES / "v314_rapid_steering_source_sample_audit_20260704"
V314_AUDIT = V314_DIR / "tables" / "v314_rapid_steering_source_audit_all_delay0.csv"
V314_SUMMARY = V314_DIR / "tables" / "v314_source_category_summary.csv"

OUT = BASELINES / "v315_rapid_steering_filter_reanchor_plan_20260704"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"

LATE_REANCHOR_MARGIN_S = 0.50
PRE_REANCHOR_MARGIN_S = 0.30


POLICY_CN = {
    "keep_current_window": "保留当前窗口样本",
    "isolate_late_fast_for_reanchor": "隔离：当前平缓但后续才快转，候选后移锚点",
    "isolate_pre_fast_for_reanchor": "隔离：锚点前已快转，候选前移锚点",
    "exclude_weak_fast_source": "隔离：全程快转证据弱，候选剔除",
    "isolate_ambiguous_source": "隔离：快转来源不清晰，候选复查锚点",
    "isolate_missing_source": "隔离：原始快转证据缺失",
}

NEXT_ACTION_CN = {
    "train_with_current_window": "可进入当前窗口训练",
    "reanchor_later_candidate": "候选后移锚点后重新切窗",
    "reanchor_earlier_candidate": "候选前移锚点后重新切窗",
    "exclude_from_current_task": "从当前任务中剔除或单独归档",
    "inspect_source_alignment": "单独检查来源对齐",
}


def ensure_dirs() -> None:
    """创建输出目录。"""

    for folder in (TABLES, FIGURES, REPORTS, LOGS):
        folder.mkdir(parents=True, exist_ok=True)


def clean_out_dir() -> None:
    """只清理第315版自己的输出目录。"""

    resolved_out = OUT.resolve()
    resolved_base = BASELINES.resolve()
    if resolved_base not in resolved_out.parents:
        raise RuntimeError(f"拒绝清理非预期目录：{resolved_out}")
    if OUT.exists():
        shutil.rmtree(OUT)
    ensure_dirs()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """表格使用 utf-8-sig，方便 Windows 表格软件查看中文。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_json(payload: Dict[str, object], path: Path) -> None:
    """保存 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def file_sha256(path: Path) -> str:
    """计算文件哈希。"""

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def to_float(value: object, default: float = math.nan) -> float:
    """安全转成浮点数。"""

    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def to_bool(value: object) -> bool:
    """兼容常见布尔文本。"""

    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y", "是"}


def markdown_table(df: pd.DataFrame) -> str:
    """不依赖额外包，生成简单 Markdown 表格。"""

    if df.empty:
        return "（空表）"
    cols = list(df.columns)

    def cell(value: object) -> str:
        if isinstance(value, float):
            text = f"{value:.6g}" if np.isfinite(value) else ""
        else:
            text = str(value)
        return text.replace("|", "｜").replace("\n", " ")

    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(cell(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def load_audit_table() -> pd.DataFrame:
    """读取第314版全量排查表并规范字段类型。"""

    if not V314_AUDIT.exists():
        raise FileNotFoundError(f"缺少第314版全量排查表：{V314_AUDIT}")
    df = pd.read_csv(V314_AUDIT, encoding="utf-8-sig")
    bool_cols = [
        "raw_available_for_rate",
        "fast_current",
        "fast_late",
        "fast_pre",
        "fast_near_anchor",
        "is_v309_severe",
        "is_user_screenshot_case",
        "fast_steer_source_ok_current",
        "suspect_not_current_fast_steer",
        "strong_steer",
        "vehicle_strong",
        "coarse_label_horizon_mismatch",
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].map(to_bool)
    float_cols = [
        "observation_s",
        "rate_current_peak_abs",
        "rate_current_peak_time_s",
        "rate_late_peak_abs",
        "rate_late_peak_time_s",
        "rate_pre_peak_abs",
        "rate_pre_peak_time_s",
        "delta_current_peak_abs_raw",
        "delta_late_peak_abs_raw",
        "local_0_2_peak_abs",
        "late_2_6_peak_abs",
        "late_over_local_abs_ratio",
        "v307_rmse",
        "v300_rmse",
        "delta_v307_minus_v300",
    ]
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].map(to_float)
    return df


def choose_policy(row: pd.Series) -> Dict[str, object]:
    """将第314版来源分级转成第315版训练前处理策略。"""

    source = str(row.get("source_category", ""))
    obs = to_float(row.get("observation_s"))
    shift = math.nan
    new_obs = math.nan
    reason = ""

    if source in {"current_and_late_fast_steer", "current_window_fast_steer_supported"}:
        policy = "keep_current_window"
        action = "train_with_current_window"
        include_current = True
        include_after_reanchor = False
        reason = "当前0到2秒窗口内有方向盘快转证据，样本来源符合当前任务定义。"
    elif source == "late_fast_steer_not_current_window":
        policy = "isolate_late_fast_for_reanchor"
        action = "reanchor_later_candidate"
        include_current = False
        include_after_reanchor = True
        peak_t = to_float(row.get("rate_late_peak_time_s"))
        if np.isfinite(peak_t):
            shift = max(0.0, peak_t - LATE_REANCHOR_MARGIN_S)
            new_obs = obs + shift if np.isfinite(obs) else math.nan
        reason = "当前窗口快转证据不足，但后续窗口出现明显快转；当前任务中应隔离，候选后移锚点。"
    elif source == "anchor_after_fast_steer":
        policy = "isolate_pre_fast_for_reanchor"
        action = "reanchor_earlier_candidate"
        include_current = False
        include_after_reanchor = True
        peak_t = to_float(row.get("rate_pre_peak_time_s"))
        if np.isfinite(peak_t):
            shift = min(-0.05, peak_t - PRE_REANCHOR_MARGIN_S)
            new_obs = obs + shift if np.isfinite(obs) else math.nan
        reason = "锚点前已经出现明显快转；当前锚点可能偏晚，候选前移锚点。"
    elif source == "no_clear_fast_steer_evidence":
        policy = "exclude_weak_fast_source"
        action = "exclude_from_current_task"
        include_current = False
        include_after_reanchor = False
        reason = "锚点前后均缺少方向盘快转证据，不符合当前样本定义，优先候选剔除。"
    elif source == "raw_missing_or_invalid":
        policy = "isolate_missing_source"
        action = "inspect_source_alignment"
        include_current = False
        include_after_reanchor = False
        reason = "原始车辆方向盘信号缺失或无效，不能确认样本来源。"
    else:
        policy = "isolate_ambiguous_source"
        action = "inspect_source_alignment"
        include_current = False
        include_after_reanchor = False
        reason = "快转来源不清晰，不能直接进入当前窗口训练。"

    return {
        "v315_policy": policy,
        "v315_policy_cn": POLICY_CN[policy],
        "v315_next_action": action,
        "v315_next_action_cn": NEXT_ACTION_CN[action],
        "v315_include_current_window_training": include_current,
        "v315_candidate_after_reanchor": include_after_reanchor,
        "v315_candidate_anchor_shift_s": shift,
        "v315_candidate_observation_s": new_obs,
        "v315_policy_reason_cn": reason,
    }


def build_policy_table(audit: pd.DataFrame) -> pd.DataFrame:
    """生成全量第315版处理策略表。"""

    rows: List[Dict[str, object]] = []
    for _, row in audit.iterrows():
        record = row.to_dict()
        record.update(choose_policy(row))
        rows.append(record)
    out = pd.DataFrame(rows)
    out["v315_current_window_train_keep"] = out["v315_include_current_window_training"].map(bool)
    out["v315_isolate_from_current_window"] = ~out["v315_current_window_train_keep"]
    return out


def build_summaries(policy: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """生成过滤与重锚定统计表。"""

    split_summary = (
        policy.groupby("split", as_index=False)
        .agg(
            original_event_n=("event_uid", "count"),
            keep_current_window_n=("v315_current_window_train_keep", "sum"),
            isolate_current_window_n=("v315_isolate_from_current_window", "sum"),
            reanchor_candidate_n=("v315_candidate_after_reanchor", "sum"),
            severe_n=("is_v309_severe", "sum"),
            screenshot_n=("is_user_screenshot_case", "sum"),
        )
        .sort_values("split")
    )
    split_summary["keep_rate"] = split_summary["keep_current_window_n"] / split_summary["original_event_n"].clip(lower=1)

    policy_summary = (
        policy.groupby(["v315_policy", "v315_policy_cn", "v315_next_action_cn"], as_index=False)
        .agg(event_n=("event_uid", "count"), severe_n=("is_v309_severe", "sum"), screenshot_n=("is_user_screenshot_case", "sum"))
        .sort_values("event_n", ascending=False)
    )
    scene_summary = (
        policy.groupby(["coarse_scene_label", "coarse_scene_label_cn", "v315_policy_cn"], as_index=False)
        .agg(event_n=("event_uid", "count"), severe_n=("is_v309_severe", "sum"))
        .sort_values(["coarse_scene_label_cn", "event_n"], ascending=[True, False])
    )
    reanchor_summary = (
        policy[policy["v315_candidate_after_reanchor"]]
        .groupby(["split", "v315_next_action_cn"], as_index=False)
        .agg(
            event_n=("event_uid", "count"),
            mean_shift_s=("v315_candidate_anchor_shift_s", "mean"),
            median_shift_s=("v315_candidate_anchor_shift_s", "median"),
            min_shift_s=("v315_candidate_anchor_shift_s", "min"),
            max_shift_s=("v315_candidate_anchor_shift_s", "max"),
        )
        .sort_values(["split", "v315_next_action_cn"])
    )
    return {
        "split_summary": split_summary,
        "policy_summary": policy_summary,
        "scene_summary": scene_summary,
        "reanchor_summary": reanchor_summary,
    }


def write_summary_figures(split_summary: pd.DataFrame, policy_summary: pd.DataFrame) -> List[Path]:
    """生成统计图。"""

    paths: List[Path] = []
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    x = np.arange(len(split_summary))
    width = 0.35
    ax.bar(x - width / 2, split_summary["keep_current_window_n"], width=width, label="保留", color="#2563EB")
    ax.bar(x + width / 2, split_summary["isolate_current_window_n"], width=width, label="隔离", color="#DC2626")
    ax.set_xticks(x)
    ax.set_xticklabels(split_summary["split"].astype(str).tolist())
    ax.set_ylabel("事件数")
    ax.set_title("第315版按数据划分的保留与隔离数量")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    p1 = FIGURES / "v315_split_keep_isolate_counts.png"
    fig.savefig(p1, dpi=160)
    plt.close(fig)
    paths.append(p1)

    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    tmp = policy_summary.sort_values("event_n", ascending=True)
    ax.barh(tmp["v315_policy_cn"], tmp["event_n"], color="#0F766E")
    ax.set_xlabel("事件数")
    ax.set_title("第315版处理策略分布")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    p2 = FIGURES / "v315_policy_counts.png"
    fig.savefig(p2, dpi=160)
    plt.close(fig)
    paths.append(p2)
    return paths


def write_file_inventory() -> pd.DataFrame:
    """记录输出文件清单。"""

    rows = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "file_inventory.csv":
            rows.append(
                {
                    "relative_path": str(path.relative_to(OUT)),
                    "size_bytes": int(path.stat().st_size),
                    "sha256": file_sha256(path),
                }
            )
    inv = pd.DataFrame(rows)
    write_csv(inv, LOGS / "file_inventory.csv")
    return inv


def make_zip_package() -> Tuple[Path, bool]:
    """打包产物并做压缩包自检。"""

    zip_path = OUT / "v315_rapid_steering_filter_reanchor_plan_20260704.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(OUT.rglob("*")):
            if path == zip_path or not path.is_file():
                continue
            zf.write(path, path.relative_to(OUT))
    with zipfile.ZipFile(zip_path, "r") as zf:
        bad = zf.testzip()
    return zip_path, bad is None


def write_report(policy: pd.DataFrame, summaries: Dict[str, pd.DataFrame], guardrail: Dict[str, object]) -> Path:
    """写中文报告。"""

    total = len(policy)
    keep = int(policy["v315_current_window_train_keep"].sum())
    isolate = int(policy["v315_isolate_from_current_window"].sum())
    reanchor = int(policy["v315_candidate_after_reanchor"].sum())
    severe_isolate = int((policy["is_v309_severe"] & policy["v315_isolate_from_current_window"]).sum())
    screenshot_isolate = int((policy["is_user_screenshot_case"] & policy["v315_isolate_from_current_window"]).sum())

    lines = [
        "# 第315版方向盘快转过滤与重锚定候选方案",
        "",
        "## 结论",
        "",
        "- 本轮不训练模型，只把第314版来源审计转成下一轮训练前的数据处理策略。",
        f"- 全量事件：`{total}`。",
        f"- 保留当前窗口训练：`{keep}`。",
        f"- 从当前窗口训练隔离：`{isolate}`。",
        f"- 其中候选重锚定：`{reanchor}`。",
        f"- 第309版严重错误样本中需隔离：`{severe_isolate}`。",
        f"- 用户截图样本中需隔离：`{screenshot_isolate}`，即此前确认的 #020。",
        "",
        "## 主要输出",
        "",
        f"- 全量处理策略表：`{TABLES / 'v315_current_window_training_policy_all_delay0.csv'}`",
        f"- 当前任务保留清单：`{TABLES / 'v315_current_window_keep_manifest.csv'}`",
        f"- 当前任务隔离清单：`{TABLES / 'v315_current_window_isolate_manifest.csv'}`",
        f"- 重锚定候选表：`{TABLES / 'v315_reanchor_candidate_manifest.csv'}`",
        f"- 按划分统计：`{TABLES / 'v315_split_filter_summary.csv'}`",
        "",
        "## 按划分统计",
        "",
        markdown_table(summaries["split_summary"]),
        "",
        "## 处理策略分布",
        "",
        markdown_table(summaries["policy_summary"]),
        "",
        "## 重锚定候选统计",
        "",
        markdown_table(summaries["reanchor_summary"]),
        "",
        "## 后续建议",
        "",
        "- 下一轮若训练当前0到2秒任务，应先使用保留清单，隔离清单不参与当前窗口强动作监督。",
        "- 重锚定候选需要重新切车辆窗口和目标曲线，不能只改表里的锚点时间后直接训练。",
        "- 来源成立但仍预测差的严重样本，进入幅值、相位和极端动作跟随修正；来源不成立的样本不应再用于惩罚模型。",
    ]
    path = REPORTS / "v315_rapid_steering_filter_reanchor_plan_cn.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    clean_out_dir()
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    audit = load_audit_table()
    policy = build_policy_table(audit)
    keep = policy[policy["v315_current_window_train_keep"]].copy()
    isolate = policy[policy["v315_isolate_from_current_window"]].copy()
    reanchor = policy[policy["v315_candidate_after_reanchor"]].copy()
    weak_exclude = policy[policy["v315_policy"].eq("exclude_weak_fast_source")].copy()

    summaries = build_summaries(policy)
    write_csv(policy, TABLES / "v315_current_window_training_policy_all_delay0.csv")
    write_csv(keep, TABLES / "v315_current_window_keep_manifest.csv")
    write_csv(isolate, TABLES / "v315_current_window_isolate_manifest.csv")
    write_csv(reanchor, TABLES / "v315_reanchor_candidate_manifest.csv")
    write_csv(weak_exclude, TABLES / "v315_weak_fast_source_exclusion_candidates.csv")
    write_csv(summaries["split_summary"], TABLES / "v315_split_filter_summary.csv")
    write_csv(summaries["policy_summary"], TABLES / "v315_policy_summary.csv")
    write_csv(summaries["scene_summary"], TABLES / "v315_scene_policy_summary.csv")
    write_csv(summaries["reanchor_summary"], TABLES / "v315_reanchor_shift_summary.csv")
    figure_paths = write_summary_figures(summaries["split_summary"], summaries["policy_summary"])

    input_hashes = pd.DataFrame(
        [
            {"input_name": "v314_audit_table", "path": str(V314_AUDIT), "sha256": file_sha256(V314_AUDIT)},
            {"input_name": "v314_summary_table", "path": str(V314_SUMMARY), "sha256": file_sha256(V314_SUMMARY) if V314_SUMMARY.exists() else ""},
        ]
    )
    write_csv(input_hashes, LOGS / "input_hashes.csv")

    guardrail: Dict[str, object] = {
        "pass": True,
        "version": "v315_rapid_steering_filter_reanchor_plan_20260704",
        "training_run": False,
        "event_n": int(len(policy)),
        "current_window_keep_n": int(len(keep)),
        "current_window_isolate_n": int(len(isolate)),
        "reanchor_candidate_n": int(len(reanchor)),
        "weak_fast_source_exclusion_candidate_n": int(len(weak_exclude)),
        "severe_isolate_n": int((policy["is_v309_severe"] & policy["v315_isolate_from_current_window"]).sum()),
        "screenshot_isolate_n": int((policy["is_user_screenshot_case"] & policy["v315_isolate_from_current_window"]).sum()),
        "uses_test_error_as_training_feature": False,
        "candidate_selection_uses_test": False,
        "directly_changes_original_manifest": False,
        "reanchor_candidates_require_window_rebuild": True,
        "figure_paths": [str(p) for p in figure_paths],
    }
    report_path = write_report(policy, summaries, guardrail)
    guardrail["report_path"] = str(report_path)
    write_json(guardrail, LOGS / "guardrail_check.json")
    write_file_inventory()
    zip_path, zip_ok = make_zip_package()
    guardrail["zip_path"] = str(zip_path)
    guardrail["zip_testzip"] = bool(zip_ok)
    write_json(guardrail, LOGS / "guardrail_check.json")
    write_file_inventory()
    print(json.dumps(guardrail, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
