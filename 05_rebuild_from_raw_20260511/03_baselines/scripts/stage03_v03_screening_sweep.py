# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(os.environ.get("DATA_PROCESS_ROOT", r"F:/data_set_process/data_process"))
ROOT = Path(os.environ.get("REBUILD_ROOT", str(PROJECT_ROOT / "05_rebuild_from_raw_20260511")))
SCRIPT_DIR = ROOT / "03_baselines" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import stage03_v03_vehicle_only_inclusion_ablation as incl  # noqa: E402


OUT_ROOT = ROOT / "03_baselines" / "stage03_v03_screening_sweep"
DATASET_ROOT = ROOT / "03_processed_datasets" / "extreme_condition_v0_3_screening_sweep"
REPORT_DIR = ROOT / "09_reports"
NOTES_DIR = ROOT / "00_project_notes"
DAILY_LOG = NOTES_DIR / "daily_logs" / "2026-05-19.md"
ARTIFACT_INDEX = NOTES_DIR / "ARTIFACT_INDEX_CN.md"
SERVER_RUNS = NOTES_DIR / "SERVER_RUNS_CN.md"

TABLE_ROOT = ROOT / "02_samples" / "extreme_condition_episodes_v0_3" / "tables"
EPISODE_TABLE = TABLE_ROOT / "extreme_condition_episodes_all_v0_3.csv"
NEW_RULE_TABLE = TABLE_ROOT / "new_rule_auto_candidate_groups_v0_3.csv"
FAST_SPLIT_TABLE = TABLE_ROOT / "fast_steer_vehicle_response_split_v0_3.csv"
ANCHOR_TIMING_TABLE = TABLE_ROOT / "fast_steer_anchor_timing_audit_v0_3.csv"

SUMMARY_PATH = OUT_ROOT / "tables" / "v03_screening_sweep_summary.csv"
EXTRA_SOURCE_PATH = OUT_ROOT / "tables" / "v03_screening_sweep_extra_source_counts.csv"
RANKING_PATH = OUT_ROOT / "tables" / "v03_screening_sweep_ranking.csv"

BASE_CATEGORIES = sorted(incl.CLEAN_CATEGORIES | {"manual_review"})
DROP_COORDINATE_RISK_FEATURES = ["lateral_distance_selected"]
LATE_ANCHOR_LABELS = {"EXCLUDE_LATE_ANCHOR_STABILIZED", "RISK_LATE_ANCHOR_REVIEW"}


def ensure_dirs() -> None:
    for path in [OUT_ROOT / "tables", REPORT_DIR, NOTES_DIR / "daily_logs"]:
        path.mkdir(parents=True, exist_ok=True)


def configure_inclusion_module() -> None:
    incl.OUT_ROOT = OUT_ROOT
    incl.DATASET_ROOT = DATASET_ROOT
    incl.REPORT_DIR = REPORT_DIR
    incl.NOTES_DIR = NOTES_DIR
    incl.DAILY_LOG = DAILY_LOG
    incl.ARTIFACT_INDEX = ARTIFACT_INDEX


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def uid_set(df: pd.DataFrame, mask: pd.Series) -> set[str]:
    return set(df.loc[mask.fillna(False), "episode_uid"].dropna().astype(str))


def load_uid_groups() -> tuple[dict[str, set[str]], pd.DataFrame]:
    episodes = read_table(EPISODE_TABLE)
    episodes["episode_uid"] = episodes["episode_uid"].astype(str)
    base_uids = uid_set(episodes, episodes["v0_3_category"].isin(BASE_CATEGORIES))

    new_rule = read_table(NEW_RULE_TABLE)
    new_rule["episode_uid"] = new_rule["episode_uid"].astype(str)
    fast = read_table(FAST_SPLIT_TABLE)
    fast["episode_uid"] = fast["episode_uid"].astype(str)
    timing = read_table(ANCHOR_TIMING_TABLE)
    timing["episode_uid"] = timing["episode_uid"].astype(str)

    late_uids = uid_set(timing, timing["anchor_timing_label"].astype(str).isin(LATE_ANCHOR_LABELS))
    groups: dict[str, set[str]] = {}

    groups["base"] = base_uids
    groups["weakpost"] = uid_set(timing, timing["anchor_timing_label"].astype(str).eq("FAST_STEER_WEAK_POST_RESPONSE"))
    groups["fast_visible"] = uid_set(
        fast, fast["fast_vehicle_response_split"].astype(str).eq("FAST_STEER_WITH_VISIBLE_VEHICLE_RESPONSE")
    )
    groups["fast_visible_boundary"] = uid_set(
        fast,
        fast["fast_vehicle_response_split"]
        .astype(str)
        .isin(["FAST_STEER_WITH_VISIBLE_VEHICLE_RESPONSE", "FAST_STEER_BODY_RESPONSE_BOUNDARY"]),
    )
    groups["fast_all_nonlate"] = set(fast["episode_uid"].dropna().astype(str)) - late_uids

    groups["new_keep_extreme"] = uid_set(new_rule, new_rule["new_rule_auto_label"].astype(str).eq("KEEP_EXTREME_MAIN"))
    groups["new_body_strong"] = uid_set(new_rule, new_rule["new_rule_auto_label"].astype(str).eq("RISK_POOL_BODY_STRONG"))
    groups["new_boundary"] = uid_set(new_rule, new_rule["new_rule_auto_label"].astype(str).eq("MANUAL_REVIEW_BOUNDARY"))
    groups["new_keep_weak"] = uid_set(new_rule, new_rule["new_rule_auto_label"].astype(str).eq("KEEP_WEAK_CONSERVATIVE"))
    groups["new_keep_delay"] = uid_set(
        new_rule, new_rule["new_rule_auto_label"].astype(str).isin(["KEEP_DELAYED", "KEEP_DELAYED_WEAK"])
    )
    groups["new_selected_nonlight"] = uid_set(
        new_rule,
        ~new_rule["new_rule_auto_label"]
        .astype(str)
        .isin(["NORMAL_CONTROL_OR_EXCLUDE_LIGHT_STEER", "EXCLUDE_LIGHT_OR_COORD_RISK"]),
    )
    groups["strong_body_attitude"] = uid_set(new_rule, new_rule["strong_body_attitude_auto"].astype(str).eq("True"))
    groups["moderate_body_attitude"] = uid_set(new_rule, new_rule["moderate_body_attitude_auto"].astype(str).eq("True"))
    groups["steer_and_body"] = uid_set(
        new_rule,
        new_rule["strong_or_medium_steer_auto"].astype(str).eq("True")
        & new_rule["moderate_body_attitude_auto"].astype(str).eq("True"),
    )

    groups["excluded_low_mu"] = uid_set(
        episodes, episodes["v0_3_category"].astype(str).eq("excluded") & episodes["condition_context_cn"].astype(str).eq("低附着")
    )
    groups["excluded_roll"] = uid_set(
        episodes, episodes["v0_3_category"].astype(str).eq("excluded") & episodes["condition_context_cn"].astype(str).eq("横滚/姿态")
    )
    groups["excluded_curve"] = uid_set(
        episodes, episodes["v0_3_category"].astype(str).eq("excluded") & episodes["condition_context_cn"].astype(str).eq("弯道/曲率")
    )
    groups["excluded_lateral_dyn"] = uid_set(
        episodes, episodes["v0_3_category"].astype(str).eq("excluded") & episodes["condition_context_cn"].astype(str).eq("横向动态")
    )
    groups["excluded_context_all"] = uid_set(episodes, episodes["v0_3_category"].astype(str).eq("excluded"))

    for key in list(groups):
        if key != "base":
            groups[key] = groups[key] - late_uids

    rows = []
    for name, values in sorted(groups.items()):
        rows.append(
            {
                "group_name": name,
                "uid_count": len(values),
                "extra_beyond_base": len(values - base_uids),
                "overlap_base": len(values & base_uids),
            }
        )
    return groups, pd.DataFrame(rows)


def union_groups(groups: dict[str, set[str]], names: list[str]) -> list[str]:
    out: set[str] = set()
    for name in names:
        out |= groups.get(name, set())
    return sorted(out)


def make_variants(groups: dict[str, set[str]]) -> list[dict[str, Any]]:
    def variant(variant_id: str, name_cn: str, group_names: list[str], with_lateral: bool = False) -> dict[str, Any]:
        item: dict[str, Any] = {
            "variant_id": variant_id,
            "name_cn": name_cn,
            "description_cn": "v0.3 样本筛选策略连续对比，只改变额外纳入的 episode 范围。",
            "categories": BASE_CATEGORIES,
            "extra_episode_uids": union_groups(groups, group_names),
            "extra_episode_source": "+".join(group_names),
        }
        if not with_lateral:
            item["drop_features"] = DROP_COORDINATE_RISK_FEATURES
        return item

    return [
        variant("s00_base_nolat", "基础：干净集 + 待复核，去横向偏移", []),
        variant("s01_weakpost_nolat", "加锚点后响应弱，去横向偏移", ["weakpost"]),
        variant("s02_fast_visible_nolat", "加快速转向且车辆响应可见，去横向偏移", ["fast_visible"]),
        variant("s03_fast_visible_boundary_nolat", "加快速转向可见/边界车辆响应，去横向偏移", ["fast_visible_boundary"]),
        variant("s04_fast_all_nonlate_nolat", "加全部非偏晚快速转向候选，去横向偏移", ["fast_all_nonlate"]),
        variant("s05_keep_extreme_nolat", "加新规则核心极限样本，去横向偏移", ["new_keep_extreme"]),
        variant("s06_body_strong_nolat", "加强车身姿态候选，去横向偏移", ["new_body_strong"]),
        variant("s07_steer_body_nolat", "加方向盘中强且车身姿态明显，去横向偏移", ["steer_and_body"]),
        variant("s08_keep_weak_nolat", "加弱/保守响应，去横向偏移", ["new_keep_weak"]),
        variant("s09_keep_delay_nolat", "加延迟/无明显转向响应，去横向偏移", ["new_keep_delay"]),
        variant("s10_lowmu_excl_nolat", "加低附着 excluded，去横向偏移", ["excluded_low_mu"]),
        variant("s11_roll_excl_nolat", "加横滚/姿态 excluded，去横向偏移", ["excluded_roll"]),
        variant("s12_curve_excl_nolat", "加弯道/曲率 excluded，去横向偏移", ["excluded_curve"]),
        variant("s13_lowmu_roll_nolat", "加低附着 + 横滚/姿态 excluded，去横向偏移", ["excluded_low_mu", "excluded_roll"]),
        variant("s14_selected_nonlight_nolat", "加非轻微/非坐标风险自动候选，去横向偏移", ["new_selected_nonlight"]),
        variant("s15_all_context_excl_nolat", "加全部 excluded，去横向偏移", ["excluded_context_all"]),
        variant("s16_weakpost_lat", "加锚点后响应弱，保留横向偏移", ["weakpost"], with_lateral=True),
        variant("s17_roll_excl_lat", "加横滚/姿态 excluded，保留横向偏移", ["excluded_roll"], with_lateral=True),
        variant("s18_selected_nonlight_lat", "加非轻微/非坐标风险自动候选，保留横向偏移", ["new_selected_nonlight"], with_lateral=True),
    ]


def load_existing_result(variant_id: str) -> dict[str, Any] | None:
    path = OUT_ROOT / variant_id / "logs" / f"{variant_id}_summary.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    result = payload.get("result")
    return result if isinstance(result, dict) else None


def load_dataset_summary(variant_id: str) -> dict[str, Any]:
    path = DATASET_ROOT / variant_id / "logs" / f"{variant_id}_dataset_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any) -> str:
    try:
        v = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(v):
        return "NA"
    return f"{v:.4f}"


def score_rows(summary: pd.DataFrame) -> pd.DataFrame:
    df = summary.copy()
    for col in [
        "test_rmse_steer",
        "test_wrong_side_rate_large",
        "test_severe_amp_under_rate_large",
        "test_large_response_recall",
        "test_tail_rmse_2_5s",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    base = df[df["variant_id"].eq("s00_base_nolat")]
    if len(base):
        b = base.iloc[0]
        df["delta_rmse_vs_base"] = df["test_rmse_steer"] - float(b["test_rmse_steer"])
        df["delta_wrong_side_vs_base"] = df["test_wrong_side_rate_large"] - float(b["test_wrong_side_rate_large"])
        df["delta_severe_under_vs_base"] = df["test_severe_amp_under_rate_large"] - float(
            b["test_severe_amp_under_rate_large"]
        )
        df["delta_large_recall_vs_base"] = df["test_large_response_recall"] - float(b["test_large_response_recall"])
    else:
        df["delta_rmse_vs_base"] = np.nan
        df["delta_wrong_side_vs_base"] = np.nan
        df["delta_severe_under_vs_base"] = np.nan
        df["delta_large_recall_vs_base"] = np.nan

    # Lower is better for errors; higher is better for recall. This is only a sorting aid.
    df["screening_score"] = (
        -df["delta_rmse_vs_base"].fillna(0.0)
        - 0.35 * df["delta_wrong_side_vs_base"].fillna(0.0)
        - 0.25 * df["delta_severe_under_vs_base"].fillna(0.0)
        + 0.15 * df["delta_large_recall_vs_base"].fillna(0.0)
    )
    return df.sort_values(["screening_score", "test_rmse_steer"], ascending=[False, True])


def markdown_table(df: pd.DataFrame, limit: int = 12) -> str:
    cols = [
        "variant_id",
        "name_cn",
        "sample_count",
        "extra_episode_count",
        "test_rmse_steer",
        "delta_rmse_vs_base",
        "test_wrong_side_rate_large",
        "test_severe_amp_under_rate_large",
        "test_large_response_recall",
        "screening_score",
    ]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.head(limit)[cols].iterrows():
        vals = []
        for col in cols:
            val = row[col]
            vals.append(fmt(val) if col not in ["variant_id", "name_cn"] else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(ranking: pd.DataFrame) -> None:
    report = REPORT_DIR / "stage03_v03_screening_sweep_user_summary_cn.md"
    best_rmse = ranking.sort_values("test_rmse_steer").iloc[0].to_dict()
    best_score = ranking.iloc[0].to_dict()
    base = ranking[ranking["variant_id"].eq("s00_base_nolat")].iloc[0].to_dict()
    lines = [
        "# v0.3 样本筛选策略连续对比",
        "",
        "## 为什么做",
        "",
        "当前训练样本只有 800 多个，直接全丢 excluded 会太少，但全量加入又会混入锚点偏晚、轻微直线维持、坐标风险和正常弯道等样本。本轮在服务器上连续尝试多种筛选策略，只改样本纳入范围，不改模型结构，用同一套车辆-only 基线比较哪类样本更值得纳入。",
        "",
        "## 基准",
        "",
        f"- 基础版本 `s00_base_nolat`：样本数 {int(base['sample_count'])}，test RMSE={fmt(base['test_rmse_steer'])}，大响应错侧率={fmt(base['test_wrong_side_rate_large'])}，严重幅值不足率={fmt(base['test_severe_amp_under_rate_large'])}，大响应召回={fmt(base['test_large_response_recall'])}。",
        "",
        "## 排名前 12 的筛选策略",
        "",
        markdown_table(ranking, 12),
        "",
        "## 自动读法",
        "",
        f"- 单看整体 RMSE，最好的是 `{best_rmse['variant_id']}`：RMSE={fmt(best_rmse['test_rmse_steer'])}。",
        f"- 按综合分数，最好的是 `{best_score['variant_id']}`：综合分数={fmt(best_score['screening_score'])}。",
        "- 如果某个版本 RMSE 下降但严重幅值不足率升高或大响应召回下降，说明它更像是在普通样本上拟合好了，不一定更符合极限工况。",
        "- 如果某类样本提升错侧率和幅值指标，但 RMSE 变大，可以考虑作为“极限姿态专门样本集”，而不是和普通样本混成一个回归任务。",
        "",
        "## 产物位置",
        "",
        f"- 汇总表：`{SUMMARY_PATH}`",
        f"- 排名表：`{RANKING_PATH}`",
        f"- 额外样本来源统计：`{EXTRA_SOURCE_PATH}`",
        f"- 每个版本指标和预测图：`{OUT_ROOT}`",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")


def append_notes(ranking: pd.DataFrame) -> None:
    best = ranking.iloc[0].to_dict()
    block = (
        "## 2026-05-19 v0.3 样本筛选策略连续对比\n\n"
        "- 当前阶段：车辆-only 样本筛选策略对比，不涉及连续风格、生理或脑电。\n"
        "- 本轮动作：连续比较低附着、横滚/姿态、弯道、快速转向、锚点后响应弱、自动候选标签等多种额外纳入范围。\n"
        f"- 当前综合排序第一：`{best['variant_id']}`，test RMSE={fmt(best['test_rmse_steer'])}，综合分数={fmt(best['screening_score'])}。\n"
        f"- 用户查看版报告：`{REPORT_DIR / 'stage03_v03_screening_sweep_user_summary_cn.md'}`。\n"
        f"- 输出目录：`{OUT_ROOT}`。\n"
    )
    for path in [NOTES_DIR / "PROJECT_STATUS_CN.md", NOTES_DIR / "TASK_QUEUE_CN.md", DAILY_LOG]:
        if path.exists():
            raw = path.read_text(encoding="utf-8")
            if "## 2026-05-19 v0.3 样本筛选策略连续对比" not in raw:
                path.write_text(raw.rstrip() + "\n\n" + block, encoding="utf-8")
    if ARTIFACT_INDEX.exists():
        raw = ARTIFACT_INDEX.read_text(encoding="utf-8")
        artifact = (
            "## v0.3 样本筛选策略连续对比\n\n"
            f"- 用户查看版报告：`{REPORT_DIR / 'stage03_v03_screening_sweep_user_summary_cn.md'}`\n"
            f"- 汇总表：`{SUMMARY_PATH}`\n"
            f"- 排名表：`{RANKING_PATH}`\n"
            f"- 输出目录：`{OUT_ROOT}`\n"
        )
        if "## v0.3 样本筛选策略连续对比" not in raw:
            ARTIFACT_INDEX.write_text(raw.rstrip() + "\n\n" + artifact, encoding="utf-8")


def append_server_note() -> None:
    if not SERVER_RUNS.exists():
        return
    raw = SERVER_RUNS.read_text(encoding="utf-8")
    block = (
        "## 2026-05-19 v0.3 样本筛选策略连续对比服务器记录\n\n"
        "- 服务器连接格式：`ssh -p 55060 root@connect.westc.seetacloud.com`，密码不记录。\n"
        "- 任务：运行 `stage03_v03_screening_sweep.py`，连续比较多种 v0.3 样本筛选策略。\n"
        "- 远程项目路径和日志路径在实际启动后补充。\n"
    )
    if "## 2026-05-19 v0.3 样本筛选策略连续对比服务器记录" not in raw:
        SERVER_RUNS.write_text(raw.rstrip() + "\n\n" + block, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    configure_inclusion_module()
    groups, group_counts = load_uid_groups()
    group_counts.to_csv(EXTRA_SOURCE_PATH, index=False, encoding="utf-8-sig")

    variants = make_variants(groups)
    sample_split, session_split = incl.load_reference_split()
    rows: list[dict[str, Any]] = []
    for variant in variants:
        existing = load_existing_result(str(variant["variant_id"]))
        if existing is not None:
            print(f"reuse {variant['variant_id']}", flush=True)
            result = existing
        else:
            print(
                f"run {variant['variant_id']} extra={len(set(variant.get('extra_episode_uids') or []))}",
                flush=True,
            )
            result = incl.run_variant(variant, sample_split, session_split)
        dataset_summary = load_dataset_summary(str(variant["variant_id"]))
        result["extra_episode_count"] = int(dataset_summary.get("extra_episode_count", 0))
        result["dropped_count"] = int(dataset_summary.get("dropped_count", 0))
        result["split_counts_json"] = json.dumps(dataset_summary.get("split_counts", {}), ensure_ascii=False)
        result["category_counts_json"] = json.dumps(dataset_summary.get("category_counts", {}), ensure_ascii=False)
        rows.append(result)

    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    ranking = score_rows(summary)
    ranking.to_csv(RANKING_PATH, index=False, encoding="utf-8-sig")
    write_report(ranking)
    append_notes(ranking)
    append_server_note()
    print(
        ranking[
            [
                "variant_id",
                "sample_count",
                "extra_episode_count",
                "test_rmse_steer",
                "test_wrong_side_rate_large",
                "test_severe_amp_under_rate_large",
                "test_large_response_recall",
                "screening_score",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
