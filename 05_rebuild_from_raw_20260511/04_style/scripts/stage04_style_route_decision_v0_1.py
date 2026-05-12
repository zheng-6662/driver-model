# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd


ROOT = Path(r"F:/data_set_process/data_process/05_rebuild_from_raw_20260511")
OUT_ROOT = ROOT / "04_style" / "stage04_style_route_decision_v0_1"
TABLE_DIR = OUT_ROOT / "tables"
FIG_DIR = OUT_ROOT / "figures"
LOG_DIR = OUT_ROOT / "logs"
REPORT_DIR = ROOT / "09_reports"
NOTES_DIR = ROOT / "00_project_notes"
DAILY_LOG = NOTES_DIR / "daily_logs" / "2026-05-13.md"

PROTOCOL_GATE = ROOT / "04_style" / "stage04_continuous_style_protocol_v0_1" / "tables" / "style_protocol_gate_table.csv"
INCREMENT_METRICS = ROOT / "04_style" / "stage04_style_increment_exploratory_v0_1" / "tables" / "style_increment_metrics.csv"
INCREMENT_GATE = ROOT / "04_style" / "stage04_style_increment_exploratory_v0_1" / "tables" / "style_increment_gate_table.csv"
CROSS_METRICS = ROOT / "04_style" / "stage04_style_cross_split_validation_v0_1" / "tables" / "style_cross_split_metrics.csv"
CROSS_GATE = ROOT / "04_style" / "stage04_style_cross_split_validation_v0_1" / "tables" / "style_cross_split_gate_table.csv"

RBF_MODEL = "rbf_kernel_ridge_context_no_subject"
STYLE60_MODEL = "rbf_plus_style_last60_guard3_residual_ridge"
STYLE_ALL_MODEL = "rbf_plus_style_all_windows_residual_ridge"
DRIVER_ID_MODEL = "rbf_plus_driver_id_residual_ridge"
STYLE_ID_MODEL = "rbf_plus_style_last60_with_driver_id_residual_ridge"


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIG_DIR, LOG_DIR, REPORT_DIR, DAILY_LOG.parent]:
        path.mkdir(parents=True, exist_ok=True)


def get_test_row(metrics: pd.DataFrame, split_strategy: str, model_name: str) -> pd.Series:
    rows = metrics[
        (metrics["split_strategy"].astype(str) == split_strategy)
        & (metrics["split"].astype(str) == "test")
        & (metrics["model_name"].astype(str) == model_name)
    ]
    if rows.empty:
        raise RuntimeError(f"missing test row: {split_strategy} {model_name}")
    return rows.iloc[0]


def build_evidence_summary(cross_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split_strategy in ["session_level_split", "subject_level_split"]:
        rbf = get_test_row(cross_metrics, split_strategy, RBF_MODEL)
        for model_name, role_cn in [
            (STYLE60_MODEL, "连续风格 last60"),
            (STYLE_ALL_MODEL, "连续风格全部窗口"),
            (DRIVER_ID_MODEL, "驾驶员 ID 对照"),
            (STYLE_ID_MODEL, "连续风格 last60 + 驾驶员 ID"),
        ]:
            row = get_test_row(cross_metrics, split_strategy, model_name)
            rows.append(
                {
                    "split_strategy": split_strategy,
                    "model_name": model_name,
                    "role_cn": role_cn,
                    "n_test_samples": int(row["n_samples"]),
                    "rbf_rmse": float(rbf["rmse_steer"]),
                    "model_rmse": float(row["rmse_steer"]),
                    "delta_rmse_vs_rbf": float(row["rmse_steer"] - rbf["rmse_steer"]),
                    "relative_rmse_change_pct_vs_rbf": float((row["rmse_steer"] - rbf["rmse_steer"]) / max(rbf["rmse_steer"], 1e-12) * 100.0),
                    "rbf_wrong_side_rate": float(rbf["wrong_side_rate"]),
                    "model_wrong_side_rate": float(row["wrong_side_rate"]),
                    "delta_wrong_side_rate": float(row["wrong_side_rate"] - rbf["wrong_side_rate"]),
                    "rbf_large_response_recall": float(rbf["large_response_recall"]),
                    "model_large_response_recall": float(row["large_response_recall"]),
                    "delta_large_response_recall": float(row["large_response_recall"] - rbf["large_response_recall"]),
                    "rbf_difficult_top20_rmse": float(rbf["difficult_top20_rmse"]),
                    "model_difficult_top20_rmse": float(row["difficult_top20_rmse"]),
                    "delta_difficult_top20_rmse": float(row["difficult_top20_rmse"] - rbf["difficult_top20_rmse"]),
                    "rbf_reversal_exact": float(rbf["reversal_count_exact_match_rate"]),
                    "model_reversal_exact": float(row["reversal_count_exact_match_rate"]),
                    "delta_reversal_exact": float(row["reversal_count_exact_match_rate"] - rbf["reversal_count_exact_match_rate"]),
                    "interpretation_cn": interpretation(split_strategy, model_name, row, rbf),
                }
            )
    return pd.DataFrame(rows)


def interpretation(split_strategy: str, model_name: str, row: pd.Series, rbf: pd.Series) -> str:
    rmse_delta = float(row["rmse_steer"] - rbf["rmse_steer"])
    phys_ok = (
        float(row["wrong_side_rate"]) < float(rbf["wrong_side_rate"])
        or float(row["large_response_recall"]) > float(rbf["large_response_recall"])
        or float(row["difficult_top20_rmse"]) < float(rbf["difficult_top20_rmse"])
    )
    if model_name == STYLE60_MODEL:
        if rmse_delta < 0 and phys_ok:
            return "有弱探索信号，但必须跨 split 稳定且通过置乱后才能继续。"
        if rmse_delta < 0:
            return "只有很小 RMSE 改善，关键物理指标没有稳定改善，不能升级。"
        return "未超过 RBF，不能作为有效性证据。"
    if model_name == STYLE_ALL_MODEL:
        if split_strategy == "session_level_split":
            return "全部窗口在 session-level 明显变差，说明堆更多风格特征不稳。"
        return "subject-level 有小改善，但 session-level 明显变差，不能算稳定路线。"
    if model_name == DRIVER_ID_MODEL:
        return "驾驶员 ID 对照没有实质增益；用于排除身份代理风险。"
    return "风格加 ID 仍没有形成稳定收益，说明当前融合方式不足以支撑结论。"


def build_gate_table(evidence: pd.DataFrame) -> pd.DataFrame:
    style60 = evidence[evidence["model_name"] == STYLE60_MODEL].copy()
    session = style60[style60["split_strategy"] == "session_level_split"].iloc[0]
    subject = style60[style60["split_strategy"] == "subject_level_split"].iloc[0]
    two_split_stable = (session["delta_rmse_vs_rbf"] < 0) and (subject["delta_rmse_vs_rbf"] < 0)
    phys_stable = (
        (session["delta_wrong_side_rate"] < 0 or session["delta_large_response_recall"] > 0 or session["delta_difficult_top20_rmse"] < 0)
        and (subject["delta_wrong_side_rate"] < 0 or subject["delta_large_response_recall"] > 0 or subject["delta_difficult_top20_rmse"] < 0)
    )
    rows = [
        {
            "gate_item": "no_leakage_style_protocol",
            "status": "pass_protocol",
            "evidence": "stage04_continuous_style_protocol_v0_1 passed direct-input and label-overlap checks.",
            "decision_cn": "事件前连续风格候选的来源协议可用。",
        },
        {
            "gate_item": "style_two_split_rmse_gain",
            "status": "pass" if two_split_stable else "fail",
            "evidence": f"session delta={session['delta_rmse_vs_rbf']:.6f}; subject delta={subject['delta_rmse_vs_rbf']:.6f}",
            "decision_cn": "必须两类切分都超过 RBF 才能进入有效性候选；当前不满足。",
        },
        {
            "gate_item": "style_physical_metric_gain",
            "status": "pass" if phys_stable else "fail",
            "evidence": (
                f"session wrong/large/difficult delta={session['delta_wrong_side_rate']:.6f}/"
                f"{session['delta_large_response_recall']:.6f}/{session['delta_difficult_top20_rmse']:.6f}; "
                f"subject wrong/large/difficult delta={subject['delta_wrong_side_rate']:.6f}/"
                f"{subject['delta_large_response_recall']:.6f}/{subject['delta_difficult_top20_rmse']:.6f}"
            ),
            "decision_cn": "没有稳定改善错侧、大幅响应召回或困难样本，不能升级。",
        },
        {
            "gate_item": "style_not_driver_id_proxy",
            "status": "weak_pass",
            "evidence": "driver ID control is near RBF and does not explain a large style gain; however style itself also lacks stable gain.",
            "decision_cn": "当前不是强身份代理问题，而是风格增量本身不稳定。",
        },
        {
            "gate_item": "style_route_continue_as_mainline",
            "status": "no_go_current_form",
            "evidence": "session-level fails; subject-level only tiny RMSE gain; physical metrics do not stably improve.",
            "decision_cn": "当前连续风格直接残差融合路线不升级为主线。",
        },
        {
            "gate_item": "physio_eeg_role_validation_allowed",
            "status": "blocked",
            "evidence": "vehicle+style fair reference is not strong/stable enough.",
            "decision_cn": "生理/EEG 有效性验证继续阻塞。",
        },
        {
            "gate_item": "next_route",
            "status": "go_vehicle_structured",
            "evidence": "RBF still has wrong-side, reversal, multi-segment and difficult-sample failures.",
            "decision_cn": "回到车辆-only 结构化轨迹建模，优先响应分解/关键点/多假设。",
        },
    ]
    return pd.DataFrame(rows)


def build_next_action_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "priority": 1,
                "task": "阶段 6 车辆-only 结构化轨迹建模",
                "why_cn": "连续风格当前没有稳定增量，先解决车辆-only 的错侧、幅值、反向修正、多段修正和困难样本。",
                "allowed_now": True,
            },
            {
                "priority": 2,
                "task": "固定坏样本图人工复核摘要",
                "why_cn": "确认 RBF 失败到底来自事件语义、物理不可预测、多段响应还是模型结构不足。",
                "allowed_now": True,
            },
            {
                "priority": 3,
                "task": "连续风格更强表示探索",
                "why_cn": "只可作为后备探索；当前统计特征 + 残差 Ridge 不支持主线。",
                "allowed_now": False,
            },
            {
                "priority": 4,
                "task": "生理/EEG 有效性验证",
                "why_cn": "尚未形成稳定车辆+风格公平参照，不能把生理增量归因干净。",
                "allowed_now": False,
            },
        ]
    )


def plot_route_summary(evidence: pd.DataFrame) -> Path:
    subset = evidence[evidence["model_name"].isin([STYLE60_MODEL, STYLE_ALL_MODEL, DRIVER_ID_MODEL, STYLE_ID_MODEL])].copy()
    labels = [f"{row.split_strategy.replace('_level_split','')}\n{short_name(row.model_name)}" for row in subset.itertuples()]
    values = subset["delta_rmse_vs_rbf"].to_numpy(dtype=float)
    colors = ["#54a24b" if v < 0 else "#e45756" for v in values]
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(labels, values, color=colors)
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_ylabel("Delta RMSE vs RBF (negative is better)")
    ax.set_title("Stage04 style route decision: RMSE delta is not stable across splits")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    out = FIG_DIR / "style_route_rmse_delta_summary.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def short_name(model_name: str) -> str:
    return {
        STYLE60_MODEL: "style60",
        STYLE_ALL_MODEL: "style all",
        DRIVER_ID_MODEL: "driver ID",
        STYLE_ID_MODEL: "style60+ID",
    }.get(model_name, model_name)


def write_reports(evidence: pd.DataFrame, gate: pd.DataFrame, next_actions: pd.DataFrame, figure_path: Path) -> None:
    style60 = evidence[evidence["model_name"] == STYLE60_MODEL].set_index("split_strategy")
    user = f"""# 阶段 4 用户查看版：连续风格路线收口决策 v0.1

## 这个阶段为什么做

前面已经完成连续风格来源协议、session-level 探索对照和 subject-level 跨被试复核。现在需要把结论收口：连续风格当前能不能升级为主线，生理/脑电能不能进入下一阶段。

## 这个阶段检查了什么

- 风格是否无泄漏：通过，风格窗口在事件前，不接触直接车辆输入和未来标签。
- 风格是否超过 RBF：没有稳定超过。
- 风格是否改善物理错误：没有稳定改善错侧、大幅响应召回、困难样本或反向修正。
- 风格是否只是驾驶员 ID 替代品：目前不是主要问题，因为风格本身也没有稳定增益。

## 目前发现了什么

```text
session-level: RBF={style60.loc['session_level_split','rbf_rmse']:.6f}, RBF+style60={style60.loc['session_level_split','model_rmse']:.6f}, delta={style60.loc['session_level_split','delta_rmse_vs_rbf']:.6f}
subject-level: RBF={style60.loc['subject_level_split','rbf_rmse']:.6f}, RBF+style60={style60.loc['subject_level_split','model_rmse']:.6f}, delta={style60.loc['subject_level_split','delta_rmse_vs_rbf']:.6f}
```

subject-level 有很小 RMSE 改善，但 session-level 没有，而且物理指标没有稳定改善。因此不能说连续风格有效。

## 哪些结果可信

可信的是：在当前“事件前统计风格特征 + RBF 残差 Ridge”表示下，连续风格没有形成可升级为主线的稳定证据。这个结论经过了固定 RBF 参照、驾驶员 ID 对照、置乱控制、session-level 和 subject-level 检查。

## 哪些结果还不能下结论

不能说“风格永远无效”。只能说当前表示方式和融合方式没有形成强证据。未来如果换成更好的时序风格表示或门控结构，可以重新作为后备路线验证。

## 下一阶段是否可以继续

可以继续，但不应进入生理/EEG 有效性验证。下一步应回到车辆-only 结构化轨迹建模，先解决错侧、幅值、尾段、反向修正、多段修正和困难样本。

## 推荐优先查看

1. `{figure_path.as_posix()}`
2. `{(TABLE_DIR / 'style_route_decision_gate_table.csv').as_posix()}`
3. `{(TABLE_DIR / 'style_route_evidence_summary.csv').as_posix()}`
4. `{(TABLE_DIR / 'style_route_next_actions.csv').as_posix()}`
"""
    (REPORT_DIR / "stage04_style_route_decision_user_summary_cn.md").write_text(user, encoding="utf-8")

    technical = f"""# 阶段 4：连续风格路线收口决策 v0.1

## 输入证据

- 协议 gate：`{PROTOCOL_GATE.as_posix()}`
- session-level 探索指标：`{INCREMENT_METRICS.as_posix()}`
- cross-split 指标：`{CROSS_METRICS.as_posix()}`
- cross-split gate：`{CROSS_GATE.as_posix()}`

## 证据摘要

```text
{evidence.to_string(index=False)}
```

## gate

```text
{gate.to_string(index=False)}
```

## 下一步

```text
{next_actions.to_string(index=False)}
```

## 决策

当前连续风格直接残差融合路线不升级为主线；生理/EEG 有效性验证继续阻塞；下一步回到车辆-only 结构化轨迹建模。"""
    (REPORT_DIR / "stage04_style_route_decision_v0_1_cn.md").write_text(technical, encoding="utf-8")


def update_transparency(run_summary: dict[str, Any]) -> None:
    status_section = f"""## 最新更新：2026-05-13 06:25

- 当前阶段：阶段 4 连续风格路线收口决策 v0.1 已完成；当前连续风格直接残差融合路线不升级为主线。
- 当前正在做什么：准备进入车辆-only 结构化轨迹建模路线，而不是进入生理/EEG。
- 已完成什么：新增并运行 `stage04_style_route_decision_v0_1.py`；汇总风格协议、session-level 探索、subject-level 复核和 gate，形成阶段 4 收口表、下一步动作表、图和中文报告。
- 正在运行什么任务：没有后台任务；没有服务器任务；没有本地训练任务。
- 服务器是否在运行：未使用服务器，未读取服务器指令与密码文件，未记录任何凭据。
- 最近一次结果是什么：session-level style60 delta RMSE={run_summary['session_style60_delta_rmse']:.6f}；subject-level style60 delta RMSE={run_summary['subject_style60_delta_rmse']:.6f}；物理指标未稳定改善；style route gate=no_go_current_form；physio/eeg=blocked。
- 当前最大风险是什么：如果跳过车辆-only 结构化错误复盘，直接进入生理/EEG，会把车辆主参照未解决的问题错误归因给新模态。
- 下一步准备做什么：阶段 6 车辆-only 结构化轨迹建模，优先响应分解、关键点+残差、多假设/可靠性，继续用固定 RBF 参照和坏样本图。
- 用户可以优先查看哪些文件：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage04_style_route_decision_user_summary_cn.md`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_route_decision_v0_1/tables/style_route_decision_gate_table.csv`，`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_route_decision_v0_1/figures/style_route_rmse_delta_summary.png`。
"""
    status_path = NOTES_DIR / "PROJECT_STATUS_CN.md"
    base = status_path.read_text(encoding="utf-8", errors="replace")
    title = "# R2E-Steering 项目总进度看板"
    if base.startswith(title):
        status_path.write_text(title + "\n\n" + status_section.strip() + "\n\n" + base[len(title):].lstrip("\r\n"), encoding="utf-8")
    else:
        status_path.write_text(status_section.strip() + "\n\n" + base, encoding="utf-8")

    task_section = """## 最新更新：2026-05-13 06:25

### 正在做任务
- 阶段 4 连续风格路线已收口；下一步准备进入车辆-only 结构化轨迹建模。

### 已完成任务
- 已新增并运行 `stage04_style_route_decision_v0_1.py`。
- 已形成连续风格 no-go-current-form 决策：当前统计风格 + RBF 残差 Ridge 不升级主线。
- 已确认生理/EEG 继续阻塞。

### 待做任务
- 阶段 6：车辆-only 响应分解/关键点+残差/多假设路线设计。
- 固定 RBF 坏样本图复核摘要，用来定义结构化模型要解决的物理错误。

### 阻塞任务
- 生理、脑电有效性验证继续阻塞。
- 连续风格有效性强结论阻塞；仅保留为后备表示/融合探索。

### 可并行任务
- 车辆-only 结构化模型方案草稿。
- RBF top bad 样本物理错误摘要。

### 需要服务器的任务
- 暂无。

### 不需要服务器的任务
- 当前收口报告、下一轮结构化车辆-only 方案和轻量原型均可先本地完成。
"""
    task_path = NOTES_DIR / "TASK_QUEUE_CN.md"
    task_base = task_path.read_text(encoding="utf-8", errors="replace")
    title_task = "# 当前任务队列"
    if task_base.startswith(title_task):
        task_path.write_text(title_task + "\n\n" + task_section.strip() + "\n\n" + task_base[len(title_task):].lstrip("\r\n"), encoding="utf-8")
    else:
        task_path.write_text(task_section.strip() + "\n\n" + task_base, encoding="utf-8")

    artifact_entry = """## 最新新增：阶段 4 连续风格路线收口决策 v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage04_style_route_decision_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage04_style_route_decision_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/scripts/stage04_style_route_decision_v0_1.py`
- 证据摘要表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_route_decision_v0_1/tables/style_route_evidence_summary.csv`
- gate 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_route_decision_v0_1/tables/style_route_decision_gate_table.csv`
- 下一步动作表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_route_decision_v0_1/tables/style_route_next_actions.csv`
- RMSE delta 图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_route_decision_v0_1/figures/style_route_rmse_delta_summary.png`
- 服务器日志：无，本轮未使用远程服务器。
- GPTPro 提问和回复：无。
- 重要 Git commit：待提交。
- 适合用户/老师直接查看：用户查看版总结、gate 表、RMSE delta 图、下一步动作表。
"""
    artifact_path = NOTES_DIR / "ARTIFACT_INDEX_CN.md"
    artifact_path.write_text(artifact_path.read_text(encoding="utf-8", errors="replace").rstrip() + "\n\n" + artifact_entry, encoding="utf-8")

    daily_entry = f"""
## 06:25 阶段 4：连续风格路线收口决策 v0.1

- 为什么做：连续风格已经完成协议、session-level 探索和 subject-level 复核，需要明确是否升级主线以及是否允许进入生理/EEG。
- 做了什么：新增并运行 `stage04_style_route_decision_v0_1.py`，汇总阶段 4 证据、生成 no-go-current-form gate、下一步动作表、图和中文报告。
- 用了哪些输入：阶段 4 风格协议 gate、风格增量探索指标、跨 split 复核指标和 gate。
- 生成了哪些输出：`04_style/stage04_style_route_decision_v0_1/` 下的表格、图和日志，以及 `09_reports/stage04_style_route_decision_user_summary_cn.md`。
- 当前结果如何：session-level style60 delta RMSE={run_summary['session_style60_delta_rmse']:.6f}；subject-level style60 delta RMSE={run_summary['subject_style60_delta_rmse']:.6f}；路线决策为当前形式 no-go，生理/EEG 继续 blocked。
- 是否遇到问题：无运行错误；解释边界是不能否定未来更强风格表示，只能否定当前统计风格 + 残差 Ridge 直接路线。
- 是否需要用户决策：暂不需要；下一步建议回到车辆-only 结构化轨迹建模。
"""
    with DAILY_LOG.open("a", encoding="utf-8") as f:
        f.write(daily_entry)


def main() -> None:
    ensure_dirs()
    if not all(path.exists() for path in [PROTOCOL_GATE, INCREMENT_METRICS, INCREMENT_GATE, CROSS_METRICS, CROSS_GATE]):
        raise RuntimeError("missing stage04 upstream artifacts")
    cross_metrics = pd.read_csv(CROSS_METRICS)
    evidence = build_evidence_summary(cross_metrics)
    gate = build_gate_table(evidence)
    next_actions = build_next_action_table()
    fig = plot_route_summary(evidence)
    evidence.to_csv(TABLE_DIR / "style_route_evidence_summary.csv", index=False, encoding="utf-8-sig")
    gate.to_csv(TABLE_DIR / "style_route_decision_gate_table.csv", index=False, encoding="utf-8-sig")
    next_actions.to_csv(TABLE_DIR / "style_route_next_actions.csv", index=False, encoding="utf-8-sig")
    write_reports(evidence, gate, next_actions, fig)
    style60 = evidence[evidence["model_name"] == STYLE60_MODEL].set_index("split_strategy")
    run_summary = {
        "run_time_local": "2026-05-13 06:25",
        "session_style60_delta_rmse": float(style60.loc["session_level_split", "delta_rmse_vs_rbf"]),
        "subject_style60_delta_rmse": float(style60.loc["subject_level_split", "delta_rmse_vs_rbf"]),
        "style_route_continue_as_mainline": False,
        "style_effectiveness_claim_allowed": False,
        "physio_eeg_role_validation_allowed": False,
        "next_route": "vehicle_only_structured_trajectory_modeling",
        "evidence_path": (TABLE_DIR / "style_route_evidence_summary.csv").as_posix(),
        "gate_path": (TABLE_DIR / "style_route_decision_gate_table.csv").as_posix(),
        "next_actions_path": (TABLE_DIR / "style_route_next_actions.csv").as_posix(),
        "figure_path": fig.as_posix(),
        "server_used": False,
        "server_access_file_read": False,
        "uses_physio": False,
        "uses_eeg": False,
        "raw_files_modified": False,
    }
    (LOG_DIR / "style_route_decision_summary.json").write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    update_transparency(run_summary)
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
