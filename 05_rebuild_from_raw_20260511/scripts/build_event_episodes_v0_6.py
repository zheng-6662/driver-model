# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
REBUILD_ROOT = SCRIPT_PATH.parents[1]
if str(REBUILD_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REBUILD_ROOT / "src"))

from event_episode_detection.classification import C, N, P1, P2, U, X, classify_episode
from event_episode_detection.config import load_config, resolve_paths
from event_episode_detection.context_matching import ContextMatcher
from event_episode_detection.correction import score_correction
from event_episode_detection.plot_review import configure_fonts, plot_episode, select_review_rows
from event_episode_detection.signals import load_vehicle_csv
from event_episode_detection.steering_onset import (
    build_trigger_no_effect_rows,
    detect_steering_episodes,
    infer_subject_session,
)
from event_episode_detection.vehicle_response import score_vehicle_response


CLASS_FILES = {
    P1: "primary_positive_episodes_P1_v0_6.csv",
    P2: "secondary_episodes_P2_v0_6.csv",
    C: "context_control_C_v0_6.csv",
    N: "trigger_no_effect_N_v0_6.csv",
    U: "manual_review_U_v0_6.csv",
    X: "excluded_X_v0_6.csv",
}


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def find_vehicle_files(project_root: Path, raw_glob: str) -> list[Path]:
    candidates = list(project_root.glob(raw_glob))
    candidates = [p for p in candidates if p.name.startswith("Entity_Recording_") and p.name.endswith("_vehicle.csv")]
    if candidates:
        return sorted(candidates)
    return sorted(
        p
        for p in (project_root / "01_datasets").glob("**/*vehicle*.csv")
        if p.name.startswith("Entity_Recording_") and p.name.endswith("_vehicle.csv")
    )


def get_vehicle_for_row(row: pd.Series, project_root: Path, config: dict[str, Any], cache: dict[str, pd.DataFrame]) -> pd.DataFrame:
    path_text = str(row.get("vehicle_raw_absolute_path", ""))
    if not path_text or path_text == "nan":
        rel = str(row.get("vehicle_raw_relative_path", ""))
        if rel and rel != "nan":
            path_text = str(project_root / "01_datasets" / "数据预处理" / rel)
    if not path_text:
        return pd.DataFrame()
    if path_text in cache:
        return cache[path_text]
    df, _ = load_vehicle_csv(Path(path_text), config)
    cache[path_text] = df
    return df


def class_cn(cls: str) -> str:
    return {
        P1: "核心正样本：强方向盘动作 + 车辆动态响应 + 回正/反打",
        P2: "次级样本：强方向盘动作，但车辆动态响应较弱",
        C: "正常驾驶/正常弯道/正常变道对照",
        N: "触发无效或无方向盘快速响应对照",
        U: "语义不清，需要人工复核",
        X: "排除样本",
    }.get(cls, cls)


def build_summary(
    all_df: pd.DataFrame,
    processing_df: pd.DataFrame,
    matcher: ContextMatcher,
    output_dir: Path,
    review_index: pd.DataFrame,
) -> None:
    total = len(all_df)
    counts = all_df["episode_class"].value_counts(dropna=False)
    class_lines = "\n".join([f"- `{cls}`：{int(n)} 个，{class_cn(str(cls))}" for cls, n in counts.items()])
    module_ct = pd.crosstab(all_df.get("road_context", pd.Series(dtype=str)).fillna("unknown"), all_df["episode_class"], dropna=False)
    subject_ct = pd.crosstab(all_df.get("subject_id", pd.Series(dtype=str)).fillna("unknown"), all_df["episode_class"], dropna=False)
    p1 = all_df[all_df["episode_class"].eq(P1)].copy()
    p2 = all_df[all_df["episode_class"].eq(P2)].copy()
    steering_only = all_df[all_df["row_source"].eq("steering_onset_scan")].copy()

    def delta_stats_for(df: pd.DataFrame, col: str) -> str:
        values = pd.to_numeric(df.get(col), errors="coerce").dropna()
        if values.empty:
            return "无可计算样本"
        return (
            f"n={len(values)}, 中位数={values.median():.3f}s, "
            f"p25={values.quantile(0.25):.3f}s, p75={values.quantile(0.75):.3f}s, "
            f"|差值|<=2s 比例={(values.abs() <= 2.0).mean() * 100:.1f}%"
        )

    def delta_stats(col: str) -> str:
        return delta_stats_for(all_df, col)

    trigger_no_episode = len(all_df[all_df["episode_class"].eq(N)])
    trigger_total = matcher.trigger_count()
    trigger_with_episode = max(trigger_total - trigger_no_episode, 0)
    report = output_dir / "event_episode_summary_v0_6.md"
    report.write_text(
        f"""# 方向盘动作 episode 样本重建 v0.6 汇总报告

生成时间：{now()}

## 本轮任务边界

本轮没有训练预测模型。主锚点不再使用 `.aed` 触发点、道路入口或旧流程锚点，而是从原始车辆 CSV 中扫描方向盘角速度显著升高的启动时刻 `t_steer_onset`。`.aed`、cfg/道路模块、旧锚点和 v0.5 候选点只作为上下文解释字段贴回 episode。

## 总体数量

- 总输出行数：{total}
- 自动检测到的方向盘动作 episode：{int((all_df['row_source'] == 'steering_onset_scan').sum())}
- 附加的触发无效/无快速方向盘响应对照：{trigger_no_episode}
- 已知场景触发总数：{trigger_total}
- 原场景触发附近 2 秒内存在方向盘 episode 的数量：{trigger_with_episode}
- P1 核心正样本数量：{len(p1)}
- P2 次级样本数量：{len(p2)}
- 待人工复核样本数量：{int((all_df['episode_class'] == U).sum())}

## 分类数量

{class_lines}

## 分场景数量

```text
{module_ct.to_string()}
```

## 分被试数量

```text
{subject_ct.to_string()}
```

## P1 主要来自哪些场景

```text
{p1.get('road_context', pd.Series(dtype=str)).fillna('unknown').value_counts().to_string() if not p1.empty else '无 P1 样本'}
```

## P2 主要来自哪些场景

```text
{p2.get('road_context', pd.Series(dtype=str)).fillna('unknown').value_counts().to_string() if not p2.empty else '无 P2 样本'}
```

## 自动方向盘 episode 与外部上下文的时间差

以下只统计 `row_source = steering_onset_scan`，不把 N 类触发无效行混入：

- 与最近 `.aed` 触发点：{delta_stats_for(steering_only, 'delta_to_nearest_aed_trigger')}
- 与最近旧流程锚点：{delta_stats_for(steering_only, 'delta_to_nearest_old_anchor')}
- 与最近 v0.5 候选点：{delta_stats_for(steering_only, 'delta_to_nearest_v05_candidate')}

## 与外部上下文的时间差

以下统计全部输出行，N 类触发无效行会把 `.aed` 差值固定为 0，因此主要用于检查表格字段完整性：

- 与最近 `.aed` 触发点：{delta_stats('delta_to_nearest_aed_trigger')}
- 与最近旧流程锚点：{delta_stats('delta_to_nearest_old_anchor')}
- 与最近 v0.5 候选点：{delta_stats('delta_to_nearest_v05_candidate')}

## 当前规则的主要局限

1. `t_steer_onset` 是从方向盘角速度显著升高处自动检测的，虽然没有用未来峰值定义起点，但仍然需要人工复核阈值是否过敏或漏检。
2. P1/P2/C/U/X 是弱标签，不是人工真值。它们用于筛选和复核，不应直接写成最终论文结论。
3. 正常弯道和平滑变道的区分仍然依赖道路上下文和启发式阈值，后续需要看复核图修正规则。
4. `N_trigger_no_effect_or_no_response` 是“触发附近没有快速方向盘 episode”的对照，不代表该触发完全没有纵向制动或轻微避让。
5. 横向偏移坐标跳变只做了局部步长检测，仍可能需要结合道路模块切换人工确认。

## 建议人工复核

- 先看 P1 前 30 张，确认是否真的体现“猛打方向 + 车辆动态增强 + 回正/反打”。
- 再看 U 类边界样本，决定是放宽/收紧车辆动态阈值，还是增加正常弯道判定规则。
- N 类重点看附近是否存在制动或轻微避让；如果有，应保留为纵向/弱响应对照，而不是强转向正样本。

## 复核图

- 复核图目录：`{output_dir / 'review_figures'}`
- 复核图索引：`{output_dir / 'review_figure_index_v0_6.csv'}`
- 实际生成复核图数量：{len(review_index)}

## 处理日志

- 每条车辆记录处理日志：`{output_dir / 'record_processing_log_v0_6.csv'}`
- 文本运行日志：`{output_dir / 'build_event_episodes_v0_6.log'}`
""",
        encoding="utf-8",
    )

    write_csv(module_ct.reset_index(), output_dir / "episode_class_by_scene_v0_6.csv")
    write_csv(subject_ct.reset_index(), output_dir / "episode_class_by_subject_v0_6.csv")


def update_project_notes(paths, output_dir: Path, all_df: pd.DataFrame) -> None:
    notes = paths.rebuild_root / "00_project_notes"
    notes.mkdir(parents=True, exist_ok=True)
    daily = notes / "daily_logs" / "2026-05-14.md"
    daily.parent.mkdir(parents=True, exist_ok=True)
    if not daily.exists():
        daily.write_text("# 2026-05-14 执行日志\n\n", encoding="utf-8")
    counts = all_df["episode_class"].value_counts(dropna=False).to_dict()
    with daily.open("a", encoding="utf-8") as f:
        f.write(
            f"""## 方向盘动作 episode 样本重建 v0.6

- 为什么做：把事件主锚点改为驾驶员方向盘快速启动 `t_steer_onset`，不再把 `.aed` 或道路入口当作事件真值。
- 做了什么：扫描原始车辆 CSV，检测方向盘快速动作 episode，计算车辆动态响应、回正/反打纠正、道路/触发/旧锚点上下文，并输出 P1/P2/C/N/U/X 弱标签。
- 输出目录：`{output_dir}`
- 分类数量：`{counts}`
- 本轮没有训练模型。

"""
        )

    status = notes / "PROJECT_STATUS_CN.md"
    old = status.read_text(encoding="utf-8", errors="ignore") if status.exists() else "# 项目状态\n\n"
    status_entry = f"""# 项目状态更新：方向盘动作 episode 样本重建 v0.6

更新时间：2026-05-14

当前阶段：旧流程事件样本重建继续推进。

当前完成：以方向盘角速度启动为主锚点的 episode 自动挖掘，已生成分类样本表、复核图和汇总报告。

最近一次结果：总输出 {len(all_df)} 行；P1={counts.get(P1, 0)}，P2={counts.get(P2, 0)}，C={counts.get(C, 0)}，N={counts.get(N, 0)}，U={counts.get(U, 0)}，X={counts.get(X, 0)}。

用户优先查看：

- `{output_dir / 'event_episode_summary_v0_6.md'}`
- `{output_dir / 'event_episodes_all_v0_6.csv'}`
- `{output_dir / 'primary_positive_episodes_P1_v0_6.csv'}`
- `{output_dir / 'review_figures'}`
- `{output_dir / 'review_figure_index_v0_6.csv'}`

下一步建议：先人工复核 P1、U 和 C 类代表图，再决定是否把 P1/P2 作为“早期方向盘动作预测剩余轨迹”的训练样本。

---

"""
    status.write_text(status_entry + old, encoding="utf-8")

    artifact = notes / "ARTIFACT_INDEX_CN.md"
    old_artifact = artifact.read_text(encoding="utf-8", errors="ignore") if artifact.exists() else "# 产物索引\n\n"
    artifact_entry = f"""## 2026-05-14 方向盘动作 episode 样本重建 v0.6

- 汇总报告：`{output_dir / 'event_episode_summary_v0_6.md'}`
- 主 episode 表：`{output_dir / 'event_episodes_all_v0_6.csv'}`
- P1 表：`{output_dir / 'primary_positive_episodes_P1_v0_6.csv'}`
- P2 表：`{output_dir / 'secondary_episodes_P2_v0_6.csv'}`
- C 表：`{output_dir / 'context_control_C_v0_6.csv'}`
- N 表：`{output_dir / 'trigger_no_effect_N_v0_6.csv'}`
- U 表：`{output_dir / 'manual_review_U_v0_6.csv'}`
- X 表：`{output_dir / 'excluded_X_v0_6.csv'}`
- 复核图目录：`{output_dir / 'review_figures'}`
- 复核图索引：`{output_dir / 'review_figure_index_v0_6.csv'}`
- 日志：`{output_dir / 'build_event_episodes_v0_6.log'}`

"""
    artifact.write_text(artifact_entry + old_artifact, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to event_episode_v0_6.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = resolve_paths(config)
    output_dir = paths.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    review_dir = output_dir / "review_figures"
    review_dir.mkdir(parents=True, exist_ok=True)
    for old_png in review_dir.glob("**/*.png"):
        old_png.unlink()
    log_path = output_dir / "build_event_episodes_v0_6.log"

    configure_fonts()
    matcher = ContextMatcher(paths)
    vehicle_files = find_vehicle_files(paths.project_root, paths.raw_vehicle_glob)

    all_rows: list[dict[str, Any]] = []
    process_rows: list[dict[str, Any]] = []
    vehicle_cache: dict[str, pd.DataFrame] = {}
    log_lines = [f"[{now()}] start build_event_episodes_v0_6", f"vehicle_files={len(vehicle_files)}"]

    for idx, vehicle_path in enumerate(vehicle_files):
        subject, session, rel_hint = infer_subject_session(vehicle_path)
        vehicle, meta = load_vehicle_csv(vehicle_path, config)
        rel_path = str(vehicle_path.relative_to(paths.project_root / "01_datasets" / "数据预处理")) if str(vehicle_path).startswith(str(paths.project_root / "01_datasets" / "数据预处理")) else rel_hint
        meta.update(
            {
                "subject": subject,
                "session_stamp": session,
                "vehicle_path": str(vehicle_path),
                "vehicle_raw_relative_path": rel_path,
            }
        )
        episodes = detect_steering_episodes(vehicle, meta, config)
        process_rows.append(
            {
                "vehicle_path": str(vehicle_path),
                "subject": subject,
                "session_stamp": session,
                "read_status": meta.get("read_status", ""),
                "row_count": meta.get("row_count", 0),
                "duration_s": meta.get("duration_s", np.nan),
                "median_dt_s": meta.get("median_dt_s", np.nan),
                "timestamp_gap_count_gt_0_1s": meta.get("timestamp_gap_count_gt_0_1s", 0),
                "identified_fields_json": json.dumps(meta.get("identified_fields", {}), ensure_ascii=False),
                "detected_episode_count": len(episodes),
                "warnings": "; ".join(meta.get("warnings", [])),
            }
        )
        if vehicle.empty:
            continue
        vehicle_cache[str(vehicle_path)] = vehicle
        for ep in episodes:
            vf = score_vehicle_response(vehicle, ep, config)
            cf = score_correction(vehicle, ep, vf, config)
            row = {**ep, **vf, **cf}
            row = matcher.append_context(row)
            row = classify_episode(row, config)
            all_rows.append(row)
        if (idx + 1) % 10 == 0:
            log_lines.append(f"[{now()}] processed {idx + 1}/{len(vehicle_files)} records")

    steering_df = pd.DataFrame(all_rows)
    trigger_rows = build_trigger_no_effect_rows(matcher.scene_triggers, steering_df, config)
    if not trigger_rows.empty:
        trigger_context_rows = []
        for _, r in trigger_rows.iterrows():
            row = matcher.append_context(r.to_dict())
            rel = str(row.get("vehicle_raw_relative_path", ""))
            if rel:
                row["vehicle_raw_absolute_path"] = str(paths.project_root / "01_datasets" / "数据预处理" / rel)
            trigger_context_rows.append(row)
        trigger_rows = pd.DataFrame(trigger_context_rows)
    all_df = pd.concat([steering_df, trigger_rows], ignore_index=True, sort=False) if not trigger_rows.empty else steering_df
    if all_df.empty:
        raise RuntimeError("No episodes were generated.")

    all_df["episode_class_cn"] = all_df["episode_class"].map(class_cn)
    write_csv(all_df, output_dir / "event_episodes_all_v0_6.csv")
    for cls, filename in CLASS_FILES.items():
        write_csv(all_df[all_df["episode_class"].eq(cls)].copy(), output_dir / filename)
    processing_df = pd.DataFrame(process_rows)
    write_csv(processing_df, output_dir / "record_processing_log_v0_6.csv")

    selected = select_review_rows(all_df, config)
    review_rows: list[dict[str, Any]] = []
    plot_cache: dict[str, pd.DataFrame] = {}
    for _, row in selected.iterrows():
        vehicle = get_vehicle_for_row(row, paths.project_root, config, plot_cache)
        cls = str(row.get("episode_class", "unknown"))
        episode_id = str(row.get("episode_id", "episode")).replace(":", "_").replace("\\", "_").replace("/", "_")
        out_path = review_dir / cls / f"{episode_id}.png"
        ok = plot_episode(vehicle, row, out_path, config)
        if ok:
            review_rows.append(
                {
                    "episode_id": row.get("episode_id", ""),
                    "episode_class": cls,
                    "episode_class_cn": row.get("episode_class_cn", ""),
                    "subject_id": row.get("subject_id", ""),
                    "session_stamp": row.get("session_stamp", ""),
                    "road_context": row.get("road_context", ""),
                    "figure_path": str(out_path),
                }
            )
    review_index = pd.DataFrame(review_rows)
    write_csv(review_index, output_dir / "review_figure_index_v0_6.csv")

    build_summary(all_df, processing_df, matcher, output_dir, review_index)
    update_project_notes(paths, output_dir, all_df)

    log_lines.append(f"[{now()}] finished")
    log_lines.append(f"total_rows={len(all_df)}")
    log_lines.append(f"class_counts={dict(Counter(all_df['episode_class'].tolist()))}")
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(f"done total_rows={len(all_df)}")
    print(dict(Counter(all_df["episode_class"].tolist())))
    print(f"summary={output_dir / 'event_episode_summary_v0_6.md'}")


if __name__ == "__main__":
    main()
