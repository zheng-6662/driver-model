# 项目进度中枢

这份文件是后续默认查阅入口。先看这里的当前状态与优先级，再决定是否下钻到 `progress/daily/`、`progress/decision_log.md` 和 `progress/experiment_registry.md`。

## 当前状态

- 当前主线在 2026-04-21 切换为外部执行计划：
  - `D:/下载/codex_next_steps_plan.md`
- 旧 `04_project_logs/references/current-state.md` 继续保留为背景与历史锚点，但本轮不再决定下一步方向。
- 当前任务定义锁定为：
  - 四类场景是诱发协议，不拆成 scene-specific 模型
  - 主目标是 post-trigger steering response prediction
  - 当前 maintained line 仍保持 pooled training

## 30 秒白话版

- 这轮不再优先纠缠 `220918` 的复现差距，而是先把 active script 到底吃进了什么输入、任务定义怎么写、以及 Run A / Run B 为什么各强一头查清楚。
- A/B/F/G 第一批诊断已经完成，Batch 2 输入修复也已经 smoke 验证。
- 现在应该进入 D 输入分组消融，再进入 E bridge 训练，而不是继续停在工具搭建阶段。

## 当前优先级

1. 运行 D 输入分组消融 full matrix。
2. 每个 D run 结束后立刻 same-tool recalc。
3. 把每个 D run 结果写回：
   - `progress/daily/2026-04-21.md`
   - `progress/experiment_registry.md`
4. 只有 D 代码路径稳定后才进入 E bridge 训练矩阵。

## 本轮已完成的关键块

### Batch 1 诊断已完成

- 输入列审计：
  - `04_project_logs/reports/feature_input_audit_20260421/`
- trigger-to-onset lag 分析：
  - `04_project_logs/reports/trigger_response_lag_20260421/`
- checkpoint constrained-Pareto 诊断：
  - `04_project_logs/reports/checkpoint_selection_diagnosis_20260421/`
- spike 位置诊断：
  - `04_project_logs/reports/spike_position_diagnosis_20260421/`
- 论文答辩材料框架与本轮结论回填：
  - `04_project_logs/reports/thesis_defense_materials_20260421/`

### Batch 2 输入修复已完成

- active script：
  - `02_code/final_code/model/training/future_steer_event_rollpeak_transformer_v5_8_amp_tuned_fixed.py`
- recalc tool：
  - `02_code/tools/recalc_v58_checkpoint_with_current_metrics.py`
- 已实现：
  - fixed lane / speed source parsing
  - optional input flags
  - `input_qc` per-run artifacts
  - reversal bridge env interface
  - recalc compatibility for old and new runs

### Batch 3 工具已就绪

- `02_code/tools/run_input_group_ablation.py`
- `02_code/tools/summarize_input_group_ablation.py`

## 最近关键结论

| 日期 | 类型 | 一句话结论 | 入口 |
| --- | --- | --- | --- |
| 2026-04-21 | 输入审计 | `zx1|lateraldistance` 是真实主 lane 列名，旧 exact lookup 在 `91/91` 数据里打空 | [A 报告](feature_input_audit_20260421/feature_presence_report.md) |
| 2026-04-21 | 任务定义 | trigger-to-onset median lag 为 `0.105 s`，protocol split 未启用但不阻塞本轮 | [B 报告](trigger_response_lag_20260421/TASK_DEFINITION_AND_EVENT_LOGIC.md) |
| 2026-04-21 | checkpoint 诊断 | constrained Pareto 没有推翻 Run A / Run B 当前 structured keepers | [F 报告](checkpoint_selection_diagnosis_20260421/pareto_summary.json) |
| 2026-04-21 | 失败模式诊断 | B/C/D 的 spike 更像共享时间位点 artifact，而不是 first-reversal loss 单独制造 | [G 报告](spike_position_diagnosis_20260421/spike_position_summary.md) |
| 2026-04-21 | 代码闭环 | fixed input pipeline + `input_qc` + recalc 新接口已经 smoke 验证通过 | [2026-04-21 日志](progress/daily/2026-04-21.md) |
| 2026-04-20 | keeper | Run A 强在 response-structure，Run B 强在 overall / tail fit，Run C / D 不升主线 | [2026-04-20 日志](progress/daily/2026-04-20.md) |

## 最近实验 / 分析快照

| 日期 | 实验 / 分析 | 一句话结果 | 入口 |
| --- | --- | --- | --- |
| 2026-04-21 | feature input audit (`A`) | 91 个 vehicle 文件里 lane 与 speed 风险都命中 active script | [2026-04-21 日志](progress/daily/2026-04-21.md) |
| 2026-04-21 | trigger-to-onset lag (`B`) | 3737 个强事件里 median lag = `0.105 s` | [2026-04-21 日志](progress/daily/2026-04-21.md) |
| 2026-04-21 | constrained-Pareto diagnosis (`F`) | 诊断 selector 不改变现有 structured keeper | [2026-04-21 日志](progress/daily/2026-04-21.md) |
| 2026-04-21 | spike-position diagnosis (`G`) | B/C/D 都存在跨通道同步的固定 spike 带 | [2026-04-21 日志](progress/daily/2026-04-21.md) |
| 2026-04-21 | fixed input pipeline smoke (`C`) | 新 `input_qc` JSON 已写出，same-tool recalc smoke 已完成 | [2026-04-21 日志](progress/daily/2026-04-21.md) |

## 建议查阅路径

- 想知道现在应该继续做什么：先看本页和 `progress/decision_log.md`
- 想知道 2026-04-21 这轮到底做了哪些动作：看 `progress/daily/2026-04-21.md`
- 想知道每个 run / 命名分析的对照关系：看 `progress/experiment_registry.md`
- 想追溯旧复现审计背景：再看 `04_project_logs/references/current-state.md`

## 快速入口

- [2026-04-21 每日日志](F:\data_set_process\data_process\04_project_logs\reports\progress\daily\2026-04-21.md)
- [决策日志](F:\data_set_process\data_process\04_project_logs\reports\progress\decision_log.md)
- [实验登记表](F:\data_set_process\data_process\04_project_logs\reports\progress\experiment_registry.md)
- [当前状态 handoff](F:\data_set_process\data_process\04_project_logs\references\current-state.md)
- [历史总档](F:\data_set_process\data_process\04_project_logs\reports\project_progress_master.md)
## 2026-04-21 Final D Closure

- D input-ablation is complete under the corrected wrapper path.
- Final keeper split:
  - Run A remains the response-structure keeper.
  - `baseline_fixed_input` is now the maintained-line fit / tail keeper and replaces old Run B for the live bridge target.
- Final D reading:
  - `baseline_fixed_input`: keeper for fit/tail, `rmse_steer=0.5559`, `tail_rmse_steer=0.7171`, `late_peak_recall=0.6496`.
  - `plus_pedals`: late-peak trade-off branch, not a clean replacement.
  - `plus_lat_dyn`: timing-only trade-off, not promoted.
  - `plus_road_cond`: no-go.
  - `minus_z`: no-go for default policy; `USE_Z=1` stays default.
- Current next priority:
  - start E bridge matrix: `bridge_55_45`, `bridge_50_50`, `bridge_schedule_B_to_A`.
  - after each E run, run same-tool recalc and append both `daily/2026-04-21.md` and `experiment_registry.md`.
- Evidence:
  - `04_project_logs/reports/progress/daily/2026-04-21.md`
  - `04_project_logs/reports/progress/decision_log.md`
  - `04_project_logs/reports/input_group_ablation_20260421/input_ablation_comparison_table.csv`

## 2026-04-22 E Interim Update

- E bridge status:
  - `bridge_55_45`: completed, no keeper change.
  - `bridge_50_50`: completed, provisional fit/tail bridge candidate.
  - `bridge_schedule_B_to_A`: remaining.
- Interim live split:
  - Run A remains the response-structure keeper.
  - `bridge_50_50` currently leads the E fit/tail axis with `rmse_steer=0.5385`, `tail_rmse_steer=0.6846`.
  - `baseline_fixed_input` remains the post-D comparator and fallback fit/tail keeper.
- Current reading:
  - E can improve fit/tail beyond the D baseline.
  - E has not yet recovered Run A structure, so promotion is still provisional.
- Evidence:
  - `04_project_logs/reports/progress/daily/2026-04-21.md`
  - `04_project_logs/reports/progress/decision_log.md`
  - `04_project_logs/reports/bridge_training_20260421/bridge_manifest.json`

## 2026-04-22 E Final Closure

- E bridge matrix is complete.
- Final keeper split:
  - Run A remains the response-structure keeper.
  - `baseline_fixed_input` remains the fit/tail keeper after E guardrail review.
- Final E reading:
  - `bridge_50_50`: best fit/tail in E, `rmse_steer=0.5385`, `tail_rmse_steer=0.6846`, but `strong_pos.tail_amp_ratio_pred_over_gt=0.4987` and `strong_pos.tail_flatness_rate=0.7368` block promotion.
  - `bridge_schedule_B_to_A`: mixed branch with better late-peak and timing than `bridge_50_50`, but too much fit/tail regression to promote.
  - `bridge_55_45`: no-go.
- Current handoff state:
  - use Run A vs `baseline_fixed_input` as the live keeper split.
  - use `bridge_50_50` as fit/tail frontier evidence, not as a keeper.
  - use `bridge_schedule_B_to_A` as mixed bridge evidence, not as a keeper.
- Evidence:
  - `04_project_logs/reports/progress/daily/2026-04-21.md`
  - `04_project_logs/reports/progress/decision_log.md`
  - `04_project_logs/reports/bridge_training_20260421/bridge_manifest.json`
