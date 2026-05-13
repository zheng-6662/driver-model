# 事件锚点筛选与样本重建目标完成审计

生成时间：2026-05-13

## 1. 目标复述

本阶段目标是完成事件锚点筛选与样本重建：

1. 基于 v0.4/v0.5/GPTPro 审查意见，生成清晰可选的 v0.6 事件样本清单。
2. 明确哪些事件可训练、哪些需复核、哪些仅作响应确认、哪些暂缓或排除。
3. 使用重筛样本进行纯车辆/道路预测对照。
4. 目标判断标准是：如果重筛样本预测效果优于旧样本，则说明新样本直接带来建模收益；如果没有优于旧样本，则需要用复核图、分层统计和物理指标证明样本筛选更接近目标事件，并说明下一步瓶颈。

## 2. 审计结论

本目标已达到“样本筛选与诊断闭环”完成状态，但没有达到“车辆-only 指标优于旧样本”的强结果。

当前可以下的结论是：

1. v0.6 已经生成可筛选、可复核、可追溯的 episode-first 事件样本清单。
2. v0.6 不再把 `.aed` 或道路设计点直接当作事件真值，而是先判定真实发生的车辆动态 episode，再补方向盘响应和回正/纠正信息。
3. v0.6 已经给出清晰分桶：最干净核心训练候选、坐标需复核扩展候选、弱响应/负样本、连续任务复核、场景暂缓复核、因果顺序不清复核。
4. 使用重筛样本完成了纯车辆/道路预测对照。结果没有优于旧 B 轨道：主轨道 test RMSE=0.679927，高于旧 B 轨道 0.533667。
5. 指标未提升的原因不是横向偏移特征屏蔽造成的；保留横向偏移的轨道 test RMSE=0.680265，也没有改善。
6. 当前最合理解释是：v0.6 更集中在真实大幅响应、回正、反打和复杂修正 episode 上，所以车辆-only 线性/模板类模型更难处理。该结果支持“样本语义更清楚，但模型能力不足”的诊断。

因此，本阶段可以结束；下一阶段不应直接声称连续风格/生理/EEG 有效，而应先进入车辆-only 响应分解或结构化轨迹建模。

## 3. 要求到证据的检查表

| 目标要求 | 完成状态 | 证据 |
|---|---|---|
| 基于 v0.4/v0.5/GPTPro 审查意见 | 已完成 | GPTPro 回复归档：`09_reports/gptpro_event_anchor_reply_20260512/20260512_event_anchor_v05_response_manualpaste.md`；v0.6 规则报告：`09_reports/event_v0_6_screening_rule_from_gptpro_20260512_cn.md`；episode-first 回复归档：`09_reports/gptpro_event_anchor_reply_20260512/20260513_episode_first_response_manualpaste.md` |
| 生成 v0.6 事件样本总清单 | 已完成 | `02_samples/episode_first_event_v0_6/tables/episode_candidates_v0_6.csv`，共 908 条 |
| 明确可训练事件 | 已完成 | `02_samples/episode_first_event_v0_6/tables/primary_training_events_v0_6.csv`，19 条第一版最干净核心训练候选 |
| 明确需复核事件 | 已完成 | `02_samples/episode_first_event_v0_6/tables/manual_review_events_v0_6.csv`，643 条；其中包括连续任务、弱响应、暂缓场景、因果不清等 |
| 明确仅作响应确认事件 | 已覆盖 | `02_samples/episode_first_event_v0_6/tables/response_confirm_only_v0_6.csv` 已生成；当前为 0 条，因为 episode-first 规则没有把峰值点或方向盘响应点作为独立因果锚点纳入 |
| 明确暂缓或排除事件 | 已完成 | `02_samples/episode_first_event_v0_6/tables/holdout_or_excluded_v0_6.csv`，39 条；包括 30 条场景语义暂缓复核和 9 条因果顺序不清复核 |
| 输出坐标风险扩展候选 | 已完成 | `02_samples/episode_first_event_v0_6/tables/coordinate_flagged_expansion_events_v0_6.csv`，246 条；这些样本车辆动态和方向盘响应成立，但横向偏移坐标需复核 |
| 输出弱响应/负样本 | 已完成 | `02_samples/episode_first_event_v0_6/tables/trigger_no_effect_or_weak_response_v0_6.csv`，298 条 |
| 输出分层统计 | 已完成 | `episode_decision_summary_v0_6.csv`、`episode_label_summary_v0_6.csv`、`episode_module_summary_v0_6.csv` |
| 输出复核图 | 已完成 | `02_samples/episode_first_event_v0_6/figures/episode_review_panels/`，36 张分组代表图；索引为 `episode_review_panel_index_v0_6.csv` |
| 输出概览图 | 已完成 | `02_samples/episode_first_event_v0_6/figures/episode_first_v0_6_summary.png` |
| 输出中文说明 | 已完成 | 用户版：`09_reports/stage02_episode_first_v0_6_user_summary_cn.md`；技术版：`09_reports/episode_first_event_v0_6_cn.md` |
| 使用重筛样本做纯车辆/道路预测对照 | 已完成 | `03_baselines/scripts/stage03_episode_first_vehicle_baselines_v0_1.py`；输出目录：`03_baselines/stage03_episode_first_vehicle_baselines_v0_1/` |
| 与旧样本/旧车辆基线比较 | 已完成 | 旧 B 轨道 RBF KRR：test RMSE=0.533667；v0.6 主轨道：test RMSE=0.679927；见 `stage03_episode_first_vehicle_baselines_user_summary_cn.md` 和 `episode_first_vehicle_metrics.csv` |
| 用物理指标判断是否更合理 | 已完成 | `episode_first_vehicle_metrics.csv` 包含 RMSE、错侧率、大幅响应召回、严重幅值不足率、峰值时间、尾段漂移、反向修正匹配等指标 |
| 说明结果未优于旧样本时的解释 | 已完成 | `stage03_episode_first_vehicle_baselines_user_summary_cn.md` 明确说明新样本未提升车辆-only 指标，但更集中在复杂真实 episode，暴露车辆-only 模型能力不足 |
| 更新项目日志和产物索引 | 已完成 | `00_project_notes/PROJECT_STATUS_CN.md`、`TASK_QUEUE_CN.md`、`ARTIFACT_INDEX_CN.md`、`daily_logs/2026-05-13.md` |

## 4. 关键数量核验

v0.6 事件总表：

- 总 episode：908
- 第一版最干净核心训练候选：19
- 坐标需复核扩展候选：246
- 车辆动态明显但方向盘响应不足，可作为弱响应/负样本：298
- 连续超车任务，需拆子事件：306
- 场景相关性或语义暂缓复核：30
- 因果顺序不清复核：9
- 暂缓/排除导出表：39
- 复核代表图：36 张

纯车辆/道路预测对照：

| 轨道 | 样本数 | train/val/test | 横向偏移特征 | val 选择模型 | test RMSE | 错侧率 | 大幅响应召回 | 严重幅值不足率 |
|---|---:|---|---|---|---:|---:|---:|---:|
| EP2_expanded_no_lateral_2s | 265 | 183/37/45 | 不使用 | ridge_rich_context_no_subject | 0.603605 | 0.355556 | 0.000000 | 0.400000 |
| EP3_expanded_no_lateral_3s | 265 | 183/37/45 | 不使用 | formal_ridge_vehicle_context_no_subject | 0.679927 | 0.266667 | 0.250000 | 0.355556 |
| EP3_expanded_with_lateral_3s | 265 | 183/37/45 | 使用 | formal_ridge_vehicle_context_no_subject | 0.680265 | 0.288889 | 0.250000 | 0.355556 |
| 旧 B_response3s_strict_core | 270 | test=40 | 旧设置 | rbf_kernel_ridge_context_no_subject | 0.533667 | 0.225000 | 0.750000 | 0.125000 |

## 5. 覆盖范围和不能下的结论

已覆盖：

- 事件样本筛选和重建。
- 清晰可选的 v0.6 表格。
- 复核图、分层统计和物理指标。
- 纯车辆/道路预测对照。
- 与旧样本车辆基线的比较。

不能下的结论：

- 不能说 v0.6 在车辆-only 预测指标上优于旧样本。
- 不能说连续驾驶风格、生理数据或 EEG 已经在 v0.6 上有效。
- 不能说坐标需复核扩展候选已经全部人工确认。
- 不能把 19 个严格核心直接当成足够训练完整深度模型的数据量。

## 6. 后续建议

下一阶段不应直接加连续风格、生理或 EEG。更合理路线是：

1. 以 v0.6 正样本为基础，先做车辆-only 响应分解模型。
2. 将方向、幅值、峰值时间、回正/反打、多段修正、尾段稳定作为结构化目标。
3. 只有当车辆-only 响应分解模型能解释当前复杂 episode，再进入连续风格和生理数据的增量验证。

