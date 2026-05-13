# GPTPro 事件锚点审查决策记录

生成时间：2026-05-12

## 接受的建议

1. 接受“先暂停训练，继续推进 v0.6 样本清单”的建议。  
   理由：当前事件锚点语义仍未完全清楚，直接训练会让模型结果难以解释。

2. 接受“增加被试相关/暴露点”的建议。  
   后续样本清单需要区分 `t_design`、`t_trigger`、`t_ego_exposure`、`t_response_onset`、`t_response_peak`、`t_train_anchor`。

3. 接受“v0.5 高置信只是复核池，不是训练池”的建议。  
   后续不会把 314 个高置信候选直接作为训练样本。

4. 接受“方向盘和车身姿态峰值只能用于响应确认，不能单独定义因果锚点”的建议。  
   后续会把响应峰值候选降级为 `response_confirm_only`。

5. 接受“第一版训练样本宁可少而干净”的建议。  
   第一版 v0.6 优先考虑 `curve1/curve2`、`differentmu_road` raw μ、人工通过的 `fix_road`。

6. 接受“先做车辆/道路-only 强基线，再谈连续风格和生理数据”的建议。  
   后续风格/生理有效性必须建立在新锚点和强车辆基线固定之后。

## 暂缓或需要本地验证的建议

1. `t_ego_exposure` 当前不一定能全自动计算。  
   对变道/停车事件，后续先用可用的代理指标，如目标车道、触发时间、被试车辆横向/纵向响应、必要时人工图像复核。若缺少相对目标车位置/TTC，不强行填充精确暴露点。

2. `curve1/curve2` 的曲率开始点和有效曲率阈值点需要进一步解析。  
   目前 v0.5 主要有道路模块入口和姿态峰值，v0.6 应补曲率阈值候选，但需要用原始车辆曲率或道路几何确认。

3. `differentmu_road` 优先 raw μ 是合理的，但 cfg 低附着段变化点仍可保留为复核候选。  
   最终训练优先 raw μ 跳变/首次低附着，cfg 仅在映射可靠时纳入。

4. `longstraight` 和 `stop` 暂缓进入第一版训练，但不删除。  
   它们进入诊断/扩展池，用于后续检查被试相关性和纵向/避让事件。

## 不采用或不立即采用

1. 不立即构建包含全部事件类型的大样本训练集。
2. 不直接把 `middle_section` 混入单点事件模型。
3. 不用旧锚点接近程度作为强纳入条件，只保留为解释字段。
4. 不用方向盘变化量作为唯一纳入条件。

## 本地证据

- v0.4 候选锚点清单：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/ego_direction_design_anchor_v0_4/tables/ego_direction_design_anchor_candidates_v0_4.csv`
- v0.5 事件候选评分表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/event_candidate_filter_v0_5/tables/event_candidate_scores_v0_5.csv`
- v0.5 高置信复核清单：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/event_candidate_filter_v0_5/tables/event_candidates_high_confidence_v0_5.csv`
- v0.5 用户版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage02_event_filter_user_summary_cn.md`
- GPTPro 回复归档：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/gptpro_event_anchor_reply_20260512/20260512_event_anchor_v05_response_manualpaste.md`
