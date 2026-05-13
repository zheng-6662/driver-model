# longstraight 场景解释修正说明

生成时间：2026-05-12 19:15

## 修正原因

用户补充说明：`longstraight` 场景中 25/26 车道那一侧的车辆是连续出现的驾驶车辆，主要用于模拟高速公路上的交通流背景。后续分析只需要关注被试车辆实际行驶的方向。

因此，之前把 25/26 车道的 `Activate`、`Stop`、`ChangeLane` 直接解释为被试方向上的主要事件触发原因，是过强解释，需要修正。

## 修正后的解释

1. `longstraight.autosave.1.aed` 中确实存在 25/26 车道的车辆、车流源和触发点。
2. 这些 25/26 车道设置应优先标记为“背景交通流”，不能直接作为被试方向事件锚点。
3. 当前被试车道投影显示，被试车辆在 `longstraight` 场景触发点附近主要位于 21/22/23 侧：
   - 23 车道：524 行；
   - 22 车道：68 行；
   - 21 车道：3 行。
4. 当前解析到的 `longstraight` 交通触发点与被试车同方向侧的行数为 0。
5. 因此，`longstraight` 的后续锚点重建应只关注被试方向 21/22/23 侧的事件来源、道路任务设定和车辆姿态变化。

## 对后续锚点重建的影响

后续不应把 25/26 侧连续交通流的激活、停车、换道触发点直接作为模型样本锚点。

更合理的流程是：

1. 先确定被试车辆实际行驶方向和车道；
2. 只在被试方向上查找真实事件来源；
3. 如果场景文件中没有被试方向触发点，则继续查实验设计文本、任务说明或其它场景配置；
4. 再用车辆姿态、横向加速度、横摆、侧向偏移等非方向盘信号确认被试是否真的受到影响；
5. 最后才截取方向盘未来响应作为预测标签。

## 已更新产物

- 完整报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/scene_trigger_audit_v0_2_cn.md`
- 用户版说明：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage02_scene_trigger_user_summary_cn.md`
- 被试同方向触发候选表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scene_trigger_audit_v0_2/tables/longstraight_ego_direction_relevant_triggers_v0_2.csv`
- 背景交通触发映射表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scene_trigger_audit_v0_2/tables/longstraight_background_traffic_triggers_v0_2.csv`
- 被试车道投影图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scene_trigger_audit_v0_2/figures/longstraight_ego_lane_projection_v0_2.png`

