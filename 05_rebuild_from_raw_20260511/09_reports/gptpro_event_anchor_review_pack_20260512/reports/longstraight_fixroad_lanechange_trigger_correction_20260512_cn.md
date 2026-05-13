# longstraight 与维修路段变道触发修正说明

生成时间：2026-05-12

## 为什么要修正

之前我把 `longstraight` 中 25/26 车道的连续车辆主要理解为高速背景车流，因此表述偏保守，认为这些触发点不应直接作为被试方向主锚点。用户进一步说明后，需要修正：`longstraight` 和维修路段本身也都涉及变道，并且设置了触发点。

因此，现在不能把这些场景简单分成“背景交通”或“待确认”。更合理的处理是：

1. 连续车流可以是背景；
2. 其中显式设置的变道/停车触发点要单独进入候选锚点；
3. 候选锚点是否能变成训练样本，还必须看触发前后被试车辆是否真的出现合理响应。

## 已确认的触发点

### longstraight

`.aed` 触发文件中已经定位到：

- MAN TGL truck：`ChangeLane`，25 车道到 26 车道；
- Chrysler300：`Stop`，25 车道；
- 26 车道还有连续车辆源，用于模拟高速车流背景。

本轮重建后，`longstraight` 生成：

- 场景上下文入口：85 行；
- 显式变道触发点：85 行；
- 显式停车触发点：85 行。

### fix_road / 维修路段

`.aed` 触发文件中已经定位到：

- MAN TGL truck：`ChangeLane`，25 车道到 26 车道；
- BMW m340：`ChangeLane`，26 车道到 25 车道。

本轮重建后，`fix_road` 生成：

- 显式变道触发点：140 行；
- 道路模块入口/中点：各 71 行；
- 横向加速度、横摆角速度、车身横滚速率峰值：各 71 行。

## 当前判断

1. `longstraight` 不能再只写成“背景交通”。准确说法是：普通连续车流是背景，但显式变道/停车触发点是候选事件。
2. `fix_road` 不能再写成“需继续确认具体设计后才处理”。准确说法是：维修/施工路段存在明确变道触发点，已经可以进入候选锚点可视化审查。
3. 这些触发点仍不能直接等同最终训练锚点，因为它们是否真正影响被试车辆还要看触发点附近的横向加速度、横摆角速度、横向偏移、制动、车速和方向盘响应。

## 对下一步的影响

下一轮锚点可视化审查需要把以下场景放入高优先级：

1. `middle_section` 连续超车；
2. `longstraight` 显式变道/停车触发；
3. `fix_road` 显式变道触发；
4. `curve1/curve2` 弯道；
5. `differentmu_road` 低附着。

暂时不建议继续训练模型。应该先检查这些候选锚点与旧锚点、车身姿态、方向盘响应是否对齐，再决定新的样本清单。

## 已更新的文件

- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scripts/build_ego_direction_design_anchors_v0_4.py`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/ego_direction_design_anchor_v0_4/tables/ego_direction_design_anchor_candidates_v0_4.csv`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/ego_direction_design_anchor_v0_4/tables/ego_direction_design_anchor_module_summary_v0_4.csv`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scene_trigger_audit_v0_2/tables/ego_direction_scene_event_source_map_v0_3.csv`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/ego_direction_design_anchor_rebuild_v0_4_cn.md`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage02_ego_direction_design_anchor_user_summary_cn.md`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/scene_design_working_map_v0_3_cn.md`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage02_scene_trigger_user_summary_cn.md`
