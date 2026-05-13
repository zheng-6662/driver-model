# 场景触发点与旧锚点审计 v0.2

生成时间：2026-05-12 20:08:25

## 这一步为什么做

上一版道路事件审计只知道车辆经过了哪个道路模块，以及旧锚点是否接近道路边界、道路曲率或车身动态。但用户进一步关心的是：例如 `longstraight` 场景里，被试所在道路附近到底布置了哪些交通车，具体设置了哪些激活、停车、换道触发点。

因此本次审计直接读取 SILAB `.aed` 场景布局文件，提取交通参与对象和触发点，再把这些触发点换算到道路纵向位置和每条被试记录的相对时间轴上，最后与旧 v400 锚点对齐。

## 当前关键结论

- 解析到交通对象行数：81
- 解析到场景触发点行数：19
- 触发点换算到被试记录时间轴后的行数：1436
- 全部场景触发点处被试车道估计行数：1436
- 与旧 v400 锚点完成最近邻对齐的行数：6247
- 旧锚点 1 秒内接近场景触发点的数量：175
- 旧锚点 2 秒内接近场景触发点的数量：356

触发点类型计数：

```text
{
  "Activate": 13,
  "ChangeLane": 3,
  "Stop": 3
}
```

全部场景按“是否在被试同方向侧”的初步分类：

```text
{
  "background_or_opposite": 1436
}
```

各道路模块解释摘要：

| module_name | traffic_object_count | traffic_trigger_count | traffic_trigger_names | ego_direction_related_rows | background_or_opposite_rows | unknown_lane_rows | interpretation_cn |
| --- | --- | --- | --- | --- | --- | --- | --- |
| curve1 | 3 | 1 | Activate | 0 | 82 | 0 | 当前显式触发主要不在被试同方向侧，应优先标记为背景/对向交通，不能直接作为被试方向样本锚点。 |
| curve2 | 3 | 1 | Activate | 0 | 69 | 0 | 当前显式触发主要不在被试同方向侧，应优先标记为背景/对向交通，不能直接作为被试方向样本锚点。 |
| differentmu_road | 4 | 0 |  | 0 | 0 | 0 | 解析到交通对象，但没有显式交通触发点；更可能是背景交通或静态场景布置，需要结合实验设计文本确认。 |
| fix_road | 8 | 8 | Activate；ChangeLane | 0 | 562 | 0 | 当前显式触发主要不在被试同方向侧，应优先标记为背景/对向交通，不能直接作为被试方向样本锚点。 |
| longstraight | 4 | 7 | Activate；ChangeLane；Stop | 0 | 595 | 0 | 当前显式触发主要不在被试同方向侧，应优先标记为背景/对向交通，不能直接作为被试方向样本锚点。 |
| middle_section | 48 | 0 |  | 0 | 0 | 0 | 解析到交通对象，但没有显式交通触发点；更可能是背景交通或静态场景布置，需要结合实验设计文本确认。 |
| stop | 6 | 2 | Stop | 0 | 128 | 0 | 当前显式触发主要不在被试同方向侧，应优先标记为背景/对向交通，不能直接作为被试方向样本锚点。 |
| zd | 5 | 0 |  | 0 | 0 | 0 | 解析到交通对象，但没有显式交通触发点；更可能是背景交通或静态场景布置，需要结合实验设计文本确认。 |

旧锚点相对最近场景触发点的时间差分组：

```text
{
  "old_after_scene_gt2s": 3857,
  "old_before_scene_gt2s": 2031,
  "old_close_to_scene_2s": 181,
  "old_close_to_scene_0p5s": 92,
  "old_close_to_scene_1s": 83,
  "no_scene_trigger": 3
}
```

## longstraight 被试车道初步投影

本次还对 `longstraight` 做了第一版被试车道估计。方法是：在每条被试记录中找到车辆经过场景触发点的时刻，读取此时车辆的横向位置，并按 `longstraight_Area2.cfg` 中 21-27 号车道中心偏移做最近车道匹配。

被试车道计数：

```text
{
  "23": 524,
  "22": 68,
  "21": 3
}
```

- 被试车与触发点在同一车道的行数：0
- 被试车与触发点在同一方向侧的行数：0

根据用户补充说明，25/26 车道那一侧的车辆是用于模拟高速公路连续交通流的背景车辆，不应作为被试行驶方向上的主要事件触发原因。结合当前投影结果，被试车主要处在 21/22/23 侧，而解析到的 `longstraight` 交通触发点主要在 25/26 侧。因此本次解析到的 25/26 侧 Activate / Stop / ChangeLane 应标记为“背景交通触发”，不能直接作为被试方向上的事件锚点。

- 当前可视为被试同方向相关的 `longstraight` 触发点行数：0
- 当前应标记为背景交通的 `longstraight` 触发点时间映射行数：595

## longstraight 场景可以确认的信息

`longstraight` 的 `.aed` 文件不是只有道路几何，它的第 5 层里确实有交通对象和交通触发点。当前解析到的交通对象如下：

| figure_name | figure_id | raw_name | title | lane_id | tau | relative_s_in_module_m | road_s_global_m | number_of_vehicles | vehicle_type | start_speed | target_speed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f0 | 76 | Source | Vehicle source | 26 | 0.4318 | 194.6970 | 1355.3220 | 5 | Car | 30 | 32 |
| f1 | 60 | Source | Vehicle source | 26 | 0.1209 | 54.4989 | 1215.1239 | 4 | Car | 28.0 | 32.0 |
| f2 | 51 | MAN_TGL | MAN TGL truck | 25 | 0.5022 | 226.4246 | 1387.0496 |  |  | 15 | 22 |
| f3 | 40 | Chrysler300 | Chrysler300 | 25 | 0.5185 | 233.7719 | 1394.3969 |  |  | 15 | 22 |

当前解析到的触发点如下：

| figure_name | figure_id | raw_name | lane_id | tau | relative_s_in_module_m | road_s_global_m | target_name | target_title | target_lane_id | change_target_lane | change_time_or_distance | description |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f4 | 53 | ChangeLane | 25 | 0.5526 | 249.1526 | 1409.7776 | MAN_TGL | MAN TGL truck | 25 | 26 | 10 | Change lane |
| f5 | 62 | Activate | 26 | 0.0675 | 30.4380 | 1191.0630 | Source | Vehicle source | 26 |  |  | Activates the traffic source. |
| f6 | 71 | Stop | 25 | 0.5512 | 248.5374 | 1409.1624 | Chrysler300 | Chrysler300 | 25 |  |  | Stop the vehicle immediately. |
| f7 | 72 | Activate | 25 | 0.4489 | 202.4144 | 1363.0394 | MAN_TGL | MAN TGL truck | 25 |  |  | Activates a vehicle. |
| f8 | 73 | Activate | 25 | 0.4492 | 202.5347 | 1363.1597 | Chrysler300 | Chrysler300 | 25 |  |  | Activates a vehicle. |
| f9 | 75 | Activate | 25 | 0.0675 | 30.4257 | 1191.0507 | Source | Vehicle source | 26 |  |  | Activates the traffic source. |
| f10 | 78 | Activate | 25 | 0.4492 | 202.5310 | 1363.1560 | Source | Vehicle source | 26 |  |  | Activates the traffic source. |

用白话说，`longstraight` 至少包含这些背景交通设置：

- 26 车道上的两个车流源：一个生成 4 辆小汽车，一个生成 5 辆小汽车；
- 25 车道上的一辆 `MAN TGL truck`；
- 25 车道上的一辆 `Chrysler300` 小轿车；
- 25 车道附近的车辆激活点；
- `Chrysler300` 的立即停车触发点；
- `MAN TGL truck` 从 25 车道向 26 车道换道的触发点。

但根据用户补充说明，这些 25/26 车道车辆主要是为了模拟高速公路上的连续交通流。后续建模和锚点重建时，不能把这些背景交通触发点直接当成被试方向的真实事件。

## 仍然不能直接下的结论

1. 还不能仅凭 `.aed` 说被试车一定在 25 车道或 26 车道。要确认被试实际车道，需要把每条车辆轨迹坐标投影到车道线，而不是只看交通车所在车道。
2. `.aed` 触发点是场景设定触发点；旧 v400 锚点多来自方向盘速率或后处理上下文。两者不是同一个定义。
3. 如果旧锚点明显晚于场景触发点，旧模型可能是在事件已经发生、甚至驾驶员已经开始响应之后才对齐样本。
4. 如果旧锚点明显早于场景触发点，可能是旧锚点和真实场景事件错配，也可能旧锚点抓到了其它车辆/道路动态。

## 建议下一步

1. 对 `longstraight`，优先只看被试方向 21/22/23 侧的事件来源，不再把 25/26 侧连续交通流作为主锚点。
2. 继续查实验设计文本或其它场景文件，确认被试方向上是否另有触发点、道路扰动、任务指令或车辆姿态触发规则。
3. 对旧 v400 锚点按“是否处在被试方向、是否接近同方向场景触发、是否更像车身响应后验”重新分组。
4. 抽查旧模型坏样本是否集中在“背景交通误配为事件”或“无同方向场景触发”的样本上。
5. 如果被试方向场景触发点无法在 `.aed` 中找到，则后续锚点应更多依赖道路任务设计文本 + 被试车辆姿态，而不是对向侧背景交通。

## 主要产物

- 交通对象表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\scene_trigger_audit_v0_2\tables\aed_traffic_objects_v0_2.csv`
- 场景触发点表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\scene_trigger_audit_v0_2\tables\aed_traffic_triggers_v0_2.csv`
- 触发点到每条被试记录的时间映射：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\scene_trigger_audit_v0_2\tables\scene_trigger_session_times_v0_2.csv`
- longstraight 被试车道估计：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\scene_trigger_audit_v0_2\tables\longstraight_ego_lane_at_scene_triggers_v0_2.csv`
- longstraight 被试同方向触发候选：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\scene_trigger_audit_v0_2\tables\longstraight_ego_direction_relevant_triggers_v0_2.csv`
- longstraight 背景交通触发映射：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\scene_trigger_audit_v0_2\tables\longstraight_background_traffic_triggers_v0_2.csv`
- 全部场景被试车道估计：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\scene_trigger_audit_v0_2\tables\all_scene_ego_lane_at_scene_triggers_v0_2.csv`
- 全部场景被试同方向触发候选：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\scene_trigger_audit_v0_2\tables\all_scene_ego_direction_relevant_triggers_v0_2.csv`
- 全部场景背景/对向触发映射：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\scene_trigger_audit_v0_2\tables\all_scene_background_or_opposite_triggers_v0_2.csv`
- 各场景设计摘要：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\scene_trigger_audit_v0_2\tables\scene_design_by_module_summary_v0_2.csv`
- 旧锚点与最近场景触发点对齐表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\scene_trigger_audit_v0_2\tables\old_anchor_vs_scene_trigger_v0_2.csv`
- 审计汇总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\scene_trigger_audit_v0_2\tables\scene_trigger_audit_summary_v0_2.csv`
- longstraight 场景触发点图：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\scene_trigger_audit_v0_2\figures\longstraight_scene_trigger_map_v0_2.png`
- 旧锚点相对场景触发点时间差图：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\scene_trigger_audit_v0_2\figures\old_anchor_scene_trigger_delta_hist_v0_2.png`
- longstraight 被试车道投影图：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\scene_trigger_audit_v0_2\figures\longstraight_ego_lane_projection_v0_2.png`
