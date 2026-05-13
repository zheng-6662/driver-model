# 阶段 2 补充：场景触发点审计用户版说明

生成时间：2026-05-12 20:08:25

## 2026-05-12 最新修正：longstraight 和维修路段也有变道触发点

用户进一步说明：`longstraight` 和维修路段都涉及变道，并且设置了触发点。结合 `.aed` 解析结果，当前应按以下方式修正：

1. `longstraight` 的 25/26 连续车流确实有高速背景含义，但其中 MAN TGL truck 的 ChangeLane 25->26 和 Chrysler300 的 Stop 是显式触发点，必须进入候选锚点可视化审查。
2. `fix_road`/维修路段有 MAN TGL truck 25->26 和 BMW m340 26->25 两个显式变道触发，符合施工/维修路段变道避让设计，不能再只写“待确认”。
3. 这些触发点仍不能直接当最终训练锚点；下一步要看触发点附近被试车辆是否确实出现横向、纵向或方向盘响应。

## 这个阶段为什么做

我们之前怀疑旧模型卡住，不只是模型结构问题，也可能是事件锚点没有对准真实场景事件。你问到 `longstraight` 场景中被试开的车道附近有哪些车、设置了什么触发点，这正是需要补的一层信息。

## 目前发现了什么

我已经从 `longstraight.autosave.1.aed` 里解析到交通车辆和触发点。这个场景里，25/26 车道附近确实有连续交通流背景设置：26 车道有车流源，25 车道有货车和小轿车，并且有激活、停车、换道触发点。根据最新补充，这些车流一方面模拟高速公路背景，另一方面其中的 MAN TGL 变道和 Chrysler300 停车触发也属于需要审查的候选事件点。

其中比较关键的是：

- `Chrysler300` 小轿车在 25 车道附近有立即停车触发；
- `MAN TGL truck` 货车在接近位置有换道触发，目标车道写为 26；
- 26 车道还设置了车流源。

根据你的补充说明，25/26 车道那边的车辆是连续出现的背景交通，主要用于模拟高速公路交通流；但你随后进一步说明 `longstraight` 也设置了变道触发点。因此现在的解释应再次修正为：25/26 侧的普通连续车流只作为背景，显式 Stop / ChangeLane 触发点必须单独作为候选锚点审查。

我还做了第一版被试车道投影。结果显示，被试车在场景触发点附近的估计车道分布为：

```text
{
  "23": 524,
  "22": 68,
  "21": 3
}
```

这一步只是直道上的几何近似。后续仍要重点确认被试实际行驶方向和触发点相对关系，但不能再因为 25/26 是连续车流就排除其中的显式变道/停车触发。正确做法是：触发点先进入候选清单，再用车身姿态和旧锚点对齐图判断它是否是有效样本锚点。

我也把同样的判断推广到了其它场景：不是所有 `.aed` 中的车辆/触发点都能直接作为模型样本锚点，必须先判断它是否处在被试同方向侧。全部场景初步分类如下：

```text
{
  "background_or_opposite": 1436
}
```

各场景的摘要在这里：

`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\scene_trigger_audit_v0_2\tables\scene_design_by_module_summary_v0_2.csv`

## 目前还不能确定什么

还不能只凭这个文件说被试车一定在哪条车道。下一步必须把每条被试车辆轨迹投影到车道线上，确认被试通过触发点时与这些交通车的相对位置。

## 对旧流程有什么影响

这一步说明旧流程用方向盘速率或车辆响应后验选锚点，确实可能和真实场景触发点不是一个时刻。后续更合理的做法是：

1. 先用场景触发点定义事件发生位置；
2. 再用车身姿态确认被试是否真的受到影响；
3. 最后才截取方向盘未来响应作为预测标签。

这样比单纯用方向盘变化找锚点更符合因果顺序。

## 推荐你优先看

1. `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\scene_trigger_audit_v0_2_cn.md`
2. `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\scene_trigger_audit_v0_2\tables\aed_traffic_triggers_v0_2.csv`
3. `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\scene_trigger_audit_v0_2\tables\longstraight_ego_lane_at_scene_triggers_v0_2.csv`
4. `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\scene_trigger_audit_v0_2\tables\old_anchor_vs_scene_trigger_v0_2.csv`
5. `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\scene_trigger_audit_v0_2\figures\longstraight_scene_trigger_map_v0_2.png`
6. `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\scene_trigger_audit_v0_2\figures\longstraight_ego_lane_projection_v0_2.png`
7. `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\scene_trigger_audit_v0_2\tables\scene_design_by_module_summary_v0_2.csv`
## 2026-05-12 追加修正：其他场景也按“被试方向设计”继续审计

用户进一步确认：`longstraight` 中 25/26 车道的车是连续出现的背景交通，用来模拟高速公路车流；同时又补充 `longstraight` 和维修路段都涉及变道触发点。因此，普通连续车流和显式触发对象需要分开处理。

用户随后补充：道路之间的连接路段也会有事件，驾驶员会在这些连接段连续超车。因此 `middle_section` 不能再简单当作普通过渡段或背景交通段，而应作为连续超车负荷事件段单独审计。

因此当前结论修正为：

1. `.aed` 交通触发点不能直接等同于被试方向主事件锚点。
2. `longstraight` 的 25/26 普通连续车流按背景处理，但 MAN TGL 25->26 变道和 Chrysler300 Stop 要进入候选锚点审查。
3. `fix_road` 的 MAN TGL 25->26 和 BMW m340 26->25 是维修/施工路段变道触发候选，需要进入可视化审查。
4. 其他场景没有显式同侧交通触发，不代表没有事件设计；更可能说明主事件来自道路几何、路面附着变化、实验任务点或车辆姿态响应。
5. `middle_section` 的主线应改为连续超车锚点：连接段入口是负荷开始，段内横向偏移变化、横摆角速度、横向加速度用于确认是否发生真实超车/变道响应。
6. 下一步锚点重建应以“被试方向道路/任务设计点 + 车身姿态确认”为主，方向盘转角只作为事件后的预测标签。

新增材料：

- 场景设计与被试方向锚点工作图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/scene_design_working_map_v0_3_cn.md`
- 各场景被试方向事件来源工作表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scene_trigger_audit_v0_2/tables/ego_direction_scene_event_source_map_v0_3.csv`
