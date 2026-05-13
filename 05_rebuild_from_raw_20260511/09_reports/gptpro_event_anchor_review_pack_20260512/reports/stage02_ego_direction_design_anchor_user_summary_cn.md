# 阶段 2 补充：被试方向设计点与候选锚点重建 v0.4

生成时间：2026-05-12

## 这个阶段为什么做

我们怀疑旧流程的预测效果上不去，可能不是模型结构本身的问题，而是事件锚点没有真正对准“场景开始影响被试”的时刻。

这次用户提供了小论文文档，因此可以把论文里的实验设计说明作为依据，不再只依赖旧事件表或方向盘响应反推锚点。

## 这个阶段检查了什么

本轮检查了三类信息：

1. 小论文中的实验设计文字和表格；
2. SILAB 道路配置里的车道、路面附着系数和道路模块；
3. 已有车辆轨迹投影得到的每条记录通过各道路模块的时间。

## 目前发现了什么

小论文明确给出了七类场景：

| 场景类型 | 设计含义 | 更合理的锚点依据 |
|---|---|---|
| 连续超车负荷 | `middle_section` 重复 9 次，约 23 m/s 交通车；用户确认连接路段会让驾驶员连续超车 | 连续超车负荷事件段，连接段入口和段内横向动态共同确认 |
| 大货车紧急变道 | 目标车侵入本车道 | 显式触发或方向盘角速度峰值，但必须是被试方向 |
| 施工/维修路段 | 车道受限并有交通车干扰 | 位置触发 + 车身姿态确认 |
| 前车急停 | 前车减速或停止 | 被试方向前车触发或制动/减速度确认 |
| 低附着路段 | `mu=0.8-0.2` 不同附着区域 | 进入低 `mu` 区域或 `mu` 变化点 |
| 弯道路段 | `curve1/curve2/curve3` | 局部横滚峰值、横向加速度峰值、道路曲率位置 |
| 匝道/汇入 | 多车道汇入和交通交互 | 汇入区位置 + 车身姿态确认 |

当前最清楚的是：

1. 弯道路段、低附着路段和 `middle_section` 连续超车路段仍然是优先重建对象，因为它们有明确道路/任务设计依据。
2. `longstraight` 不能再只按背景交通处理。它有高速连续车流背景，但 `.aed` 中也能定位 MAN TGL 25->26 显式变道和 Chrysler300 停车触发，应进入候选锚点审查。
3. `fix_road` 也不能再只写“待确认”。它有 MAN TGL 25->26 和 BMW m340 26->25 两类显式变道触发，符合维修/施工路段变道避让设计。
4. `stop`、`zd` 仍需要继续查具体实验设计，否则不能直接用背景交通触发点当真值。
5. 方向盘转角不能作为锚点来源，只能作为事件后的预测目标；但小论文里旧方案确实用了 `steer-rate peak80`，这也是我们现在要重新审计的重点风险之一。

## 已生成的结果

1. 小论文场景依据摘录：  
   `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/small_paper_scene_design_extract_v0_4.md`

2. 被试方向候选锚点重建报告：  
   `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/ego_direction_design_anchor_rebuild_v0_4_cn.md`

3. 候选锚点清单：  
   `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/ego_direction_design_anchor_v0_4/tables/ego_direction_design_anchor_candidates_v0_4.csv`

4. 场景模块汇总表：  
   `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/ego_direction_design_anchor_v0_4/tables/ego_direction_design_anchor_module_summary_v0_4.csv`

5. 道路配置车道/附着解析表：  
   `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/ego_direction_design_anchor_v0_4/tables/cfg_lane_mu_geometry_v0_4.csv`

6. 被试方向低附着段表：  
   `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/ego_direction_design_anchor_v0_4/tables/cfg_ego_direction_mu_segments_v0_4.csv`

## 当前候选数量

根据用户补充“道路之间的连接路段也会有连续超车事件”，以及进一步说明 `longstraight` 和维修路段都有变道触发点，我已经重新生成候选表。当前共生成 4519 行候选锚点/上下文点：

- `middle_section` 连续超车候选：2260 行，包括连续超车段入口、中点、横向偏移变化峰值、横向加速度峰值、横摆角速度峰值各 452 行；
- `longstraight` 候选：255 行，包括场景上下文入口 85 行、MAN TGL 显式变道触发点 85 行、Chrysler300 显式停车触发点 85 行；
- `curve1` 候选：420 行；
- `curve2` 候选：350 行；
- `differentmu_road` 候选：514 行；
- `fix_road` 候选：495 行，其中显式变道触发点 140 行，另有道路入口/中点和车身横向姿态峰值候选；
- `stop` 候选：225 行；
- `curve3` 和 `zd` 当前没有记录级候选，说明现有车辆轨迹映射对这两个模块还不充分，或者记录未稳定覆盖到这两个模块。

## 哪些结果可信

目前比较可信的是：

1. 小论文中关于场景类型和锚点依据的描述；
2. `differentmu_road` 的低附着设计，因为配置文件和原始车辆 `zx1|mu` 都能提供证据；
3. `curve1/curve2` 的弯道锚点方向，因为小论文明确说弯道用局部横滚峰值；
4. `middle_section` 是连续超车负荷事件段这一点，因为用户已经确认；
5. `longstraight` 同时包含高速背景车流和显式变道/停车触发点这一点，因为用户补充说明和 `.aed` 触发文件一致；
6. `fix_road` 包含维修/施工变道触发这一点，因为用户补充说明、小论文文字和 `.aed` 中两个 ChangeLane 触发一致。

## 哪些还不能下结论

1. 不能说当前 4209 个候选都可以直接变成训练样本。
2. 不能说 `longstraight` 和 `fix_road` 的显式触发点已经可以直接变成最终训练锚点，它们还需要车身姿态和预测窗口可视化确认。
3. 不能说 `stop`、`zd` 的主锚点已经确定。
4. 不能说旧 `steer-rate peak80` 锚点完全错误，但它有“用驾驶员响应定义事件”的风险，需要和场景设计点、车身姿态点对比。
5. 不能直接进入风格/生理建模，因为样本锚点还需要可视化复核。

## 下一阶段建议

下一步优先做候选锚点可视化复核：

1. 先画 `middle_section` 连续超车段的入口、横向偏移变化峰值、横摆角速度峰值、横向加速度峰值和旧锚点对比图；
2. 画 `longstraight` 的 MAN TGL 25->26 显式变道、Chrysler300 停车触发、车身横向/制动响应和旧锚点对比图；
3. 画 `fix_road` 的 MAN TGL 25->26、BMW m340 26->25 两类显式变道触发、车身姿态峰值和旧锚点对比图；
4. 再画 `curve1/curve2` 的道路入口、横滚峰值、横向加速度峰值和旧锚点对比图；
5. 再画 `differentmu_road` 的低 `mu` 进入点、`mu` 跳变点、车身姿态和方向盘响应；
6. 把旧锚点分成“可保留、偏早、偏晚、语义不清”四类；
7. 只把视觉和物理意义都合理的候选锚点进入下一版样本清单。

目前不建议继续训练模型。先把锚点问题确认清楚，比继续改模型更重要。
