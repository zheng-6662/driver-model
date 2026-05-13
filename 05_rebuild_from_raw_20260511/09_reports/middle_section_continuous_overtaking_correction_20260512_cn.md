# middle_section 连续超车事件修正说明

生成时间：2026-05-12

## 用户补充

用户说明：道路之间的连接路段也会有事件，在这些连接路段会要求驾驶员连续超车。

## 对当前锚点审计的影响

之前把 `middle_section` 暂时写成“过渡/背景交通段”，这个表述需要修正。

新的解释是：

1. `middle_section` 不是普通空白连接路段；
2. 它承担连续超车负荷任务；
3. 这里的事件不是单个交通触发点，而是一段连续驾驶负荷；
4. 不能因为 `.aed` 没有显式触发点，就排除 `middle_section`；
5. 也不能把所有连接段入口都无条件当强事件，仍需要车身姿态和轨迹变化确认。

## 新锚点规则

`middle_section` 后续按“连续超车负荷事件段”处理。

候选锚点包括：

- 连续超车段入口；
- 连续超车段中点；
- 横向偏移变化峰值；
- 横向加速度峰值；
- 横摆角速度峰值。

其中，入口和中点来自道路/任务设计；横向偏移、横向加速度、横摆角速度来自车身姿态确认。方向盘转角仍只作为事件后的响应标签，不作为锚点定义依据。

## 已更新产物

- 候选锚点清单已重新生成：  
  `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/ego_direction_design_anchor_v0_4/tables/ego_direction_design_anchor_candidates_v0_4.csv`

- `middle_section` 当前新增 2260 行候选：  
  连续超车段入口、中点、横向偏移变化峰值、横向加速度峰值、横摆角速度峰值各 452 行。

- 完整报告已更新：  
  `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/ego_direction_design_anchor_rebuild_v0_4_cn.md`

- 用户查看版总结已更新：  
  `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage02_ego_direction_design_anchor_user_summary_cn.md`

## 下一步

优先为 `middle_section` 生成可视化图，检查连接段入口、横向偏移变化峰值、横向加速度峰值、横摆角速度峰值和旧锚点之间的关系。

如果图上能看到明确超车/变道响应，则 `middle_section` 可以进入新样本锚点重建的主候选；如果部分连接段没有明显超车动作，则需要筛掉弱响应或无响应样本。
