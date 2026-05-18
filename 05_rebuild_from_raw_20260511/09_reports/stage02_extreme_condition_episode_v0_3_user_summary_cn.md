# 全量原始数据极限工况 episode 重筛 v0.3（用户查看版）

生成时间：2026-05-18 19:23:23

## 这次和 v0.2 最大区别

- 本次入口是 `原始车辆数据` 下的所有原始车辆 CSV，不再从旧 v0.2/v0.5/v0.6 候选表继续筛。
- 旧候选表只作为最近上下文贴回，用来解释当前 episode 是否靠近旧锚点或旧道路模块。
- 不再要求事件后一定出现明显回正或反打；弱响应、保守响应、延迟响应和无明显转向都保留下来。

## 全量扫描情况

- 扫描 CSV 文件数：92
- 成功读取车辆记录数：89
- 非被试 CSV 跳过：1
- 记录过短跳过：2
- 检测到 episode 总数：1574

## episode 分类结果

| v0_3_category | v0_3_category_cn | count | ratio |
| --- | --- | --- | --- |
| excluded | 排除样本 | 781 | 0.496188 |
| manual_review | 待人工复核 | 311 | 0.197586 |
| weak_or_conservative | 弱响应/保守响应 | 208 | 0.132147 |
| delayed_or_no_steer | 延迟或无明显转向响应 | 139 | 0.08831 |
| normal_control | 正常驾驶/普通弯道对照 | 86 | 0.0546379 |
| strong_response | 强响应型极限工况 | 49 | 0.0311309 |

## 当前解释边界

- 这一步仍然不是模型训练结果，只是重新定义样本库。
- 强响应样本适合后续轨迹预测试验；弱响应/保守样本更适合驾驶风格和生理状态差异分析。
- 如果车辆-only 基线在这套新样本上仍然出现方向错侧、幅值压缩或预测图物理意义不对，需要继续回到样本和锚点规则，而不是马上解释生理数据。

## 推荐优先查看

- 总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_3\tables\extreme_condition_episodes_all_v0_3.csv`
- 强响应：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_3\tables\strong_response_episodes_v0_3.csv`
- 弱/保守响应：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_3\tables\weak_or_conservative_response_episodes_v0_3.csv`
- 延迟/无明显转向：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_3\tables\delayed_or_no_steer_response_episodes_v0_3.csv`
- 复核图索引：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_3\tables\extreme_condition_review_panel_index_v0_3.csv`
- 复核图目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_3\figures\review_panels`