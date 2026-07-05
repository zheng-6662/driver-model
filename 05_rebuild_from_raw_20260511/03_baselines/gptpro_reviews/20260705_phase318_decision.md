# 本地高级模型第318版决策记录

- 提问文件：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\gptpro_reviews\20260705_phase318_prompt.md`
- 回复文件：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\gptpro_reviews\20260705_phase318_response.md`
- 原始备份：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\gptpro_reviews\20260705_phase318_response_raw_with_prompt.md`
- 页面截图：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\gptpro_reviews\20260705_phase318_after_reply_20260705_073659.png`

## 采纳内容

- 第318版不继续扩展候选库，也不重训第316版主模型；先沿用第316版原预测和第317版候选库，重点修复门控机制。
- 输出默认保持第316版原预测，只有高置信、低退化风险样本才允许离开原预测。
- 门控拆成两段：第一段判断样本是否值得校正，第二段只在通过第一段后选择收益最大且风险最低的候选。
- 候选曲线不直接全量替换原预测，采用小幅残差融合，降低普通样本被过度修改的风险。
- 阈值选择优先使用训练集内部交叉验证结果；验证集只用于通过或失败判定；测试集继续保持隔离。
- 普通样本必须加硬约束：原预测保持率不低于约定下限，普通样本校正率不超过约定上限，整体校正率也需要上限控制。
- 第318版主线采用“三步递进”消融：先做可校正门控，再加候选收益选择，最后加小幅残差融合。

## 暂缓内容

- 不采纳“继续扩大候选库”作为下一步主线，因为第317版已经显示候选最优上限有价值，失败点主要在选择机制。
- 不采纳“每个样本强制选择一个非原始候选”的形式，因为这正是普通样本大面积退化的直接原因。
- 不优先做候选加权平均，除非作为附加消融；若做，也必须给原预测保留足够权重。
- 不在验证失败前报告测试集结果。

## 证据链接

- 第317版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v317_two_stage_candidate_gate_20260704\reports\v317_two_stage_candidate_gate_cn.md`
- 第317版验证分组表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v317_two_stage_candidate_gate_20260704\tables\v317_validation_group_summary.csv`
- 第317版门槛检查表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v317_two_stage_candidate_gate_20260704\tables\v317_validation_gate_check.csv`
- 第317版候选使用表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v317_two_stage_candidate_gate_20260704\tables\v317_validation_candidate_usage.csv`
