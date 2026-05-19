# v0.3 临时加入“锚点后响应弱”样本训练结果
## 为什么做
你复核图片后认为，有些“锚点后响应弱”的样本虽然不像强极限工况，但也可能代表保守驾驶员、小幅维持、弱响应或轻微姿态变化。直接丢掉会让数据集太小，也可能把保守反应这类驾驶行为排除掉。因此本轮不改模型，只把这部分样本临时加入车辆-only 训练，看它到底是补充信息，还是拉乱任务。
## 本轮怎么合并
- 基础范围：干净集四类 + 待复核样本，也就是之前相对更稳的训练范围。
- `FAST_STEER_WEAK_POST_RESPONSE` 候选池共 69 个。
- 其中 53 个本来已经在基础范围内，真正额外新增进入本轮训练的是 16 个。
- 这 16 个额外新增样本在当前旧划分中的分布是：test:3 / train:7 / val:6。
- 另有 `ANCHOR_USABLE_FAST_RESPONSE` 共 1 个，单独做了一个加 1 个样本的对照。
- 继续排除：明显锚点偏晚、锚点后已经稳定的样本，不加入。
- 主要版本去掉 `lateral_distance_selected`，避免横向偏移坐标跳变把任务带偏；另跑一个保留横向偏移的对照。
## 结果表
| variant_id | name_cn | sample_count | extra_episode_count | test_best_model | test_rmse_steer | test_primary_rmse_0_2s | test_tail_rmse_2_5s | test_wrong_side_rate_large | test_severe_amp_under_rate_large | test_large_response_recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v03_plus_review_ref_no_lateral | 干净集 + 待复核（去横向偏移） | 793 | 0 | rbf_kernel_vehicle_context_alpha0.1_g1 | 0.6015 | 0.4546 | 0.6826 | 0.2143 | 0.6786 | 0.6786 |
| v03_plus_review_fast_weakpost_no_lateral | 干净集 + 待复核 + 锚点后响应弱（去横向偏移） | 809 | 16 | rbf_kernel_vehicle_context_alpha0.1_g1 | 0.5977 | 0.4517 | 0.6783 | 0.2105 | 0.7018 | 0.6667 |
| v03_weakpost_usable_nolat | 干净集 + 待复核 + 锚点后响应弱 + 可用快速响应（去横向偏移） | 809 | 16 | rbf_kernel_vehicle_context_alpha0.1_g1 | 0.5977 | 0.4517 | 0.6783 | 0.2105 | 0.7018 | 0.6667 |
| v03_weakpost_with_lateral | 干净集 + 待复核 + 锚点后响应弱（保留横向偏移） | 809 | 16 | rbf_kernel_vehicle_context_alpha0.1_g1 | 0.5889 | 0.4505 | 0.6658 | 0.1930 | 0.7018 | 0.6140 |
## 本轮结论
- 去横向偏移主版本加入 16 个额外弱后续响应样本后，整体 RMSE 从 0.6015 降到 0.5977，属于小幅改善。
- 但它的大响应严重幅值不足率从 0.6786 升到 0.7018，大响应召回从 0.6786 降到 0.6667，说明它没有解决“大幅动作预测太轻”的核心问题。
- 保留横向偏移后 RMSE 进一步降到 0.5889，错侧率也降低，但大响应召回降到 0.6140，所以它更像是全局拟合改善，不一定更符合极限工况物理目标。
- 当前建议：可以暂时保留“锚点后响应弱”作为扩充/保守响应样本池，但不要把它升级为极限姿态核心正样本；下一步应按图片复核结果，把它拆成“弱但有效车辆响应”和“只是轻微方向盘维持”两类。
## 自动读法
- 本轮整体 RMSE 最低的是 `v03_weakpost_with_lateral`，test RMSE=0.5889。
- 额外新增样本在测试集中有 3 个，这几个样本自身 RMSE 聚合约 0.2731；但数量太少，只能作为方向性参考。
- 如果加入锚点后响应弱后 RMSE 下降，同时大响应错侧率、严重幅值不足率不恶化，说明这部分样本可以继续保留。
- 如果 RMSE 下降但大响应物理指标变差，说明它可能只是在普通样本上增加数量，不适合作为极限姿态主训练集。
- 如果保留横向偏移版本明显变差，说明横向偏移坐标风险仍然需要谨慎。
## 可以查看的图
- 每个版本固定样本预测图和坏样本预测图：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_fast_weakpost_temp_train`
- 临时加入的 episode 清单：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_fast_weakpost_temp_train\tables\v03_fast_weakpost_extra_episode_uids.csv`
- 按临时新增来源拆开的测试诊断：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_fast_weakpost_temp_train\tables\v03_fast_weakpost_source_test_diagnostics.csv`
- 汇总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_fast_weakpost_temp_train\tables\v03_fast_weakpost_temp_train_summary.csv`