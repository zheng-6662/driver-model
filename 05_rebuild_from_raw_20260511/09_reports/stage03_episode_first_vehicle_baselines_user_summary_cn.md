# episode-first v0.6 纯车辆/道路预测对照 v0.1

## 为什么做

这一步不是验证生理或连续风格，而是先检查新筛出来的 episode 样本能不能让纯车辆/道路预测任务更清楚。核心问题是：如果样本和锚点本身更合理，车辆-only 基线至少应该在任务定义、分层和物理错误解释上更清楚；否则继续加生理数据也容易变成补偿错样本。

## 检查了什么

- 输入样本来自 `episode-first` v0.6。
- 正样本轨道使用“严格核心 + 坐标需复核扩展候选”，共 265 个事件。
- 主对照是 3 秒标签、不使用横向偏移特征的轨道，避免道路坐标跳变污染模型。
- 额外保留一个“使用横向偏移特征”的 3 秒轨道，只用于判断坐标特征是否虚高。
- 模型仍然只使用车辆历史和道路/事件上下文，不使用生理、脑电、连续风格、驾驶员 ID 或服务器。

## 样本轨道

```text
                    track_id              window_config_id  zero_lateral_offset_feature   n  train_n  val_n  test_n  strict_clean_n  coordinate_flagged_n                                                   module_counts_json                               description_cn
  EP2_expanded_no_lateral_2s          pre2_label2_old_main                         True 265      183     37      45              19                   246 {"curve1": 97, "fix_road": 69, "differentmu_road": 56, "curve2": 43}         episode-first 正样本扩展集，2秒标签，不使用横向偏移特征。
  EP3_expanded_no_lateral_3s pre3_label3_response_coverage                         True 265      183     37      45              19                   246 {"curve1": 97, "fix_road": 69, "differentmu_road": 56, "curve2": 43}         episode-first 正样本扩展集，3秒标签，不使用横向偏移特征。
EP3_expanded_with_lateral_3s pre3_label3_response_coverage                        False 265      183     37      45              19                   246 {"curve1": 97, "fix_road": 69, "differentmu_road": 56, "curve2": 43} episode-first 正样本扩展集，3秒标签，保留横向偏移特征；仅作坐标风险诊断。
```

## 当前结果

旧 B 轨道 RBF KRR：test RMSE=0.533667，错侧率=0.225000，大幅响应召回=0.750000。

本轮按验证集选择模型后的 test 结果：

- EP2_expanded_no_lateral_2s：val 选择 `ridge_rich_context_no_subject`；test RMSE=0.603605，错侧率=0.355556，大幅响应召回=0.000000，严重幅值不足率=0.400000。
- EP3_expanded_no_lateral_3s：val 选择 `formal_ridge_vehicle_context_no_subject`；test RMSE=0.679927，错侧率=0.266667，大幅响应召回=0.250000，严重幅值不足率=0.355556。
- EP3_expanded_with_lateral_3s：val 选择 `formal_ridge_vehicle_context_no_subject`；test RMSE=0.680265，错侧率=0.288889，大幅响应召回=0.250000，严重幅值不足率=0.355556。

## 当前判断

本轮 episode-first 扩展正样本没有让纯车辆/道路预测指标超过旧 B 轨道。3 秒、不使用横向偏移的主轨道 test RMSE=0.679927，明显高于旧 B 轨道 RBF KRR 的 0.533667；大幅响应召回也从旧 B 的 0.750000 降到 0.250000。保留横向偏移特征并没有变好，说明这次结果不是因为我们屏蔽横向偏移导致的简单退化。

这个结果不能说明 v0.6 筛错了，反而说明 episode-first 正样本更集中在真实的大幅响应、回正和复杂修正片段上，车辆-only 线性/模板类模型更难处理。当前可以说：新筛样本在语义上更接近目标事件，但尚未带来车辆-only 预测提升；下一步如果继续建模，应优先做响应分解或结构化模型，而不是马上加连续风格/生理去补偿。

## 推荐查看

1. `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_episode_first_vehicle_baselines_v0_1\tables\episode_first_vehicle_metrics.csv`
2. `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_episode_first_vehicle_baselines_v0_1\tables\episode_first_vehicle_val_selected_models.csv`
3. `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_episode_first_vehicle_baselines_v0_1\tables\episode_first_track_summary.csv`
4. `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_episode_first_vehicle_baselines_v0_1\figures\episode_first_vehicle_metric_summary_test.png`
5. `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_episode_first_vehicle_baselines_v0_1\figures\EP3_expanded_no_lateral_3s_bad_samples_test.png`
