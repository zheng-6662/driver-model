# v2.0 待复核样本纳入训练：车辆-only GPU 基线

## 这次为什么做

用户提出：待复核样本不一定都应该排除，其中可能有很多可训练片段。因此本轮在 v2.0 无历史继承重审样本上，把待复核样本也纳入训练，检查它是否能扩大数据覆盖并改善车辆-only 预测。

本轮仍然只训练车辆-only，不加入连续风格、生理数据、脑电或教师蒸馏。

## 运行设置

- 设备：`cuda`，本地 CUDA。
- 主锚点：`model_anchor_s_v1_8`。
- 输入：锚点前 2 秒车辆历史，20 Hz。
- 标签：锚点后 5 秒方向盘相对变化，20 Hz。
- 划分：test=`cwh/gf/tyy`，val=`byx/gzj/yyl`，其余被试为 train。
- 对照：全量训练候选 + 待复核、非弯道训练候选 + 非弯道待复核、弯道训练候选 + 弯道待复核。

## 结果表

| variant_id | name_cn | sample_count | val_selected_model | test_rmse_steer | test_primary_rmse_0_2s | test_tail_rmse_2_5s | test_wrong_side_rate_large | test_severe_amp_under_rate_large | test_large_response_recall | screening_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v20_train_review_anchor_nolat | v2.0 训练候选 + 全部待复核，推荐锚点，去横向偏移 | 1092.0000 | torch_mlp256_vehicle_context | 0.3842 | 0.3134 | 0.4267 | 0.3333 | 0.5000 | 1.0000 | 0.0000 |
| v20_train_review_anchor_lat | v2.0 训练候选 + 全部待复核，推荐锚点，保留横向偏移 | 1092.0000 | torch_mlp256_vehicle_context | 0.3773 | 0.3126 | 0.4160 | 0.3333 | 0.5000 | 0.8333 | -0.0181 |
| v20_noncurve_train_review_anchor_nolat | v2.0 非弯道训练候选 + 非弯道待复核，推荐锚点，去横向偏移 | 767.0000 | torch_mlp512_vehicle_context | 0.3810 | 0.3437 | 0.4052 | 0.8333 | 0.5000 | 1.0000 | -0.1719 |
| v20_curve_train_review_anchor_nolat | v2.0 弯道训练候选 + 弯道待复核，推荐锚点，去横向偏移 | 381.0000 | torch_mlp256_vehicle_context | 0.5456 | 0.4177 | 0.6167 | 0.5000 | 0.5000 | 1.0000 | -0.2197 |

## 当前读法

- 综合排序第一：`v20_train_review_anchor_nolat`，test RMSE=0.3842，大响应错侧率=0.3333，严重幅值不足率=0.5000。
- 单看整体 RMSE 最低：`v20_train_review_anchor_lat`，test RMSE=0.3773。
- 如果加待复核后整体指标和物理指标同时改善，说明待复核样本里确实有可用训练信息。
- 如果只改善 RMSE 但恶化错侧、幅值或大响应召回，说明待复核样本会让任务平均化，后续需要继续分层纳入。

## 产物位置

- 汇总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v20_review_inclusion_vehicle_only_gpu\tables\v20_review_inclusion_vehicle_only_gpu_summary.csv`
- 排名表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v20_review_inclusion_vehicle_only_gpu\tables\v20_review_inclusion_vehicle_only_gpu_ranking.csv`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v20_review_inclusion_vehicle_only_gpu`