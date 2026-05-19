# v0.4 重筛样本车辆-only GPU 基线

## 这次为什么做

这次不是继续比较旧的 809 样本，而是使用 v0.4 从 1574 个初始片段重新筛出的样本。v0.4 的核心规则是：锚点后车辆有变化就保留，即使驾驶员操作弱；锚点后车辆和驾驶员都弱就排除；快打方向但车辆变化弱先谨慎处理。

本轮仍然只跑车辆-only，不加入连续驾驶风格、生理或脑电。目的只是检查 v0.4 样本定义本身是否更适合建模。

## 运行设置

- 设备：`cuda`，本地 GPU。
- 输入：车辆历史 + 事件/工况上下文字段。
- 标签：锚点后的方向盘相对轨迹。
- 模型：无学习基线 + PyTorch 线性头/小型神经网络；按验证集 RMSE 选模型，再报告测试集。

## 结果表

| variant_id | name_cn | sample_count | val_selected_model | test_rmse_steer | test_primary_rmse_0_2s | test_tail_rmse_2_5s | test_wrong_side_rate_large | test_severe_amp_under_rate_large | test_large_response_recall | screening_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v04_primary_secondary_nolat | v0.4 主+次级候选，去横向偏移 | 956.0000 | torch_mlp512_vehicle_context | 0.8402 | 0.6343 | 0.9536 | 0.1707 | 0.2683 | 0.9512 | 0.0363 |
| v04_primary_secondary_lat | v0.4 主+次级候选，保留横向偏移 | 956.0000 | torch_mlp256_vehicle_context | 0.8405 | 0.6304 | 0.9560 | 0.2073 | 0.3293 | 0.9512 | 0.0079 |
| v04_primary_nolat | v0.4 主训练候选，去横向偏移 | 855.0000 | torch_mlp256_vehicle_context | 0.8794 | 0.6716 | 0.9956 | 0.1688 | 0.2727 | 0.9740 | 0.0000 |
| v04_primary_lat | v0.4 主训练候选，保留横向偏移 | 855.0000 | torch_mlp512_vehicle_context | 0.8995 | 0.6811 | 1.0210 | 0.1169 | 0.2727 | 0.9740 | -0.0019 |

## 当前读法

- 综合排序第一：`v04_primary_secondary_nolat`，test RMSE=0.8402，综合分数=0.0363。
- 单看整体 RMSE 最低：`v04_primary_secondary_nolat`，test RMSE=0.8402。
- 如果“主+次级”比“主训练候选”更好，说明次级样本有助于扩充任务覆盖；如果变差，说明次级样本仍需要继续分层。
- 如果保留横向偏移改善物理指标但恶化 RMSE，说明横向偏移可能更像极限姿态提示，而不是稳定的通用轨迹输入。

## 产物位置

- 汇总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v04_vehicle_only_gpu_baseline\tables\v04_vehicle_only_gpu_summary.csv`
- 排名表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v04_vehicle_only_gpu_baseline\tables\v04_vehicle_only_gpu_ranking.csv`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v04_vehicle_only_gpu_baseline`