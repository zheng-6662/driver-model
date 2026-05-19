# v0.3 样本筛选策略 GPU 快筛

## 为什么改成 GPU

之前的连续筛选脚本沿用 sklearn 核岭回归，默认只能走 CPU。它适合做传统基线，但连续扫描十几个样本策略太慢。本轮改用 PyTorch 车辆-only 小网络，在 GPU 上训练同一结构，用来快速判断哪些样本筛选方向值得继续。

注意：这张表用于筛选样本方向，不和旧 sklearn 核岭回归表直接混作同一模型结论。

## 运行设置

- 设备：`cuda`。
- 输入：车辆历史 + 事件/上下文表格特征，不含连续风格、生理或脑电。
- 模型：线性头、256 隐层网络、512 隐层网络；按验证集 RMSE 选模型，再报告测试集。

## 基础版本

- `s00_base_nolat` 样本数 793，test RMSE=0.6376，大响应错侧率=0.2692，严重幅值不足率=0.4038，大响应召回=0.9231。

## 排名前 12 的筛选策略

| variant_id | name_cn | sample_count | extra_episode_count | val_selected_model | test_rmse_steer | delta_rmse_vs_base | test_wrong_side_rate_large | test_severe_amp_under_rate_large | test_large_response_recall | screening_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| s16_weakpost_lat | 加锚点后响应弱，保留横向偏移 | 809.0000 | 16.0000 | torch_mlp256_vehicle_context | 0.6446 | 0.0071 | 0.2453 | 0.3019 | 0.9245 | 0.0270 |
| s04_fast_all_nonlate_nolat | 加全部非偏晚快速转向候选，去横向偏移 | 809.0000 | 16.0000 | torch_mlp256_vehicle_context | 0.6487 | 0.0111 | 0.2830 | 0.3396 | 0.9245 | 0.0003 |
| s00_base_nolat | 基础：干净集 + 待复核，去横向偏移 | 793.0000 | 0.0000 | torch_mlp256_vehicle_context | 0.6376 | 0.0000 | 0.2692 | 0.4038 | 0.9231 | 0.0000 |
| s02_fast_visible_nolat | 加快速转向且车辆响应可见，去横向偏移 | 798.0000 | 5.0000 | torch_mlp256_vehicle_context | 0.6528 | 0.0153 | 0.2692 | 0.3654 | 0.9423 | -0.0028 |
| s03_fast_visible_boundary_nolat | 加快速转向可见/边界车辆响应，去横向偏移 | 804.0000 | 11.0000 | torch_mlp256_vehicle_context | 0.6524 | 0.0148 | 0.3077 | 0.3462 | 0.9808 | -0.0052 |
| s08_keep_weak_nolat | 加弱/保守响应，去横向偏移 | 793.0000 | 0.0000 | torch_mlp512_vehicle_context | 0.6596 | 0.0220 | 0.2885 | 0.3462 | 0.9615 | -0.0085 |
| s10_lowmu_excl_nolat | 加低附着 excluded，去横向偏移 | 1203.0000 | 420.0000 | torch_mlp256_vehicle_context | 0.7047 | 0.0672 | 0.1667 | 0.4111 | 0.9556 | -0.0282 |
| s09_keep_delay_nolat | 加延迟/无明显转向响应，去横向偏移 | 793.0000 | 0.0000 | torch_mlp512_vehicle_context | 0.6735 | 0.0359 | 0.2692 | 0.4231 | 0.9615 | -0.0350 |
| s11_roll_excl_nolat | 加横滚/姿态 excluded，去横向偏移 | 1092.0000 | 300.0000 | torch_mlp256_vehicle_context | 0.7341 | 0.0965 | 0.2143 | 0.2500 | 0.9464 | -0.0353 |
| s01_weakpost_nolat | 加锚点后响应弱，去横向偏移 | 809.0000 | 16.0000 | torch_mlp256_vehicle_context | 0.6419 | 0.0044 | 0.3396 | 0.4340 | 0.9245 | -0.0363 |
| s05_keep_extreme_nolat | 加新规则核心极限样本，去横向偏移 | 793.0000 | 0.0000 | torch_mlp256_vehicle_context | 0.6493 | 0.0118 | 0.3654 | 0.3846 | 0.9423 | -0.0377 |
| s17_roll_excl_lat | 加横滚/姿态 excluded，保留横向偏移 | 1092.0000 | 300.0000 | torch_mlp512_vehicle_context | 0.7565 | 0.1189 | 0.1786 | 0.2143 | 0.9107 | -0.0417 |

## 自动读法

- 按综合分数，最好的是 `s16_weakpost_lat`，test RMSE=0.6446，综合分数=0.0270。
- 单看整体 RMSE，最低的是 `s00_base_nolat`，test RMSE=0.6376。
- 如果 RMSE 降低但大响应召回/严重幅值不足恶化，说明它更像普通拟合改善，不一定适合作为极限主样本。
- 如果物理指标改善但 RMSE 不占优，可以考虑作为极限姿态专用样本集，而不是和普通样本混训。

## 产物位置

- 汇总表：`/root/autodl-tmp/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_v03_screening_sweep_gpu/tables/v03_screening_sweep_gpu_summary.csv`
- 排名表：`/root/autodl-tmp/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_v03_screening_sweep_gpu/tables/v03_screening_sweep_gpu_ranking.csv`
- 输出目录：`/root/autodl-tmp/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_v03_screening_sweep_gpu`