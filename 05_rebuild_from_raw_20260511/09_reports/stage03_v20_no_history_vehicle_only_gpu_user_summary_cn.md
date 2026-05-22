# v2.0 全量无历史继承重审样本：车辆-only GPU 基线

## 这次训练的是什么模型

这次不是训练连续驾驶风格模型，也不是训练生理/脑电模型。它是一个车辆-only 诊断基线：只使用车辆历史和事件上下文字段，验证 v2.0 新样本定义本身是否更适合建模。

具体模型集合包括：

- 无学习基线：零变化、训练集同类均值、训练集全局均值、历史趋势外推；
- 线性头：把锚点前车辆历史压平成特征后直接预测后续方向盘相对轨迹；
- 小型多层感知机：同样使用车辆历史特征，但允许非线性关系。

最终按验证集 RMSE 选择一个模型，再报告测试集表现。因此这一步的目的不是最终刷分，而是先看 v2.0 样本能否让车辆-only 模型学到更合理的方向盘后续变化。

## 运行设置

- 设备：`cuda`，本地 CUDA。
- 样本入口：`record_level_episodes_all_v2_0.csv`。
- 历史标签使用方式：不参与样本选择，只作为审计字段保留。
- 主训练锚点：`model_anchor_s_v1_8`。这个锚点比原始 episode_start 更接近前面已讨论过的“去除过长平稳前奏后”的模型锚点。
- 输入窗口：锚点前 2 秒车辆历史，20 Hz。
- 标签窗口：锚点后 5 秒方向盘相对变化，20 Hz。
- 划分：test=`cwh/gf/tyy`，val=`byx/gzj/yyl`，其余被试为 train。
- 主比较版本：全量训练候选去横向偏移、全量训练候选保留横向偏移、非弯道候选、弯道候选。

## 结果表

| variant_id | name_cn | sample_count | val_selected_model | test_rmse_steer | test_primary_rmse_0_2s | test_tail_rmse_2_5s | test_wrong_side_rate_large | test_severe_amp_under_rate_large | test_large_response_recall | screening_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v20_noncurve_train_anchor_nolat | v2.0 非弯道训练候选，推荐锚点，去横向偏移 | 581.0000 | torch_mlp512_vehicle_context | 0.3906 | 0.3523 | 0.4152 | 0.0000 | 0.5000 | 0.5000 | 0.3916 |
| v20_curve_train_anchor_nolat | v2.0 弯道训练候选，推荐锚点，去横向偏移 | 169.0000 | torch_linear_vehicle_context | 0.3092 | 0.2843 | 0.3252 | 0.0000 | 1.0000 | 0.0000 | 0.2729 |
| v20_all_train_anchor_lat | v2.0 全量训练候选，推荐锚点，保留横向偏移 | 750.0000 | torch_mlp512_vehicle_context | 0.3807 | 0.3523 | 0.4025 | 0.6000 | 0.6000 | 0.4000 | 0.1514 |
| v20_all_train_anchor_nolat | v2.0 全量训练候选，推荐锚点，去横向偏移 | 750.0000 | torch_mlp512_vehicle_context | 0.3921 | 0.3459 | 0.4212 | 1.0000 | 0.6000 | 0.4000 | 0.0000 |

## 当前读法

- 综合排序第一：`v20_noncurve_train_anchor_nolat`，test RMSE=0.3906，大响应错侧率=0.0000，严重幅值不足率=0.5000。
- 单看整体 RMSE 最低：`v20_curve_train_anchor_nolat`，test RMSE=0.3092。
- 如果全量候选明显优于非弯道/弯道单独候选，说明 v2.0 的合并样本池更适合先做统一模型。
- 如果非弯道或弯道单独候选更好，说明后续应考虑按道路/工况分开建模。
- 如果保留横向偏移只改善部分物理指标但恶化 RMSE，需要继续按道路坐标质量分层使用，不应直接全局加入。

## 产物位置

- 汇总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v20_no_history_vehicle_only_gpu_baseline\tables\v20_no_history_vehicle_only_gpu_summary.csv`
- 排名表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v20_no_history_vehicle_only_gpu_baseline\tables\v20_no_history_vehicle_only_gpu_ranking.csv`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v20_no_history_vehicle_only_gpu_baseline`