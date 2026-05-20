# v1.1 完整记录级样本车辆-only GPU 基线

## 这次为什么做

v1.1 是从完整原始车辆记录中重建 episode 后，经人工复核整理出的主训练候选。它不再直接继承旧 `.aed`、道路入口或旧锚点。本轮只训练车辆-only，目的是先看新样本定义本身是否能让车辆模型学到更稳定的方向盘后续变化。

## 运行设置

- 设备：`cuda`，CUDA。不要加入连续风格、生理或脑电。
- 样本入口：`train_candidate_extreme_episodes_v1_1.csv`。
- 切分：test=`cwh/gf/tyy`，val=`byx/gzj/yyl`，其余被试为 train，用于和 v0.5 新样本阶段保持同类被试划分逻辑。
- 输入：锚点前 2 秒车辆历史，20 Hz。
- 标签：锚点后 5 秒方向盘相对变化，20 Hz。
- 模型：无学习基线 + PyTorch 线性头/小型神经网络；按验证集 RMSE 选模型，再报告测试集。

## 结果表

| variant_id | name_cn | sample_count | val_selected_model | test_rmse_steer | test_primary_rmse_0_2s | test_tail_rmse_2_5s | test_wrong_side_rate_large | test_severe_amp_under_rate_large | test_large_response_recall | screening_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v11_vehicle_onset_nolat | v1.1 车辆响应开始锚点，去横向偏移 | 1065.0000 | torch_mlp256_vehicle_context | 0.3532 | 0.3186 | 0.3751 | 0.0000 | 0.5000 | 0.5000 | 0.3253 |
| v11_episode_start_lat | v1.1 episode 开始锚点，保留横向偏移 | 1065.0000 | torch_mlp256_vehicle_context | 0.3499 | 0.3217 | 0.3683 | 0.2727 | 0.6364 | 0.5455 | 0.2059 |
| v11_episode_start_nolat | v1.1 episode 开始锚点，去横向偏移 | 1065.0000 | train_context_mean | 0.3194 | 0.2951 | 0.3362 | 0.4545 | 1.0000 | 0.0000 | 0.0000 |

## 当前读法

- 综合排序第一：`v11_vehicle_onset_nolat`，test RMSE=0.3532，大响应错侧率=0.0000，严重幅值不足率=0.5000。
- 单看整体 RMSE 最低：`v11_episode_start_nolat`，test RMSE=0.3194。
- 这次不能只看整体 RMSE。`episode 开始锚点，去横向偏移` 的 RMSE 最低，但验证集选中的是上下文均值模型，说明它更像把曲线平均化；它的大响应召回为 0，严重幅值不足率为 1.0，不符合极限工况建模目标。
- `车辆响应开始锚点，去横向偏移` 的 RMSE 更高，但大响应错侧率最低，说明它更接近“方向不乱判”的物理目标；不过幅值仍明显不足，不能直接作为最终方案。
- 因此本轮结论是：v1.1 样本能训练，但车辆-only 仍没有真正学好极限样本的幅值和形态。下一步应先继续改任务定义或输出形式，不急着加入风格/生理。
- 如果车辆响应开始锚点明显好于 episode 开始锚点，说明当前 episode 开始点可能仍偏早或任务中包含较多前奏；如果 episode 开始点更好，说明它更适合预测完整后续变化。
- 如果保留横向偏移改善错侧但恶化 RMSE，说明横向偏移可能是局部强提示，后续应按场景或质量分层使用。

## 图和表

- 汇总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v11_vehicle_only_gpu_baseline\tables\v11_vehicle_only_gpu_summary.csv`
- 排名表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v11_vehicle_only_gpu_baseline\tables\v11_vehicle_only_gpu_ranking.csv`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v11_vehicle_only_gpu_baseline`
