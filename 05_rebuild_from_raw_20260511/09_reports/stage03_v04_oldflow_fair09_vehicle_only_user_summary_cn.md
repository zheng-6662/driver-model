# v0.4 新样本集：旧流程 FAIR09 车辆-only 粗细双头

生成时间：2026-05-20 11:19:54

## 这次对齐的是什么

用户要求对齐截图里旧流程那套模型和参数，而不是使用 `05_rebuild` 里的新 Transformer。

本次采用旧流程车辆-only 版本：

- 版本口径：`FAIR09 / E1`，即“车辆数据 + 粗细双头 + 无显式事件注入”。
- 不使用连续驾驶风格。
- 不使用生理数据。
- 不使用脑电。
- 不使用教师蒸馏。
- 只把 v0.4 新筛出来的样本清单接入旧流程模型。

## 旧流程训练参数

- seed：2026
- device：`cuda`
- epochs：40
- min_epochs：40
- batch_size：64
- lr：1e-3
- weight_decay：0
- grad_clip：1.0
- selection_mode：legacy_rmse
- 模型：历史车辆 Transformer 编码器 + 粗细双头轨迹解码器
- d_model：128
- nhead：2
- encoder layers：2
- decoder layers：2
- ffn_dim：256
- dropout：0.1
- event_embed_dim：96
- event_bin_size：20
- conditioning_mode：vehicle_direct_coarse_fine
- teacher_forcing_ratio：0
- event_loss_weight：0

## 样本来源

- v0.4 主训练候选 + 次级训练候选 + 待复核样本。
- manifest 原始行数：1422
- manifest split：{'train': 823, 'test': 325, 'val': 274}
- 样本来源分布：{'primary': 1128, 'manual_review': 193, 'secondary': 101}
- 旧流程实际可读取：{'status': 'ok', 'manifest_rows': 1422, 'old_loader_kept_rows': 1410, 'old_loader_dropped_rows': 12, 'split_counts_after_old_loader': {'train': 814, 'test': 323, 'val': 273}, 'source_group_counts_after_old_loader': {'primary': 1126, 'manual_review': 183, 'secondary': 101}}

## 当前结果

- test steer RMSE：0.540754
- primary RMSE：0.348139
- tail RMSE：0.478053
- selection：0.984084


## 输出位置

- 旧流程 manifest：`/root/autodl-tmp/data_process/05_rebuild_from_raw_20260511/03_processed_datasets/stage03_v04_oldflow_fair09_vehicle_only/tables/oldflow_fair09_vehicle_only_v04_primary_secondary_review_manifest.csv`
- manifest 检查：`/root/autodl-tmp/data_process/05_rebuild_from_raw_20260511/03_processed_datasets/stage03_v04_oldflow_fair09_vehicle_only/tables/oldflow_fair09_vehicle_only_manifest_validity_check.json`
- 运行目录：`/root/autodl-tmp/data_process/tmp/event_conditioned_runs/V04_OLD_FLOW_FAIR09_vehicle_only_coarse_fine_seed2026_20260520_111848`
- 运行记录：`/root/autodl-tmp/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_v04_oldflow_fair09_vehicle_only/tables/oldflow_fair09_vehicle_only_run_record.csv`
- 预测总览图：`/root/autodl-tmp/data_process/tmp/event_conditioned_runs/V04_OLD_FLOW_FAIR09_vehicle_only_coarse_fine_seed2026_20260520_111848/prediction_figures/test/overview.png`
- 预测图目录：`/root/autodl-tmp/data_process/tmp/event_conditioned_runs/V04_OLD_FLOW_FAIR09_vehicle_only_coarse_fine_seed2026_20260520_111848/prediction_figures/test`
- 逐样本指标：`/root/autodl-tmp/data_process/tmp/event_conditioned_runs/V04_OLD_FLOW_FAIR09_vehicle_only_coarse_fine_seed2026_20260520_111848/prediction_figures/test/prediction_sample_metrics.csv`


## 解释边界

这一步只回答：在新筛样本集上，旧流程“粗细双头车辆-only”能做到什么程度。
它不能证明连续风格、生理数据或脑电有效，也不能直接替代后续更严格的新流程车辆基线。
