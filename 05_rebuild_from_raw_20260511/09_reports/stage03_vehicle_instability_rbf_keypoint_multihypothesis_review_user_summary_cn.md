# 阶段 3 用户查看版：RBF/keypoint 多候选车辆-only 复盘 v0.1

## 这个阶段为什么做

上一轮发现 RBF 整体 RMSE 稳，但 keypoint+residual 能修复一部分错侧和大幅响应。这个阶段不训练新模型，只把 RBF、keypoint、train/val selector 和 oracle best-of-two 放到同一套表和图里，判断“多候选车辆-only”是否值得继续。

## 这个阶段检查了什么

- RBF、keypoint、selector、oracle 在同一 test 集上的整体误差和物理指标。
- selector 什么时候选对，什么时候选错。
- oracle 上限和可部署 selector 之间还有多大差距。
- 固定样本图、selector 坏样本图和 oracle 增益图。

## 目前发现了什么

- RBF：RMSE=0.533667，错侧率=0.225，大幅响应召回=0.750。
- keypoint：RMSE=0.548993，错侧率=0.125，大幅响应召回=0.875。
- selector：RMSE=0.533912，错侧率=0.200，大幅响应召回=0.875。
- oracle best-of-two：RMSE=0.475095，这是事后上限，不能部署，但说明两个候选确实互补。
- test 上 selector 选择准确率=0.550，平均选择后悔=0.059123。

## 哪些结果可信

可信的是：RBF 和 keypoint 在同一数据、同一 split、同一评价指标下确实有互补；selector 目前能改善错侧、大幅响应召回和困难 top20 RMSE，但整体 RMSE 还没有稳定超过 RBF。

## 哪些结果还不能下结论

不能把 oracle 当成真实模型效果；也不能说车辆-only 已经解决，更不能据此进入连续风格、生理或 EEG 有效性结论。当前只说明多候选方向值得继续，但 selector 还需要更强的可靠性特征或结构。

## 下一阶段是否可以继续

可以继续阶段 3，但方向应是正式多假设/可靠性车辆-only，而不是直接加生理。下一步需要让模型自己输出多个候选和可靠性，而不是只在两个已训练候选之间做简单二选一。

## 推荐优先查看哪些图和表

1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_multihypothesis_review_v0_1/figures/multihypothesis_fixed_predictions_test.png`
2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_multihypothesis_review_v0_1/figures/multihypothesis_selector_bad_samples_test.png`
3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_multihypothesis_review_v0_1/figures/multihypothesis_oracle_gap_samples_test.png`
4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_multihypothesis_review_v0_1/tables/multihypothesis_metrics.csv`
5. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_multihypothesis_review_v0_1/tables/test_selector_misselected_samples.csv`
