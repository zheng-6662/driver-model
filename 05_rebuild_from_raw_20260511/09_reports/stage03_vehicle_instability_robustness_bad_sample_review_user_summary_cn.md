# 阶段 3 用户查看版：稳健性坏样本复盘

## 为什么做

前面发现 RBF/KNN 在多个切分和窗口下 RMSE 都能压过 formal ridge，但它们仍可能只是记住了相似模板，或者只改善普通样本。这个阶段专门找“反复失败”的事件。

## 检查了什么

- 每个模型/配置中 RMSE 最高的 top20% 样本。
- 哪些事件在多个模型、多个窗口、多个切分中反复成为坏样本。
- 坏样本里常见的是错侧、幅值不足、尾段漂移、反向修正错误还是多段修正错误。

## 目前发现

复发最高的坏事件是 `vehicle_instability_allraw__hzh__2025_09_26_20_50_27__000337435`，subject=`hzh`，在 15/15 个 config-model 对照中进入 top20 高误差。

## 哪些结果可信

这个复盘只读取已经生成的逐样本指标，不训练新模型，不使用生理、脑电、连续风格或驾驶员 ID。它适合用来决定下一步优先看哪些坏样本。

## 哪些还不能下结论

现在还不能说这些坏样本一定是模型结构问题。它们也可能来自事件锚点偏差、标签窗口没有覆盖完整响应、或原始车辆数据局部异常。必须继续画原始波形和预测曲线确认。

## 下一步

优先对代表坏样本表前 10-20 个事件画详细曲线：事件锚点、车辆姿态、方向盘 GT、RBF/KNN/template 预测；Transformer 只作为已经单独跑过的参照，必要时另行叠加。确认问题来源后，再决定结构化响应模型怎么设计。

## 推荐优先查看

1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_robustness_bad_sample_review_v0_1/tables/robustness_representative_bad_events.csv`
2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_robustness_bad_sample_review_v0_1/tables/robustness_bad_event_recurrence.csv`
3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_robustness_bad_sample_review_v0_1/figures/robustness_recurrent_bad_events.png`
4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_robustness_bad_sample_review_v0_1/figures/robustness_error_flag_heatmap.png`
5. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_robustness_bad_sample_review_v0_1/figures/robustness_bad_event_matrix.png`
