# 阶段 3 用户查看版：top-K 可靠性选择/回退 v0.1

## 这个阶段为什么做

上一轮 top-K 的 best-of-3 很好，但 top-1 选不中：test 上 top-1 和 best 分支一致率只有 0.300。这个阶段不重新训练轨迹模型，而是检查“能不能用车辆-only、事件前可得的信息，把 top-K 的候选分支选得更好，或者在不可靠时回退到 RBF/KNN 类强车辆基线”。

## 这个阶段检查了什么

- `branch_logreg`：只在 3 条 top-K 分支里选一条。
- `candidate_logreg`：在 RBF 与 3 条 top-K 分支之间直接选择。
- `top1_rbf_fallback`：先用 top-K 自己的 top-1，若可靠性模型认为 top-1 会比 RBF 差，则回退到 RBF。
- `best-of-3` 和 `best-of-RBF+topK` 只作为事后上限，不作为可部署结果。

## 目前发现了什么

- RBF test RMSE=0.533667，错侧率=0.225，大幅响应召回=0.750。
- top-K top-1 test RMSE=0.587865，错侧率=0.100，大幅响应召回=0.750。
- 按 validation RMSE 选中的可靠性策略是 `topk_top1_rbf_fallback_logreg`，test RMSE=0.542071，错侧率=0.225，大幅响应召回=0.750。
- `topk_top1_rbf_fallback_logreg` 在 test 上的选择来源计数：{'rbf': 39, 'top1': 1}。
- 结论：validation 选中的策略没有超过 RBF，test RMSE 比 RBF 高 0.008405；本轮不能升级为强车辆基线。
- best-of-RBF+topK 上限 test RMSE=0.415652，说明候选池仍有明显潜力，但选择机制还没有完全吃到。

## 哪些结果可信

可信的是：本轮选择器只用 train 训练，`top1_rbf_fallback` 的阈值只用 val 固定；输入特征只来自事件前车辆/道路上下文、候选模型自己的预测形态和 top-K 概率，不使用 subject ID、生理、脑电、连续风格，也不把 test 标签用于训练标准化或阈值选择。

## 哪些结果还不能下结论

不能把 best-of-RBF+topK 当成真实部署性能；它是事后知道哪条轨迹最接近真值的上限。若 validation 选中的策略 test 仍不能稳定超过 RBF，就不能说 top-K 可靠性选择已经解决问题，只能说“候选池有潜力，选择头仍需改进”。

## 下一阶段是否可以继续

可以继续阶段 3，但仍不进入风格、生理或 EEG 有效性结论。下一步应把可靠性选择作为诊断结果，决定是否做关键点条件多假设、分响应类型的选择头，或者回到更稳的 RBF/KNN 类强车辆基线作为暂定主参照。

## 推荐优先查看

1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_reliability_selector_v0_1/figures/topk_reliability_selector_metric_summary_test.png`
2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_reliability_selector_v0_1/figures/topk_reliability_selector_fixed_predictions_test.png`
3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_reliability_selector_v0_1/figures/topk_reliability_selector_bad_samples_test.png`
4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_reliability_selector_v0_1/figures/topk_reliability_selector_decision_counts_test.png`
5. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_reliability_selector_v0_1/figures/topk_reliability_selector_fallback_scatter_test.png`
