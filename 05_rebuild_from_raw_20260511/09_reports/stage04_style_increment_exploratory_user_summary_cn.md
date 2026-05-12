# 阶段 4 用户查看版：连续驾驶风格探索性增量对照 v0.1

## 这个阶段为什么做

前面已经把 B 轨道 270 个严格失稳响应样本、3 秒标签窗口和 RBF/KNN 类车辆-only 对照固定下来。这一步不是直接证明风格有效，而是先问一个更小的问题：在固定 RBF 车辆-only 预测之后，事件前 60 秒或更长的连续驾驶风格，能不能解释一部分剩余误差。

## 这个阶段检查了什么

- 主任务只用 B 轨道 `response3s_strict_core_candidate`，共 270 个样本，test 40 个。
- 主参照固定为 `rbf_kernel_ridge_context_no_subject`，不再用 Transformer 作为当前主参照。
- 连续风格只来自事件前，并且与直接车辆输入窗口 `[-3, 0]` 和标签窗口 `[0, 3]` 不重叠。
- 对照包括：RBF、RBF+last60 风格、RBF+全部风格窗口、RBF+驾驶员 ID、RBF+道路模块、RBF+风格+驾驶员 ID，以及多种置乱风格。

## 目前发现了什么

- RBF test RMSE：0.533667
- RBF+last60 风格 test RMSE：0.534559
- RBF+全部风格窗口 test RMSE：0.564153
- RBF+驾驶员 ID test RMSE：0.533661
- RBF+last60 风格+驾驶员 ID test RMSE：0.534558

物理指标上还要看错侧率、大幅响应召回、困难样本 RMSE 和坏样本图，不能只看 RMSE。

## 哪些结果可信

可信的是：本轮风格特征来源是事件前，标准化参数只来自 train split，评估使用固定 RBF 参照，并且加入了驾驶员 ID 与置乱对照。可信范围是“探索性增量对照”，不是最终有效性结论。

## 哪些结果还不能下结论

还不能说连续风格有效，也不能说它提供了跨被试泛化信息。因为当前只完成 session-level split，没有完成 subject-level 或留一被试验证；如果收益和驾驶员 ID 接近，也可能只是身份或道路分布代理。

## 下一阶段是否可以继续

可以继续做阶段 4 的更严格验证，但生理和 EEG 仍然不进入。下一步应该补 subject-level/跨 session 风格对照，并用固定预测图和坏样本图确认收益是不是来自真实物理错误改善。

## 推荐优先查看

1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_increment_exploratory_v0_1/figures/style_increment_metric_summary_test.png`
2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_increment_exploratory_v0_1/figures/style_increment_fixed_predictions_test.png`
3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_increment_exploratory_v0_1/figures/style_increment_bad_samples_test.png`
4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_increment_exploratory_v0_1/tables/style_increment_gate_table.csv`
5. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_increment_exploratory_v0_1/tables/style_increment_metrics.csv`
