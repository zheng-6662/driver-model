# 阶段 3 用户查看版：top-K 车辆-only Transformer v0.1

## 这个阶段为什么做

上一轮 RBF/keypoint 复盘显示两个候选有 oracle 互补空间，但 selector 还不能稳定选择。这个阶段开始做真正的车辆-only 多假设模型：模型一次输出 3 条可能轨迹，并给每条轨迹一个选择概率。

## 这个阶段检查了什么

- top-1：模型自己按概率选出的轨迹。
- best-of-3：事后从 3 条轨迹里选最接近真实的一条，只作为上限诊断。
- RBF 强车辆基线：仍作为主参照。
- 可靠性：top-1 概率、分支分散度和误差是否有关系。

## 目前发现了什么

- RBF：RMSE=0.533667，错侧率=0.225，大幅响应召回=0.750。
- top-1：RMSE=0.587883，错侧率=0.100，大幅响应召回=0.750。
- best-of-3：RMSE=0.477534，错侧率=0.025，大幅响应召回=0.875。
- test 上 top-1 分支与 best-of-3 分支一致率=0.300，平均 top-1 概率=0.383。

## 哪些结果可信

可信的是：这是一个真正车辆-only 的 top-K 模型，输入只含事件前车辆历史和道路/事件上下文；top-1 是可部署策略，best-of-3 只是上限。所有标准化和训练选择都只用 train/val。

## 哪些结果还不能下结论

best-of-3 不能当成可部署结果。若 top-1 没有超过 RBF，就不能说 top-K 车辆-only 已经成为主线；若 best-of-3 明显好但 top-1 不好，只能说明“候选覆盖有潜力，但选择机制还不够”。

## 下一阶段是否可以继续

可以继续阶段 3，但仍不能进入风格、生理或 EEG 增量结论。下一步应根据本轮 top-1/best-of-3 差距，决定是改 selector/可靠性头，还是换成关键点条件的多假设结构。

## 推荐优先查看

1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_vehicle_transformer_v0_1/figures/topk_fixed_predictions_test.png`
2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_vehicle_transformer_v0_1/figures/topk_bad_samples_test.png`
3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_vehicle_transformer_v0_1/figures/topk_metric_summary_test.png`
4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_vehicle_transformer_v0_1/figures/topk_reliability_scatter_test.png`
5. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_vehicle_transformer_v0_1/tables/topk_vehicle_transformer_metrics.csv`
