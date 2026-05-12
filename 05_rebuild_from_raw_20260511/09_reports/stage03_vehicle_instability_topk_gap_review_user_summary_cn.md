# 阶段 3 用户查看版：top-K top1/bestK 差距复盘 v0.1

## 这个阶段为什么做

top-K v0.1 的 best-of-3 很好，但 top-1 没有超过 RBF。这个阶段不训练新模型，只复盘“模型明明有好候选，但为什么 top-1 没选中”的样本和可靠性信号。

## 这个阶段检查了什么

- top-1 分支和 best-of-3 分支是否一致。
- top-1 与 best-of-3 的 RMSE 差距。
- top-1 概率、概率间隔、分支分散度是否能提示风险。
- 差距样本是否集中在某些被试、道路模块、响应类型或物理错误。

## 目前发现了什么

- test top-1 与 best-of-3 一致率=0.300。
- test 平均 top1-bestK gap=0.110531。
- test 高风险分数捕捉高 gap 样本比例=0.545。
- test 低置信规则捕捉高 gap 样本比例=0.636。
- 最大差距样本 `vehicle_instability_allraw__gf__2025_09_26_10_52_57__000300870__pre3_label3_response_coverage` 的 top1-bestK gap=0.447251。

## 哪些结果可信

可信的是：top-K v0.1 的主要瓶颈不是完全没有候选，而是选择头和可靠性判断不足；这由逐样本 bestK、top1 分支、概率和物理错误共同支持。

## 哪些结果还不能下结论

不能说当前可靠性规则已经可部署。这里的简单风险分数只是诊断线索，下一步如果要使用，必须在 train/val 上固定规则后再 test 评价，不能用 test 重新调参。

## 下一阶段是否可以继续

可以继续阶段 3，建议下一步做“可靠性/选择头 v0.2”或“关键点条件多假设”。仍不能进入连续风格、生理或 EEG 增量结论。

## 推荐优先查看

1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_gap_review_v0_1/figures/topk_gap_top_samples.png`
2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_gap_review_v0_1/figures/topk_gap_risk_scatter.png`
3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_gap_review_v0_1/figures/topk_gap_branch_confusion.png`
4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_gap_review_v0_1/figures/topk_gap_error_flags.png`
5. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_gap_review_v0_1/tables/topk_gap_review_sample_detail.csv`
