# v235 删除 observe_later_like 样本后的受控重训报告

## 结论

- 本轮把 v234 标记的 observe_later_like 样本从 train/val/test 全部剔除后，重新训练 v222a light prediction layer。
- 模型选择仍只使用过滤后的 validation split；test 只在选型固定后报告。
- 这是校准/融合层重训，不是 v216/v218 底座候选网络的端到端重训。
- 因此本轮适合判断“删掉这类样本是否让剩余任务更稳定”，不应直接当作最终正式榜单。

## 删除规模

- loose_main_pool/all: 原始 1167，删除 121，保留 1046，删除比例 0.104
- loose_main_pool/test: 原始 184，删除 27，保留 157，删除比例 0.147
- loose_main_pool/train: 原始 674，删除 58，保留 616，删除比例 0.086
- loose_main_pool/val: 原始 309，删除 36，保留 273，删除比例 0.117
- strict_main_pool/all: 原始 963，删除 117，保留 846，删除比例 0.121
- strict_main_pool/test: 原始 174，删除 27，保留 147，删除比例 0.155
- strict_main_pool/train: 原始 519，删除 57，保留 462，删除比例 0.110
- strict_main_pool/val: 原始 270，删除 33，保留 237，删除比例 0.122

## Validation-selected 模型

- loose_main_pool: `v235_filtered_bounded_residual_global_blend_a10p0_b0p1`，variant=filtered_bounded_residual，filtered val score=0.837741；filtered test RMSE 0.482685 -> 0.474318，tail 0.560086 -> 0.544057
- strict_main_pool: `v235_filtered_bounded_residual_global_blend_a10p0_b0p2`，variant=filtered_bounded_residual，filtered val score=0.872760；filtered test RMSE 0.506547 -> 0.504151，tail 0.594965 -> 0.592674

## 关键对照

- loose_main_pool: 旧模型 full test RMSE=0.555940；旧模型删除后同一 test 子集 RMSE=0.482685；删除后重训 RMSE=0.474318；重训相对旧过滤子集 delta=-0.008367
- strict_main_pool: 旧模型 full test RMSE=0.571966；旧模型删除后同一 test 子集 RMSE=0.506547；删除后重训 RMSE=0.504151；重训相对旧过滤子集 delta=-0.002397

## 被删除样本上的诊断

- loose_main_pool: removed test n=27，重训模型 RMSE=0.868780，tail=1.054962，under=0.407407
- strict_main_pool: removed test n=27，重训模型 RMSE=0.845273，tail=1.034791，under=0.333333

## 输出

- `tables/v235_comparison_summary.csv`：主对照表。
- `tables/v235_selected_metrics_filtered.csv`：删除后重训模型在保留样本上的指标。
- `tables/v235_old_selected_metrics_filtered.csv`：旧 v222a selected 模型在同一保留样本上的指标。
- `tables/v235_selected_metrics_removed_holdout.csv`：删除后重训模型在被删除样本上的诊断指标。
- `figures/v235_test_rmse_comparison.png` 与 `figures/v235_test_tail_rmse_comparison.png`：test 对比图。
- ZIP：`v235_remove_observe_later_retrain_pack.zip`
