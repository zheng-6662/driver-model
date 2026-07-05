# v222a 轻量软融合与受限残差报告

## 结论

- 已在固定 formal 候选池上完成非负凸融合和 bounded residual 校准。
- 所有模型只在 train 拟合，最终输出只按 validation selection score 选择。
- test 指标仅对 validation-selected 输出和固定 baseline 参考项报告。
- 未使用 `W3_B4_original_soft`、oracle、true-label fallback 或 diagnostic-only row。

## Validation-selected 输出

- loose_main_pool: `v222a_bounded_residual_global_blend_a1p0_b0p2`，variant=bounded_residual，val_score=0.900351，val_RMSE=0.833601，test_RMSE=0.555940，test_tail=0.657612，test_under=0.108696
- strict_main_pool: `v222a_bounded_residual_global_blend_a10p0_b0p2`，variant=bounded_residual，val_score=0.935135，val_RMSE=0.862189，test_RMSE=0.571966，test_tail=0.681413，test_under=0.137931

## 固定 baseline 对照

- loose_main_pool: best fixed baseline on test = `avg_joint_focus`，RMSE=0.544884，tail=0.629752，under=0.163043
- strict_main_pool: best fixed baseline on test = `peak_floor_090`，RMSE=0.571770，tail=0.658306，under=0.137931

## 选择纪律

- validation 参与排序的输出数：108
- 模型 manifest 行数：92
- selection score = RMSE + 0.05 * tail_RMSE + 0.1 * strong_under_rate
- ZIP：`v222a_light_fusion_residual_pack.zip`
