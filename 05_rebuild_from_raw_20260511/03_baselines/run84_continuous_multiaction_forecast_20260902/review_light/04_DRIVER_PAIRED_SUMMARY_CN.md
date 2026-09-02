# 逐驾驶员配对汇总

每名驾驶员先在各自全部 OOF 窗口求误差，再跨三个种子平均；获益表示同一驾驶员的误差低于全局选定的较强简单基线。

| model                                  | 模型                       | reference   |   drivers |   benefited_drivers |   benefit_fraction |   median_improvement_pct |
|:---------------------------------------|:---------------------------|:------------|----------:|--------------------:|-------------------:|-------------------------:|
| extra_trees                            | ExtraTrees                 | hold        |        38 |                  31 |             0.8158 |                   6.0612 |
| transformer                            | 小型Transformer            | hold        |        38 |                  31 |             0.8158 |                   7.406  |
| et_transformer_residual                | ExtraTrees+Transformer残差 | hold        |        38 |                  38 |             1      |                   9.8515 |
| et_transformer_residual_vs_best_single | 残差融合相对较强单模型     | transformer |        38 |                  36 |             0.9474 |                   2.9322 |

逐驾驶员明细见 `results/driver_paired_results.csv`，块级配对明细见 `results/block_metrics.csv`。
