# v248 best-anchor 后残余轨迹形状误差审查

## 结论摘要

- v248 不训练新模型，也不继续调 selector；它只读取 v247 fine-grid + locked v241 预测，量化 best anchor 后剩余错误类型。
- test/all：0ms 平均 RMSE `0.475`，best-anchor 后 `0.253`，锚点平均解释 `0.222`。
- test/bad_top10：0ms 平均 RMSE `1.198`，best-anchor 后 `0.616`，仍高于 `0.65` 的比例 `0.474`。
- test/very_bad_top5：0ms 平均 RMSE `1.382`，best-anchor 后 `0.642`。
- best-anchor 后仍然很差的 test 样本：n=`10`，平均 best RMSE `0.828`。

## 残余错误类型

- `mostly_fixed_low_residual`: n=144, mean best RMSE=0.170, range_ratio=1.290, slope_ratio=0.950
- `amplitude_underestimation_smoothing`: n=14, mean best RMSE=0.526, range_ratio=0.353, slope_ratio=0.371
- `direction_or_reversal_error`: n=12, mean best RMSE=0.586, range_ratio=0.998, slope_ratio=0.608
- `amplitude_underestimation`: n=6, mean best RMSE=0.612, range_ratio=0.650, slope_ratio=0.587
- `calibration_amplitude_bias`: n=3, mean best RMSE=0.485, range_ratio=29.325, slope_ratio=13.716

## 方法解释

v247 证明换锚点有上限收益，但图上已经能看到，橙色 best-anchor 预测仍然经常比真实轨迹更平滑、幅值更小，或者错过快速回正/转折。v248 把这种视觉判断量化成幅值比例、斜率比例、转折次数差、线性校准收益和时间平移收益。

如果 `linear_gain_frac` 高，说明主要是幅值/偏置可校准；如果 `time_shift_gain_frac` 高，说明主要是相位错；如果二者都不高但 RMSE 仍大，通常就是轨迹形状本身没有建好。

## 关键产物

- `tables/v248_best_anchor_residual_decomposition.csv`：每个事件 current 0ms 与 best-anchor 后的形状误差分解。
- `tables/v248_peak_underestimation_table.csv`：峰值/幅值低估和斜率低估排序表。
- `tables/v248_shape_error_categories.csv`：残余错误类别汇总。
- `tables/v248_anchor_vs_shape_summary.csv`：按 split/group 的锚点收益与残余形状指标。
- `figures/v248_best_anchor_still_bad_casebook.png`：best anchor 后仍然最差的样本。
- `figures/v248_improved_but_still_wrong_casebook.png`：换锚点改善明显但形状仍不对的样本。
- `figures/v248_peak_underestimation_casebook.png`：峰值/幅值低估最明显样本。

## 下一步判断

如果 v248 显示主要残余是 amplitude/shape smoothing，而不是 phase/anchor 错位，那么下一步应从 sequential selector 转向 trajectory shape modeling，例如完整曲线 decoder + peak/slope loss，或基于 v241 的 shape residual corrector。