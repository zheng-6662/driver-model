# Hard-Late Fine `003953` Summary From Daily Log

说明：当前 workspace 中未保留 `003953` 的独立 summary artifact，因此这里把 `reports/progress/daily/2026-04-16.md` 中已登记的正式结果单独摘出，供 GPT Pro 快速参考。

## 实验身份

- run：`TRAIN_V5_4_STATECOND_REV_20260417_003953`
- 基线对照：`2026-04-16 220918`
- 路线：`coarse-fine + phase-adaptive + hard-late fine`

## 它在问什么

如果只对 `strong/hard late bucket` 取消部分保守约束，并直接给 fine residual 加 hard-late 补偿，能不能在不破坏 `220918` 大盘收益的前提下，把 `strong_pos / reversal` 拉回来？

## 关键结果

相对 `220918`：

- `rmse_steer: 0.5697 -> 0.6273`
- `late_peak_recall: 0.5940 -> 0.4444`
- `first_reversal_time_mae_sec: 0.6206 -> 0.4576`
- `reversal_count_exact_match_rate: 0.3958 -> 0.5530`
- `strong_pos.tail_amp_ratio_pred_over_gt: 0.3732 -> 0.4193`

## 固定结论

- 它证明 `hard late bucket` 的确值得被单独照顾
- 但“直接加 generic hard-late fine residual loss”会明显破坏 `220918` 赖以成立的 overall/tail/late-peak 收益
- 因此它是 `no-go`
- 后续不应继续沿这条 generic hard-late fine 路线加码
