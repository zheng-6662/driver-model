# input_ablation Summary

- Manifest: `F:\data_set_process\data_process\04_project_logs\reports\input_group_ablation_20260421\input_ablation_manifest.json`
- Table: `F:\data_set_process\data_process\04_project_logs\reports\input_group_ablation_20260421\input_ablation_comparison_table.csv`

## Compact Table

```csv
group_name,status,run_dir,selection_source,rmse_steer,tail_rmse_steer,late_peak_recall,first_reversal_time_mae_sec,reversal_count_exact_match_rate,head_amp_ratio_pred_over_gt,strong_pos_tail_amp_ratio_pred_over_gt,strong_pos_tail_flatness_rate
baseline_fixed_input,completed,F:\data_set_process\data_process\03_results\tmp\input_group_ablation_20260421\baseline_fixed_input\TRAIN_V5_4_STATECOND_REV_20260421_223235,best_by_structured,0.5559393763542175,0.7171384919281619,0.6495726495726496,0.5107232704402515,0.5151515151515151,1.3399970416732658,1.3489819546120279,0.21052631578947367
plus_pedals,completed,F:\data_set_process\data_process\03_results\tmp\input_group_ablation_20260421\plus_pedals\TRAIN_V5_4_STATECOND_REV_20260421_225126,best_by_structured,0.5663110613822937,0.7445443652592684,0.8504273504273504,0.45500000000000007,0.5303030303030303,1.3909124412316665,1.1458483415281322,0.42105263157894735
plus_lat_dyn,completed,F:\data_set_process\data_process\03_results\tmp\input_group_ablation_20260421\plus_lat_dyn\TRAIN_V5_4_STATECOND_REV_20260421_230953,best_by_structured,0.5918551683425903,0.7764903737454972,0.5384615384615384,0.41359195402298854,0.5208333333333334,1.5072440423450577,1.0378759465703213,0.3157894736842105
plus_road_cond,completed,F:\data_set_process\data_process\03_results\tmp\input_group_ablation_20260421\plus_road_cond\TRAIN_V5_4_STATECOND_REV_20260421_232824,best_by_structured,0.6604995727539062,0.8649199837355365,0.6239316239316239,0.4396026490066225,0.5151515151515151,1.4712354002053378,0.9515823978673696,0.47368421052631576
minus_z,completed,F:\data_set_process\data_process\03_results\tmp\input_group_ablation_20260421\minus_z\TRAIN_V5_4_STATECOND_REV_20260421_234650,best_by_structured,0.5759932994842529,0.7385062148102367,0.6324786324786325,0.39509202453987735,0.5037878787878788,1.2346933018678365,0.821413919276791,0.47368421052631576
```

## Commentary
- `plus_pedals` vs baseline: delta_rmse_steer=0.010371685028076172, delta_tail_rmse_steer=0.027405873331106445.
- `plus_lat_dyn` vs baseline: delta_rmse_steer=0.0359157919883728, delta_tail_rmse_steer=0.059351881817335306.
- `plus_road_cond` vs baseline: delta_rmse_steer=0.10456019639968872, delta_tail_rmse_steer=0.14778149180737454.
- `minus_z` vs baseline: delta_rmse_steer=0.0200539231300354, delta_tail_rmse_steer=0.02136772288207478.

## Metrics Used
- `rmse_steer`
- `tail_rmse_steer`
- `late_peak_recall`
- `first_reversal_time_mae_sec`
- `reversal_count_exact_match_rate`
- `head_amp_ratio_pred_over_gt`
- `strong_pos_tail_amp_ratio_pred_over_gt`
- `strong_pos_tail_flatness_rate`
