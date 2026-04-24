# bridge Summary

- Manifest: `F:\data_set_process\data_process\04_project_logs\reports\bridge_training_20260421\bridge_manifest.json`
- Table: `F:\data_set_process\data_process\04_project_logs\reports\bridge_training_20260421\bridge_comparison_table.csv`

## Compact Table

```csv
group_name,status,run_dir,selection_source,rmse_steer,tail_rmse_steer,late_peak_recall,first_reversal_time_mae_sec,reversal_count_exact_match_rate,head_amp_ratio_pred_over_gt,strong_pos_tail_amp_ratio_pred_over_gt,strong_pos_tail_flatness_rate
bridge_55_45,completed,F:\data_set_process\data_process\03_results\tmp\bridge_training_20260421\bridge_55_45\TRAIN_V5_4_STATECOND_REV_20260422_002200,best_by_structured,0.6020424365997314,0.7828868673506366,0.48717948717948717,0.5471974522292994,0.5151515151515151,1.3195054267054585,1.090986748238639,0.42105263157894735
bridge_50_50,completed,F:\data_set_process\data_process\03_results\tmp\bridge_training_20260421\bridge_50_50\TRAIN_V5_4_STATECOND_REV_20260422_004147,best_by_structured,0.5385185480117798,0.6845753417133474,0.6196581196581197,0.5642207792207792,0.4659090909090909,1.3624128032782694,0.4987152039892587,0.7368421052631579
bridge_schedule_B_to_A,completed,F:\data_set_process\data_process\03_results\tmp\bridge_training_20260421\bridge_schedule_B_to_A\TRAIN_V5_4_STATECOND_REV_20260422_010228,best_by_structured,0.5819476842880249,0.7749194158734908,0.6581196581196581,0.4922891566265059,0.48484848484848486,1.35184281606082,1.0622650855216762,0.2631578947368421
```

## Commentary
- `bridge_55_45` kept `best_by_structured` with rmse_steer=0.6020424365997314, tail_rmse_steer=0.7828868673506366, late_peak_recall=0.48717948717948717, first_reversal_time_mae_sec=0.5471974522292994, strong_pos_tail_amp_ratio_pred_over_gt=1.090986748238639, strong_pos_tail_flatness_rate=0.42105263157894735.
- `bridge_50_50` kept `best_by_structured` with rmse_steer=0.5385185480117798, tail_rmse_steer=0.6845753417133474, late_peak_recall=0.6196581196581197, first_reversal_time_mae_sec=0.5642207792207792, strong_pos_tail_amp_ratio_pred_over_gt=0.4987152039892587, strong_pos_tail_flatness_rate=0.7368421052631579.
- `bridge_schedule_B_to_A` kept `best_by_structured` with rmse_steer=0.5819476842880249, tail_rmse_steer=0.7749194158734908, late_peak_recall=0.6581196581196581, first_reversal_time_mae_sec=0.4922891566265059, strong_pos_tail_amp_ratio_pred_over_gt=1.0622650855216762, strong_pos_tail_flatness_rate=0.2631578947368421.

## Metrics Used
- `rmse_steer`
- `tail_rmse_steer`
- `late_peak_recall`
- `first_reversal_time_mae_sec`
- `reversal_count_exact_match_rate`
- `head_amp_ratio_pred_over_gt`
- `strong_pos_tail_amp_ratio_pred_over_gt`
- `strong_pos_tail_flatness_rate`
