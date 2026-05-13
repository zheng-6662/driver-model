# episode-first v0.6 vehicle-only baseline v0.1

## Inputs

- v0.6 episode table: `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\episode_first_event_v0_6\tables\episode_candidates_v0_6.csv`
- sample manifest: `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\vehicle_instability_response_task_decision_v0_1\tables\sample_response_task_manifest.csv`
- split: `session_level_split`

## Tracks

```text
                    track_id              window_config_id  zero_lateral_offset_feature   n  train_n  val_n  test_n  strict_clean_n  coordinate_flagged_n                                                   module_counts_json                               description_cn
  EP2_expanded_no_lateral_2s          pre2_label2_old_main                         True 265      183     37      45              19                   246 {"curve1": 97, "fix_road": 69, "differentmu_road": 56, "curve2": 43}         episode-first 正样本扩展集，2秒标签，不使用横向偏移特征。
  EP3_expanded_no_lateral_3s pre3_label3_response_coverage                         True 265      183     37      45              19                   246 {"curve1": 97, "fix_road": 69, "differentmu_road": 56, "curve2": 43}         episode-first 正样本扩展集，3秒标签，不使用横向偏移特征。
EP3_expanded_with_lateral_3s pre3_label3_response_coverage                        False 265      183     37      45              19                   246 {"curve1": 97, "fix_road": 69, "differentmu_road": 56, "curve2": 43} episode-first 正样本扩展集，3秒标签，保留横向偏移特征；仅作坐标风险诊断。
```

## Val-selected test results

- EP2_expanded_no_lateral_2s：val 选择 `ridge_rich_context_no_subject`；test RMSE=0.603605，错侧率=0.355556，大幅响应召回=0.000000，严重幅值不足率=0.400000。
- EP3_expanded_no_lateral_3s：val 选择 `formal_ridge_vehicle_context_no_subject`；test RMSE=0.679927，错侧率=0.266667，大幅响应召回=0.250000，严重幅值不足率=0.355556。
- EP3_expanded_with_lateral_3s：val 选择 `formal_ridge_vehicle_context_no_subject`；test RMSE=0.680265，错侧率=0.288889，大幅响应召回=0.250000，严重幅值不足率=0.355556。

## Figures

{
  "metric_summary": "F:\\data_set_process\\data_process\\05_rebuild_from_raw_20260511\\03_baselines\\stage03_episode_first_vehicle_baselines_v0_1\\figures\\episode_first_vehicle_metric_summary_test.png"
}
