# v0.5 新样本集连续风格与生理机制验证

生成时间：2026-05-20 15:14:24

## 本轮要回答什么

本轮不是再跑一个零散对照，而是固定 v0.5 新样本、固定被试划分、固定旧流程粗细双头结构，系统判断：

1. 连续驾驶风格是否仍然有效；
2. 心率、皮电、肌电、脑电单独输入是否有增量；
3. 非脑电生理组合、全生理组合是否优于单信号；
4. 生理信号更适合直接输入、响应类型辅助，还是作为训练期教师；
5. 改善是否体现在整体误差、方向、幅值、尾段、困难样本、分被试或分场景上。

## 固定数据和训练条件

- manifest：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_processed_datasets\stage03_v05_server_aligned_subject_oldflow_fair09\tables\oldflow_fair09_vehicle_only_server_aligned_v05_subject_split_manifest.csv`
- 旧 loader 检查：

```json
{
  "status": "ok",
  "manifest_rows": 1388,
  "old_loader_kept_rows": 1376,
  "old_loader_dropped_rows": 12,
  "split_counts_after_old_loader": {
    "train": 953,
    "val": 260,
    "test": 163
  },
  "subject_counts_after_old_loader": {
    "zx": 178,
    "hzh": 120,
    "byx": 107,
    "txj": 99,
    "zdq": 99,
    "yzy": 93,
    "zxy": 91,
    "rjy": 77,
    "yyl": 77,
    "gzj": 76,
    "gf": 76,
    "jy": 58,
    "cwh": 58,
    "lxy": 48,
    "lx": 38,
    "xst": 30,
    "tyy": 29,
    "zt": 22
  },
  "source_group_counts_after_old_loader": {
    "primary": 1158,
    "manual_review": 148,
    "secondary": 70
  }
}
```

- test 被试：cwh / gf / tyy
- val 被试：byx / gzj / yyl
- train 被试：其余被试
- seed：2026
- epochs：40
- batch：64
- lr：0.001
- device：cuda

## 生理数据可用性检查

        signal              mode                   status  kept_samples  dropped_by_old_loader  train_samples  val_samples  test_samples  context_dim_after_append                                                                                                                                                                                                                                                                              component_names                                                                                                        all_missing_names  min_component_valid_ratio_train                       future_window_risk_note
            HR       raw_hr_only                       ok          1376                     12            953          260           163                         6                                                                                                                                                                                                                                                                                physio_raw_hr                                                                                                                      NaN                         0.918153 使用旧流程按锚点前窗口构造的生理上下文；本轮作为输入窗口生理特征验证，不使用标签窗口统计。
           EDA      raw_eda_only                       ok          1376                     12            953          260           163                         7                                                                                                                                                                                                                                                   physio_raw_eda_tonic|physio_raw_eda_phasic                                                                                                                      NaN                         0.867786 使用旧流程按锚点前窗口构造的生理上下文；本轮作为输入窗口生理特征验证，不使用标签窗口统计。
           EMG      raw_emg_only                       ok          1376                     12            953          260           163                         6                                                                                                                                                                                                                                                                           physio_raw_emg_rms                                                                                                                      NaN                         0.918153 使用旧流程按锚点前窗口构造的生理上下文；本轮作为输入窗口生理特征验证，不使用标签窗口统计。
           EEG      raw_eeg_only has_relevant_all_missing          1376                     12            953          260           163                        13                                                                                                     eeg_raw_alpha_asym|eeg_raw_occ_ta_beta|eeg_raw_frontal_ta_beta|eeg_raw_temporal_ta_beta|eeg_raw_occ_alpha_abs|eeg_raw_temporal_gamma_rel|eeg_raw_occ_gamma_rel|eeg_raw_frontal_gamma_rel alpha_asym|occ_ta_beta|frontal_ta_beta|temporal_ta_beta|occ_alpha_abs|temporal_gamma_rel|occ_gamma_rel|frontal_gamma_rel                         0.000000 使用旧流程按锚点前窗口构造的生理上下文；本轮作为输入窗口生理特征验证，不使用标签窗口统计。
    HR+EDA+EMG raw_physio_no_eeg                       ok          1376                     12            953          260           163                         9                                                                                                                                                                                                                  physio_raw_hr|physio_raw_eda_tonic|physio_raw_eda_phasic|physio_raw_emg_rms                                                                                                                      NaN                         0.867786 使用旧流程按锚点前窗口构造的生理上下文；本轮作为输入窗口生理特征验证，不使用标签窗口统计。
HR+EDA+EMG+EEG        raw_physio has_relevant_all_missing          1376                     12            953          260           163                        17 physio_raw_hr|physio_raw_eda_tonic|physio_raw_eda_phasic|physio_raw_emg_rms|physio_raw_alpha_asym|physio_raw_occ_ta_beta|physio_raw_frontal_ta_beta|physio_raw_temporal_ta_beta|physio_raw_occ_alpha_abs|physio_raw_temporal_gamma_rel|physio_raw_occ_gamma_rel|physio_raw_frontal_gamma_rel alpha_asym|occ_ta_beta|frontal_ta_beta|temporal_ta_beta|occ_alpha_abs|temporal_gamma_rel|occ_gamma_rel|frontal_gamma_rel                         0.000000 使用旧流程按锚点前窗口构造的生理上下文；本轮作为输入窗口生理特征验证，不使用标签窗口统计。

## 当前完成结果

exp_id                   label_cn  test_steer_rmse  primary_rmse  tail_rmse  selection  large_wrong_side_rate  large_severe_under_rate  large_response_recall
    S1                    车辆 + 心率         0.375885      0.247541   0.327355   0.834710               0.094118                 0.317647               0.682353
    S2                    车辆 + 皮电         0.389752      0.251093   0.333393   0.838734               0.105882                 0.282353               0.717647
    S3                    车辆 + 肌电         0.421814      0.262265   0.360342   0.849215               0.082353                 0.235294               0.764706
   SF1             车辆 + 连续风格 + 心率         0.371878      0.276714   0.315840   0.824366               0.105882                 0.258824               0.741176
   SF2             车辆 + 连续风格 + 皮电         0.332864      0.234360   0.270137   0.783221               0.070588                 0.329412               0.670588
   SF3             车辆 + 连续风格 + 肌电         0.350464      0.254048   0.348584   0.825009               0.105882                 0.388235               0.611765
    A1    车辆 + 连续风格 + 肌电 + 响应类型辅助         0.352993      0.234261   0.278117   0.787202               0.094118                 0.400000               0.600000
    A2 车辆 + 连续风格 + 非脑电生理 + 响应类型辅助         0.368899      0.252720   0.331414   0.802218               0.094118                 0.376471               0.623529
    B0               车辆-only 粗细双头         0.338616      0.218387   0.310550   0.820553               0.082353                 0.270588               0.729412
    B1                  车辆 + 连续风格         0.376946      0.247150   0.291497   0.776446               0.129412                 0.388235               0.611765
    T2     非脑电生理教师 -> 车辆 + 连续风格学生         0.324739      0.216134   0.290687   0.788948               0.070588                 0.329412               0.670588
    C1          车辆 + 心率 + 皮电 + 肌电         0.375960      0.244458   0.341925   0.814801               0.047059                 0.305882               0.694118
    C2   车辆 + 连续风格 + 心率 + 皮电 + 肌电         0.362587      0.243716   0.302276   0.789580               0.141176                 0.282353               0.717647

## 产物位置

- 实验注册表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\tables\v05_physio_experiment_registry.csv`
- 运行状态表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\tables\v05_physio_run_status.csv`
- 生理可用性表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\tables\v05_physio_availability_check.csv`
- 总指标表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\tables\v05_physio_comparison_table.csv`
- 分被试表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\tables\v05_physio_subject_metrics.csv`
- 机制判断表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\tables\v05_physio_mechanism_table.csv`
- 服务器启动命令模板：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\launch_commands_server_no_password.sh`

## 当前解释边界

- B0 是已完成的车辆-only 主基准，不重复训练。
- B1 用来验证连续风格。
- S/SF/C 版本回答直接输入是否有效。
- A 版本回答生理是否更适合帮助判断响应类型。
- T 版本回答生理/脑电是否更适合作为训练期教师。
- 单个 seed 只用于筛选，不能直接形成最终论文强结论；有希望版本还要补 seed2027/2028。
