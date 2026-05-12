# 阶段 3：干净响应任务车辆-only 基线 v0.1

## 输入

- 任务 manifest：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_response_task_decision_v0_1/tables/sample_response_task_manifest.csv`
- 轨道 A：`instant2s_core_candidate`
- 轨道 B：`response3s_strict_core_candidate`
- split：`session_level_split`

## 轨道样本量

```text
                track_id              window_config_id                 task_sample_role  n_samples  train_n  val_n  test_n  subject_n                    description_cn
        A_instant2s_core          pre2_label2_old_main         instant2s_core_candidate         84       62     10      12         16 2秒即时响应核心候选：2秒标签稳定，可先验证事件后即时方向盘响应。
B_response3s_strict_core pre3_label3_response_coverage response3s_strict_core_candidate        270      188     42      40         18       3秒响应覆盖严格核心候选：2秒不足但3秒标签相对稳定。
```

## val 选择与 test 结果

- A_instant2s_core: val 选择 `knn_template_context_no_subject`；test RMSE=0.428130，错侧率=0.333333，大幅响应召回=0.600000。
- B_response3s_strict_core: val 选择 `rbf_kernel_ridge_context_no_subject`；test RMSE=0.533667，错侧率=0.225000，大幅响应召回=0.750000。

## test 事后最小 RMSE 诊断

- A_instant2s_core: 按 test RMSE 事后最小为 `zero_response_hold_current`，RMSE=0.336123，只用于诊断，不能替代 val 选择。
- B_response3s_strict_core: 按 test RMSE 事后最小为 `rbf_kernel_ridge_context_no_subject`，RMSE=0.533667，只用于诊断，不能替代 val 选择。

## 输出

- 指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_baselines_v0_1/tables/clean_task_vehicle_metrics.csv`
- 逐样本指标：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_baselines_v0_1/tables/clean_task_vehicle_per_sample_metrics.csv`
- 模型信息：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_baselines_v0_1/tables/clean_task_vehicle_model_info.csv`
- 轨道汇总：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_baselines_v0_1/tables/clean_task_track_summary.csv`
- 指标图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_baselines_v0_1/figures/clean_task_vehicle_metric_summary_test.png`

## 解释边界

本轮只用车辆历史和事件/道路上下文。没有使用生理、脑电、连续风格、驾驶员 ID 或服务器。由于 A 轨道 test 只有 12 个事件，且 KNN 类模型存在 train RMSE 接近 0 的模板记忆风险，结论必须保守；B 轨道样本量更适合后续作为 3 秒响应覆盖的强车辆候选。
