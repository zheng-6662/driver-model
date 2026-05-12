# 阶段 3 用户查看版：车辆失稳正式样本车辆基线 v0.1

生成时间：2026-05-12

## 为什么做

正式样本清单已经完成，所以现在先回答“只靠车辆历史和事件信息能预测到什么程度”。这一步仍然不进入风格、生理或脑电。

## 检查了什么

- 零响应/保持当前值。
- 历史趋势外推。
- 训练集平均响应。
- 不含驾驶员 ID 的 ridge 车辆历史模型。
- 不含驾驶员 ID 的 ridge 车辆历史 + 事件/道路上下文模型。
- 固定预测图和坏样本图。

## 目前发现

主窗口 `pre2_label2_old_main`、session-level test 的结果如下：

                      model_name  n_samples  rmse_steer  peak_direction_accuracy  wrong_side_rate  large_response_recall  peak_amp_mae  peak_amp_ratio_pred_over_gt_mean  severe_amp_under_rate  peak_time_mae_s  tail_abs_error_mean  reversal_count_exact_match_rate  difficult_top20_rmse
ridge_vehicle_context_no_subject        139    0.649341                 0.769784         0.230216                   0.08      0.654372                          1.401294               0.582734         0.449029             0.663662                         0.093525              1.265239
        train_mean_by_event_type        139    0.677212                 0.568345         0.431655                   0.00      0.847065                          0.247128               0.913669         0.616151             0.719166                         0.086331              1.318363
      zero_response_hold_current        139    0.683514                 0.517986         0.482014                   0.00      0.936887                          0.000000               1.000000         1.515432             0.725312                         0.064748              1.322311
                  train_mean_all        139    0.685789                 0.482014         0.517986                   0.00      0.861593                          0.221422               0.942446         0.971978             0.724539                         0.258993              1.326782
ridge_vehicle_history_no_subject        139    0.707027                 0.712230         0.287770                   0.12      0.621902                          1.750587               0.510791         0.495647             0.696813                         0.079137              1.344790
             history_trend_500ms        139    1.073892                 0.237410         0.762590                   0.64      0.470965                          1.559471               0.129496         0.479712             1.486614                         0.064748              1.675081

最优整体 RMSE 是 `ridge_vehicle_context_no_subject` 的 0.649341，略差于旧 `vehicle_direct` clean 对照的 0.637366。这个结果更适合作为新流程浅层车辆基线起点，因为它不使用旧 deep 结构、不使用驾驶员 ID，训练边界更清楚；但固定图和坏样本图仍然显示大幅响应和多段修正预测不足。

## 哪些结果可信

本轮没有使用生理、脑电、连续风格或驾驶员 ID。ridge 的标准化只在训练集拟合，alpha 只用验证集选择，测试集只用于最后评估。

## 哪些结果还不能下结论

这只是车辆-only 基线，不证明风格、生理或 EEG 有效。还需要结合固定图和坏样本图确认物理错误类型，不能只看 RMSE 排名。

## 下一阶段是否可以继续

可以继续细化强车辆基线和固定图协议；只有强车辆基线稳定后，才能进入连续风格和生理增量验证。

## 推荐优先查看

1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_formal_baselines_v0_1_cn.md`
2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_formal_baselines_v0_1/figures/formal_baseline_fixed_predictions_test.png`
3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_formal_baselines_v0_1/figures/formal_baseline_bad_samples_test.png`
4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_formal_baselines_v0_1/tables/formal_baseline_metrics.csv`
