# 阶段 3：车辆失稳正式样本无学习与车辆基线 v0.1

生成时间：2026-05-12

## 这次做了什么

基于正式样本清单 `vehicle_instability_highconf_v0_1`，在车辆-only 条件下建立新流程阶段 3 初始基线。这里不使用生理、脑电、连续风格、驾驶员 ID 或旧 deep 模型。

## 输入

- 样本清单：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_highconf_v0_1/tables/samples_master.csv`
- 处理后车辆窗口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_processed_datasets/vehicle_instability_allraw_highconf_v0_1`
- 主窗口：`pre2_label2_old_main`
- 默认切分：`session_level_split`

## 模型

1. `zero_response_hold_current`：事件后方向盘增量保持 0。
2. `history_trend_500ms`：用事件前 500ms 方向盘历史斜率外推。
3. `train_mean_all`：训练集平均响应轨迹。
4. `train_mean_by_event_type`：按训练集事件类型/等级平均响应，样本不足时回退到全局均值。
5. `ridge_vehicle_history_no_subject`：只用事件前车辆历史统计特征，不含驾驶员 ID。
6. `ridge_vehicle_context_no_subject`：车辆历史 + 事件/道路上下文字段，不含驾驶员 ID。

所有 ridge 标准化只在 train split 拟合，alpha 只用 val split 选择。

## 主窗口 session-level test 指标

                      model_name  n_samples  rmse_steer  peak_direction_accuracy  wrong_side_rate  large_response_recall  peak_amp_mae  peak_amp_ratio_pred_over_gt_mean  severe_amp_under_rate  peak_time_mae_s  tail_abs_error_mean  reversal_count_exact_match_rate  difficult_top20_rmse
ridge_vehicle_context_no_subject        139    0.649341                 0.769784         0.230216                   0.08      0.654372                          1.401294               0.582734         0.449029             0.663662                         0.093525              1.265239
        train_mean_by_event_type        139    0.677212                 0.568345         0.431655                   0.00      0.847065                          0.247128               0.913669         0.616151             0.719166                         0.086331              1.318363
      zero_response_hold_current        139    0.683514                 0.517986         0.482014                   0.00      0.936887                          0.000000               1.000000         1.515432             0.725312                         0.064748              1.322311
                  train_mean_all        139    0.685789                 0.482014         0.517986                   0.00      0.861593                          0.221422               0.942446         0.971978             0.724539                         0.258993              1.326782
ridge_vehicle_history_no_subject        139    0.707027                 0.712230         0.287770                   0.12      0.621902                          1.750587               0.510791         0.495647             0.696813                         0.079137              1.344790
             history_trend_500ms        139    1.073892                 0.237410         0.762590                   0.64      0.470965                          1.559471               0.129496         0.479712             1.486614                         0.064748              1.675081

## 固定图和坏样本图

- 固定预测图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_formal_baselines_v0_1/figures/formal_baseline_fixed_predictions_test.png`
- 坏样本图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_formal_baselines_v0_1/figures/formal_baseline_bad_samples_test.png`

## 当前判断

当前最优整体 RMSE 行：`ridge_vehicle_context_no_subject`，RMSE=0.649341。它略差于旧 `vehicle_direct` clean 对照的 RMSE=0.637366，但本轮是新流程正式样本上的浅层车辆基线，不使用旧 deep 结构、不使用驾驶员 ID，标准化和 alpha 选择边界更清楚。

固定图和坏样本图显示，车辆-only 浅层基线仍然明显存在大幅响应召回低、幅值不足、错侧和多段修正失败问题。这只是车辆-only 初始基线，不支持任何连续风格、生理或 EEG 有效性结论。
