# 阶段 3 用户查看版：旧车辆代码测试全原始失稳样本 v0.1

生成时间：2026-05-12

## 为什么做

你指出之前 404 个主要是弯道样本，不是你真正要的车辆失稳样本。因此这次先不用旧弯道样本，而是把全原始车辆数据重新筛出的高置信失稳事件喂给旧车辆代码，看看这些样本在旧评价体系下是什么难度。

## 检查了什么

- 908 个高置信车辆失稳事件是否能转成旧代码窗口。
- 旧的无学习基线和车辆 ridge 基线在这些样本上的误差。
- 方向、错侧、幅值不足、峰值时间、尾段误差、反向/多段修正等物理指标。
- 固定预测图和坏样本图。

## 目前发现

pre2 窗口、session-level test 的结果如下：

              model_name  n_samples  rmse_steer  peak_direction_accuracy  wrong_side_rate  large_response_recall  peak_amp_mae  peak_amp_ratio_pred_over_gt_mean  severe_amp_under_rate  peak_time_mae_s  tail_abs_error_mean  reversal_count_exact_match_rate  difficult_top20_rmse
   ridge_vehicle_summary        139    0.675055                 0.726619         0.273381                   0.08      0.682363                          1.155460               0.654676         0.446978             0.709058                         0.086331              1.298914
ridge_vehicle_no_subject        139    0.675174                 0.719424         0.280576                   0.08      0.686683                          1.105356               0.669065         0.460216             0.707152                         0.244604              1.305240
train_mean_by_event_type        139    0.677212                 0.568345         0.431655                   0.00      0.847065                          0.247128               0.913669         0.616151             0.719166                         0.086331              1.318363
            hold_current        139    0.683514                 0.517986         0.482014                   0.00      0.936887                          0.000000               1.000000         1.515432             0.725312                         0.064748              1.322311
           zero_response        139    0.683514                 0.517986         0.482014                   0.00      0.936887                          0.000000               1.000000         1.515432             0.725312                         0.064748              1.322311
          train_mean_all        139    0.685789                 0.482014         0.517986                   0.00      0.861593                          0.221422               0.942446         0.971978             0.724539                         0.258993              1.326782
     history_trend_250ms        139    1.455130                 0.244604         0.755396                   0.84      0.829240                          2.297632               0.115108         0.450216             2.045349                         0.064748              2.057304

## 哪些结果可信

- 结果使用的是重新筛选的车辆失稳事件，不是那 404 个弯道候选。
- 输入只来自原始车辆 CSV 派生窗口，原始文件未被修改。
- 训练/验证/测试 split 已分开，ridge 标准化和 alpha 选择只在 train/val 内完成。

## 哪些还不能下结论

- 这还不是正式深度模型训练结果。
- `ridge_vehicle_summary` 是旧代码原样逻辑，含被试 one-hot，只能作为旧代码诊断；更公平时应优先看 `ridge_vehicle_no_subject` 或后续 subject-level split。
- 不能由此判断生理、脑电或连续风格有效。

## 推荐优先查看

- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/oldcode_vehicle_baseline_on_instability_v0_1_cn.md`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/oldcode_vehicle_baselines_on_instability_v0_1/tables/oldcode_instability_baseline_metrics.csv`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/oldcode_vehicle_baselines_on_instability_v0_1/tables/oldcode_instability_best_test_by_window_split.csv`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/oldcode_vehicle_baselines_on_instability_v0_1/figures/oldcode_fixed_predictions_pre2_session_test.png`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/oldcode_vehicle_baselines_on_instability_v0_1/figures/oldcode_bad_samples_pre2_session_test_ridge.png`

## 旧深度模型 smoke

另外补做了一个旧 `vehicle_direct` 深度模型 smoke run。它只用 96/32/32 的 train/val/test 子集，在 CPU 上跑 2 个 epoch，结果是旧代码可以顺利读取新 manifest 并完成训练、验证和测试，丢弃样本为 0。run 目录是：

`F:/data_set_process/data_process/tmp/event_conditioned_runs/OLD_SMOKE_INSTABILITY_HIGHCONF_V0_1_20260512_165950`

这个 smoke 不能当正式性能，只说明旧深度模型入口已经接通。
