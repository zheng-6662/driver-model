# 阶段 3 用户查看版：更强车辆-only 时序/结构化基线 v0.1

生成时间：2026-05-12

## 为什么做

上一轮车辆-only ridge 基线已经说明：只靠简单统计特征，模型经常出现幅值不足、启动延迟、尾段漂移、反向修正和多段修正错误。所以这一步继续强化“纯车辆基线”，先把车辆本身能做到什么程度压实，再谈风格和生理。

## 这次检查了什么

这次仍然只用车辆历史和事件/道路上下文，不用生理、脑电、连续风格，也不用驾驶员 ID。模型包括更丰富的 ridge、RBF kernel ridge 非线性模型、KNN 模板检索、方向门控模板、峰值缩放模板。

## 目前发现

模型选择只看验证集。验证集选出的模型是：`rbf_kernel_ridge_context_no_subject`。

session-level test 的主要结果如下：

                             model_name  n_samples  rmse_steer  peak_direction_accuracy  wrong_side_rate  large_response_recall  peak_amp_mae  peak_amp_ratio_pred_over_gt_mean  severe_amp_under_rate  peak_time_mae_s  onset_delay_mae_s  tail_abs_error_mean  tail_drift_risk_rate  reversal_count_exact_match_rate  difficult_top20_rmse
        knn_template_context_no_subject        139    0.516941                 0.827338         0.172662                   0.52      0.420714                          1.280109               0.244604         0.432374           0.333741             0.501961              0.446043                         0.007194              1.040524
    rbf_kernel_ridge_context_no_subject        139    0.540287                 0.784173         0.215827                   0.60      0.422609                          0.949628               0.251799         0.455432           0.458957             0.530500              0.532374                         0.043165              1.015198
peak_scaled_template_context_no_subject        139    0.555055                 0.791367         0.208633                   0.60      0.395637                          1.798675               0.079137         0.386403           0.306403             0.552014              0.460432                         0.093525              1.066179
direction_gated_knn_template_no_subject        139    0.579581                 0.791367         0.208633                   0.64      0.396333                          2.024186               0.071942         0.397374           0.338345             0.574765              0.482014                         0.129496              1.085050
formal_ridge_vehicle_context_no_subject        139    0.649341                 0.769784         0.230216                   0.08      0.654372                          1.401294               0.582734         0.449029           0.716942             0.663662              0.625899                         0.093525              1.265239
          ridge_rich_context_no_subject        139    0.652941                 0.820144         0.179856                   0.24      0.552534                          2.096012               0.330935         0.432518           0.327734             0.651683              0.553957                         0.035971              1.254761
          ridge_rich_history_no_subject        139    0.680683                 0.755396         0.244604                   0.16      0.598466                          2.021668               0.374101         0.482302           0.373885             0.696528              0.589928                         0.057554              1.307168

旧 `vehicle_direct` clean 只作为历史参照，不能当新流程真相。

## 哪些结果可信

- 使用同一批正式高置信失稳样本。
- 使用同一个 session-level split。
- 不含生理、脑电、连续风格或驾驶员 ID。
- 标准化和距离特征筛选都只在 train split 拟合。
- 超参数和候选模型选择只用 val，test 只用于最终评估。

## 哪些还不能下结论

这一步仍然不能说明生理、脑电或连续风格是否有效。即使某个纯车辆模型 RMSE 更低，也还要看方向、幅值、尾段、反向修正、多段修正和坏样本图是否真的变好。

## 推荐优先查看

1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_strong_vehicle_baselines_v0_1_cn.md`
2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_strong_vehicle_baselines_v0_1/figures/strong_vehicle_fixed_predictions_test.png`
3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_strong_vehicle_baselines_v0_1/figures/strong_vehicle_bad_samples_test.png`
4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_strong_vehicle_baselines_v0_1/figures/strong_vehicle_model_metric_bars_test.png`
5. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_strong_vehicle_baselines_v0_1/tables/strong_vehicle_baseline_metrics.csv`
