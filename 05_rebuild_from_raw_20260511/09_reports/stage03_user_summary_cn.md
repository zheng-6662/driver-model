# 阶段 3 用户查看版总结：无学习基线与纯车辆基线

更新时间：2026-05-12

## 这个阶段为什么做

在讨论连续风格、生理和脑电之前，必须先知道只靠车辆历史和道路事件信息能做到什么程度。否则后面即使模型变好，也说不清是生理有效，还是车辆信息本来就够用。

## 这个阶段检查了什么

- 用低泄漏道路曲率候选样本做基线，不使用旧 v400 响应锚点做主结论。
- 做了零响应、保持当前、历史趋势外推、训练集平均轨迹和同类事件平均轨迹。
- 先做了一个 v0.2 ridge 基线；随后发现该基线包含 `subject` one-hot，因此只能作为驾驶员 ID 控制参考，不能作为最终纯车辆基线。
- v0.3 已补充去掉 `subject` 的纯车辆基线：`ridge_vehicle_no_subject`、`knn_vehicle_no_subject`、`rbf_krr_vehicle_no_subject`。
- 在随机切分、按记录切分、按被试切分上都算了指标。
- 生成了固定预测图、坏样本图、坏样本诊断表、错误桶和小样本过拟合测试，不只看平均 RMSE。

## 目前发现了什么

pre2 窗口、session-level test 的 v0.3 关键结果如下：

                model_name  n_samples  rmse_steer  peak_direction_accuracy  wrong_side_rate  large_response_recall  peak_amp_ratio_pred_over_gt_mean  severe_amp_under_rate  difficult_top20_rmse
rbf_krr_vehicle_no_subject         67    0.382337                 0.820896         0.179104               0.545455                          1.083850               0.283582              0.642092
    knn_vehicle_no_subject         67    0.388795                 0.746269         0.253731               0.227273                          0.774465               0.477612              0.624537
  ridge_vehicle_no_subject         67    0.418778                 0.761194         0.238806               0.136364                          0.704305               0.552239              0.711623
     ridge_vehicle_summary         67    0.422204                 0.686567         0.313433               0.181818                          0.731304               0.522388              0.710431

注意：`ridge_vehicle_summary` 是 v0.2 结果，包含被试 ID，因此现在只作为“带驾驶员 ID 的控制参考”，不再作为最终纯车辆主结论。

pre2 窗口、session-level test 的 v0.2 结果如下，保留用于历史对照：

              model_name  n_samples  rmse_steer  peak_direction_accuracy  wrong_side_rate  large_response_recall  peak_amp_mae  peak_amp_ratio_pred_over_gt_mean  severe_amp_under_rate  peak_time_mae_s  tail_abs_error_mean  reversal_count_exact_match_rate  difficult_top20_rmse
   ridge_vehicle_summary         67    0.422204                 0.686567         0.313433               0.181818      0.385982                          0.731304               0.522388         0.568134             0.406448                         0.000000              0.710431
train_mean_by_event_type         67    0.471718                 0.671642         0.328358               0.000000      0.453208                          0.612794               0.611940         0.602313             0.430601                         0.194030              0.757330
          train_mean_all         67    0.530294                 0.462687         0.537313               0.000000      0.568323                          0.246074               0.925373         0.501716             0.438699                         0.104478              0.904616
            hold_current         67    0.538630                 0.537313         0.462687               0.000000      0.646709                          0.000000               1.000000         1.369478             0.435517                         0.104478              0.929939
           zero_response         67    0.538630                 0.537313         0.462687               0.000000      0.646709                          0.000000               1.000000         1.369478             0.435517                         0.104478              0.929939
     history_trend_250ms         67    0.757656                 0.552239         0.447761               0.681818      0.556643                          2.024301               0.358209         0.617015             0.862116                         0.104478              0.969384

## 哪些结果可信

- 这些结果只依赖原始车辆数据派生出的低泄漏道路曲率候选。
- 车辆窗口处理没有改原始 CSV，没有用生理/脑电，没有用测试集统计做标准化。
- v0.3 的 `*_no_subject` 模型没有使用被试 ID，更适合作为纯车辆基线。
- 指标、固定图、坏样本诊断表和小样本过拟合表可以作为阶段 3 继续调车辆模型的起点。

## 哪些结果还不能下结论

- 不能说风格有效或生理有效。
- 不能说全部事件都已经覆盖，因为当前主线只覆盖 359 个道路曲率候选。
- 不能把 old v400 和 raw dynamic 的结果混进无泄漏主结论。
- 不能把含 `subject` 的 v0.2 ridge 当成最终纯车辆基线。
- 虽然 v0.3 RBF KRR 明显改善 RMSE、方向和幅值不足，但它仍只覆盖低泄漏道路曲率子集，不能代表全部事件。

## 下一阶段是否可以继续

可以继续阶段 3，但不是进入风格/生理阶段。下一步应先复核 v0.3 的坏样本诊断和大幅响应错误桶，必要时扩展低泄漏道路锚点或改进车辆时序基线。

## 推荐优先查看

- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_baseline_summary_cn.md`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_diagnostics_v0_3_cn.md`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_diagnostics_v0_3/tables/stage03_vehicle_model_comparison_v0_3.csv`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_diagnostics_v0_3/tables/stage03_bad_sample_diagnostics_v0_3.csv`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_diagnostics_v0_3/figures/stage03_pre2_session_model_rmse_comparison_v0_3.png`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_baselines_v0_2/tables/stage03_baseline_metrics.csv`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_baselines_v0_2/tables/stage03_best_test_by_window_split.csv`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_baselines_v0_2/figures/stage03_fixed_predictions_pre2_session_test.png`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_baselines_v0_2/figures/stage03_bad_samples_pre2_session_test_ridge.png`
