# 阶段 3 用户查看版总结：无学习基线与纯车辆基线

更新时间：2026-05-12

## 这个阶段为什么做

在讨论连续风格、生理和脑电之前，必须先知道只靠车辆历史和道路事件信息能做到什么程度。否则后面即使模型变好，也说不清是生理有效，还是车辆信息本来就够用。

## 这个阶段检查了什么

- 用低泄漏道路曲率候选样本做基线，不使用旧 v400 响应锚点做主结论。
- 做了零响应、保持当前、历史趋势外推、训练集平均轨迹和同类事件平均轨迹。
- 做了一个纯车辆 ridge 基线，只使用车辆历史窗口统计特征。
- 在随机切分、按记录切分、按被试切分上都算了指标。
- 生成了固定预测图和坏样本图，不只看平均 RMSE。

## 目前发现了什么

pre2 窗口、session-level test 的结果如下：

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
- 指标和固定图可以作为阶段 3 继续调车辆模型的起点。

## 哪些结果还不能下结论

- 不能说风格有效或生理有效。
- 不能说全部事件都已经覆盖，因为当前主线只覆盖 359 个道路曲率候选。
- 不能把 old v400 和 raw dynamic 的结果混进无泄漏主结论。
- 还需要检查固定预测图和坏样本图，确认指标是否能解释具体物理错误。

## 下一阶段是否可以继续

可以继续阶段 3，但不是进入风格/生理阶段。下一步应先看纯车辆基线的固定图和坏样本，必要时扩展低泄漏道路锚点或改进车辆基线。

## 推荐优先查看

- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_baseline_summary_cn.md`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_baselines_v0_2/tables/stage03_baseline_metrics.csv`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_baselines_v0_2/tables/stage03_best_test_by_window_split.csv`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_baselines_v0_2/figures/stage03_fixed_predictions_pre2_session_test.png`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_baselines_v0_2/figures/stage03_bad_samples_pre2_session_test_ridge.png`
