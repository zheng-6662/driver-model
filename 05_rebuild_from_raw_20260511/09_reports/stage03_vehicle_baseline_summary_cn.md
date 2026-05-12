# 阶段 3 基线总结：低泄漏道路曲率车辆窗口 v0.2

更新时间：2026-05-12

## 范围

本阶段只使用阶段 2 生成的低泄漏 `raw_road_curvature_onset` 车辆窗口，不使用生理、脑电、连续风格，也不使用 old v400 或 raw dynamic 锚点作为主线。

## 已完成

1. 无学习基线：`zero_response`、`hold_current`、`history_trend_250ms`、`train_mean_all`、`train_mean_by_event_type`。
2. 纯车辆强基线：`ridge_vehicle_summary`，只用车辆历史统计特征和事件元信息，标准化和 alpha 选择均只在 train/val 内完成。
3. 三个窗口、三种 split 均已评估：random event、session-level、subject-level。
4. 指标覆盖整体 RMSE、方向、错侧、大幅响应召回、峰值幅值、峰值时间、尾段、零线穿越、反向修正、多段修正和困难样本。
5. 固定预测图和坏样本图已经生成。

## pre2 + session-level test 关键表

              model_name  n_samples  rmse_steer  peak_direction_accuracy  wrong_side_rate  large_response_recall  peak_amp_mae  peak_amp_ratio_pred_over_gt_mean  severe_amp_under_rate  peak_time_mae_s  tail_abs_error_mean  reversal_count_exact_match_rate  difficult_top20_rmse
   ridge_vehicle_summary         67    0.422204                 0.686567         0.313433               0.181818      0.385982                          0.731304               0.522388         0.568134             0.406448                         0.000000              0.710431
train_mean_by_event_type         67    0.471718                 0.671642         0.328358               0.000000      0.453208                          0.612794               0.611940         0.602313             0.430601                         0.194030              0.757330
          train_mean_all         67    0.530294                 0.462687         0.537313               0.000000      0.568323                          0.246074               0.925373         0.501716             0.438699                         0.104478              0.904616
            hold_current         67    0.538630                 0.537313         0.462687               0.000000      0.646709                          0.000000               1.000000         1.369478             0.435517                         0.104478              0.929939
           zero_response         67    0.538630                 0.537313         0.462687               0.000000      0.646709                          0.000000               1.000000         1.369478             0.435517                         0.104478              0.929939
     history_trend_250ms         67    0.757656                 0.552239         0.447761               0.681818      0.556643                          2.024301               0.358209         0.617015             0.862116                         0.104478              0.969384

## 各窗口/切分测试集最优行

             window_config_id      split_strategy               model_name  rmse_steer  peak_direction_accuracy  wrong_side_rate  severe_amp_under_rate  difficult_top20_rmse
    pre1_label2_event_trigger  random_event_split    ridge_vehicle_summary    0.464197                 0.901961         0.098039               0.215686              0.603640
    pre1_label2_event_trigger session_level_split    ridge_vehicle_summary    0.423973                 0.761194         0.238806               0.507463              0.718079
    pre1_label2_event_trigger subject_level_split    ridge_vehicle_summary    0.564079                 0.771429         0.228571               0.300000              0.923512
         pre2_label2_old_main  random_event_split    ridge_vehicle_summary    0.417086                 0.862745         0.137255               0.294118              0.613687
         pre2_label2_old_main session_level_split    ridge_vehicle_summary    0.422204                 0.686567         0.313433               0.522388              0.710431
         pre2_label2_old_main subject_level_split    ridge_vehicle_summary    0.521596                 0.728571         0.271429               0.642857              0.832318
pre3_label3_response_coverage  random_event_split train_mean_by_event_type    0.589057                 0.764706         0.235294               0.647059              1.031355
pre3_label3_response_coverage session_level_split train_mean_by_event_type    0.515795                 0.686567         0.313433               0.671642              0.903076
pre3_label3_response_coverage subject_level_split    ridge_vehicle_summary    0.634950                 0.728571         0.271429               0.500000              1.149304

## Ridge 训练信息

             window_config_id      split_strategy            model_name status  selected_alpha  val_rmse_for_alpha  train_rmse_selected_alpha  feature_count
    pre1_label2_event_trigger  random_event_split ridge_vehicle_summary     ok            10.0            0.292432                   0.358954             78
    pre1_label2_event_trigger session_level_split ridge_vehicle_summary     ok           100.0            0.513531                   0.377403             78
    pre1_label2_event_trigger subject_level_split ridge_vehicle_summary     ok            10.0            0.407723                   0.330486             78
         pre2_label2_old_main  random_event_split ridge_vehicle_summary     ok            10.0            0.346990                   0.367175             78
         pre2_label2_old_main session_level_split ridge_vehicle_summary     ok           100.0            0.472131                   0.382352             78
         pre2_label2_old_main subject_level_split ridge_vehicle_summary     ok           100.0            0.432733                   0.378378             78
pre3_label3_response_coverage  random_event_split ridge_vehicle_summary     ok            10.0            0.406358                   0.458133             78
pre3_label3_response_coverage session_level_split ridge_vehicle_summary     ok          1000.0            0.548813                   0.556892             78
pre3_label3_response_coverage subject_level_split ridge_vehicle_summary     ok           100.0            0.509743                   0.445728             78

## 当前判断

这一步已经建立了阶段 3 的无学习基线和一个纯车辆强基线。由于当前只覆盖道路曲率候选 359 个事件，结论只能说“低泄漏道路曲率子集上的车辆基线表现”，不能外推到全部旧 v400 事件，也不能用于判断连续风格或生理是否有效。下一步应检查固定图和坏样本，确认指标能解释可视化错误后，再决定是否扩展道路锚点或进入更强车辆模型。
