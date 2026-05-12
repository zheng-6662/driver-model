# 阶段 3 v0.3 诊断：纯车辆无被试 ID 基线、坏样本和小样本过拟合

更新时间：2026-05-12

## 为什么补这一版

阶段 3 v0.2 的 `ridge_vehicle_summary` 特征中包含 `subject` one-hot。被试 ID 不是车辆历史或道路事件信息，因此它不能作为最终“纯车辆基线”，只能作为驾驶员 ID 控制/上限参考。v0.3 重新生成去掉 `subject` 的车辆基线，并补充固定图/坏样本的可解释表和小样本过拟合测试。

## 本次新增模型

- `ridge_vehicle_no_subject`：线性 ridge，去掉 subject one-hot。
- `knn_vehicle_no_subject`：基于车辆历史统计和道路/事件特征的 kNN 平均轨迹。
- `rbf_krr_vehicle_no_subject`：RBF kernel ridge，多输出轨迹回归，alpha/gamma 只用 train/val 选择。

这些模型仍然只用阶段 2 生成的低泄漏道路曲率车辆窗口，不使用生理、脑电、连续风格，也不使用 old v400 或 raw dynamic 作为主锚点。

## pre2 + session-level test 对照

                model_name  n_samples  rmse_steer  peak_direction_accuracy  wrong_side_rate  large_response_recall  peak_amp_ratio_pred_over_gt_mean  severe_amp_under_rate  peak_time_mae_s  tail_abs_error_mean  reversal_count_exact_match_rate  difficult_top20_rmse                                                                    feature_protocol_note
rbf_krr_vehicle_no_subject         67    0.382337                 0.820896         0.179104               0.545455                          1.083850               0.283582         0.543657             0.379007                         0.029851              0.642092                        v0.3 no subject one-hot; pure vehicle/history/event-road features
    knn_vehicle_no_subject         67    0.388795                 0.746269         0.253731               0.227273                          0.774465               0.477612         0.502090             0.320586                         0.044776              0.624537                        v0.3 no subject one-hot; pure vehicle/history/event-road features
  ridge_vehicle_no_subject         67    0.418778                 0.761194         0.238806               0.136364                          0.704305               0.552239         0.576119             0.383000                         0.089552              0.711623                        v0.3 no subject one-hot; pure vehicle/history/event-road features
     ridge_vehicle_summary         67    0.422204                 0.686567         0.313433               0.181818                          0.731304               0.522388         0.568134             0.406448                         0.000000              0.710431 v0.2 includes subject one-hot; use as driver-id control, not final pure-vehicle baseline
  train_mean_by_event_type         67    0.471718                 0.671642         0.328358               0.000000                          0.612794               0.611940         0.602313             0.430601                         0.194030              0.757330                                       no learned vehicle model or event average baseline
             zero_response         67    0.538630                 0.537313         0.462687               0.000000                          0.000000               1.000000         1.369478             0.435517                         0.104478              0.929939                                       no learned vehicle model or event average baseline

## 当前最好的无被试 ID 纯车辆行

    window_config_id      split_strategy split                 model_name  n_samples  rmse_steer  peak_direction_accuracy  wrong_side_rate  large_response_recall  peak_amp_ratio_pred_over_gt_mean  severe_amp_under_rate  peak_time_mae_s  onset_delay_mae_s  tail_abs_error_mean  reversal_count_exact_match_rate  difficult_top20_rmse                                             feature_protocol_note
pre2_label2_old_main session_level_split  test rbf_krr_vehicle_no_subject         67    0.382337                 0.820896         0.179104               0.545455                           1.08385               0.283582         0.543657           0.306045             0.379007                         0.029851              0.642092 v0.3 no subject one-hot; pure vehicle/history/event-road features

## 小样本过拟合测试

    window_config_id      split_strategy                 model_name  subset_size  subset_train_rmse  full_train_rmse  test_rmse    gamma    alpha              subset_selection
pre2_label2_old_main session_level_split rbf_krr_overfit_no_subject            8           0.000002         0.497726   0.511289 0.034547 0.000001 largest_gt_peak_train_samples
pre2_label2_old_main session_level_split rbf_krr_overfit_no_subject           16           0.000002         0.423151   0.493289 0.035770 0.000001 largest_gt_peak_train_samples
pre2_label2_old_main session_level_split rbf_krr_overfit_no_subject           32           0.000002         0.376132   0.488251 0.034455 0.000001 largest_gt_peak_train_samples
pre2_label2_old_main session_level_split rbf_krr_overfit_no_subject           64           0.000002         0.298024   0.466181 0.032284 0.000001 largest_gt_peak_train_samples
pre2_label2_old_main session_level_split rbf_krr_overfit_no_subject          128           0.000002         0.139305   0.447567 0.041412 0.000001 largest_gt_peak_train_samples

解释：过拟合测试只在 `pre2_label2_old_main + session_level_split` 上运行，用训练集中峰值最大的若干样本拟合 RBF KRR。若子集训练 RMSE 接近 0，但全训练/测试误差仍高，说明当前模型容量和优化能记住小样本，主要问题更可能是泛化、输入信息不足、事件锚点覆盖或响应多模态，而不是评估脚本完全失效。

## 错误桶

gt_peak_abs_bin  n_samples  rmse_mean  wrong_side_rate  severe_under_rate  peak_amp_ratio_mean
         0-0.25         18   0.162434         0.555556           0.166667             1.368392
       0.25-0.5         17   0.190710         0.470588           0.411765             0.655443
        0.5-1.0         17   0.312381         0.176471           0.647059             0.491388
        1.0-2.0         13   0.635590         0.000000           0.923077             0.347544
          >=2.0          2   1.275552         0.000000           1.000000             0.176046

## 当前判断

v0.3 修正后，阶段 3 仍不能进入风格/生理有效性结论。下一步应先确认无被试 ID 纯车辆基线是否足够强，并结合坏样本表检查错误是否集中在大幅响应、错侧、严重幅值不足或多段修正样本。只有强车辆基线稳定后，连续风格和生理增量验证才有公平参照。
