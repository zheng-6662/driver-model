# v293 physiology response visibility / latency audit

## 本轮目的

- v292 已经说明好候选存在，但 observation 前源生理 pairwise matching 不能稳定选对。
- v293 检查生理差异是否可能主要出现在 observation 后短时间窗。
- post 特征只作为 diagnostic / waiting-policy evidence，不作为当前锚点部署输入。

## decision

| target                      | is_core_decision_target   |   pre_test_best_auc |   early_post_test_best_auc |   late_post_test_best_auc |   early_minus_pre | pre_visible   | post_visibility_gain   | pre_weak_subgroup_signal_exists   | pre_route_supported_now   | post_wait_route_supported_diagnostic   |
|:----------------------------|:--------------------------|--------------------:|---------------------------:|--------------------------:|------------------:|:--------------|:-----------------------|:----------------------------------|:--------------------------|:---------------------------------------|
| bad_top10                   | True                      |            0.489633 |                   0.772568 |                  0.498884 |         0.282935  | False         | True                   | True                              | False                     | True                                   |
| bad_top10_vehicle_ambiguous | True                      |            0.601183 |                   0.662722 |                  0.585799 |         0.0615385 | True          | False                  | True                              | False                     | True                                   |
| vehicle_ambiguous           | False                     |            0.540028 |                   0.495442 |                  0.509544 |        -0.0445869 | False         | False                  | True                              | False                     | True                                   |
| candidate_pool_gain_gt_005  | True                      |            0.572206 |                   0.559314 |                  0.58084  |        -0.0128918 | False         | False                  | True                              | False                     | True                                   |
| candidate_pool_gain_gt_02   | False                     |            0.600378 |                   0.615627 |                  0.606679 |         0.0152489 | True          | False                  | True                              | False                     | True                                   |

## classifier top results

| target                      | split   | feature_set             | phase      | model_name      |   n |   positive_rate |      auc |   average_precision |   feature_n |
|:----------------------------|:--------|:------------------------|:-----------|:----------------|----:|----------------:|---------:|--------------------:|------------:|
| bad_top10                   | test    | early_post_top80        | early_post | extra_trees_d5  | 184 |       0.103261  | 0.772568 |           0.255738  |          80 |
| bad_top10                   | test    | all_prepost_top120      | mixed      | extra_trees_d5  | 184 |       0.103261  | 0.734609 |           0.207407  |         120 |
| bad_top10                   | test    | window_post0_3_top48    | early_post | extra_trees_d5  | 184 |       0.103261  | 0.725359 |           0.24746   |          48 |
| bad_top10                   | test    | window_post0_2_top48    | early_post | extra_trees_d5  | 184 |       0.103261  | 0.705263 |           0.191587  |          48 |
| bad_top10                   | test    | window_post0_5_top48    | early_post | extra_trees_d5  | 184 |       0.103261  | 0.693461 |           0.206862  |          48 |
| bad_top10                   | test    | window_post1_3_top48    | early_post | extra_trees_d5  | 184 |       0.103261  | 0.661563 |           0.210048  |          48 |
| bad_top10                   | test    | window_post0_2_top48    | early_post | logreg_balanced | 184 |       0.103261  | 0.65008  |           0.176157  |          48 |
| bad_top10                   | test    | all_prepost_top120      | mixed      | logreg_balanced | 184 |       0.103261  | 0.603828 |           0.176975  |         120 |
| bad_top10                   | val     | window_post0_1_top48    | early_post | logreg_balanced | 309 |       0.100324  | 0.65514  |           0.227749  |          48 |
| bad_top10                   | val     | window_post0_1_top48    | early_post | extra_trees_d5  | 309 |       0.100324  | 0.614296 |           0.183848  |          48 |
| bad_top10                   | val     | early_post_top80        | early_post | extra_trees_d5  | 309 |       0.100324  | 0.598863 |           0.1535    |          80 |
| bad_top10                   | val     | all_prepost_top120      | mixed      | extra_trees_d5  | 309 |       0.100324  | 0.580645 |           0.148484  |         120 |
| bad_top10                   | val     | window_post0_3_top48    | early_post | extra_trees_d5  | 309 |       0.100324  | 0.577396 |           0.161282  |          48 |
| bad_top10                   | val     | window_pre2_0_top48     | pre        | extra_trees_d5  | 309 |       0.100324  | 0.549083 |           0.137895  |          48 |
| bad_top10                   | val     | window_post0_2_top48    | early_post | extra_trees_d5  | 309 |       0.100324  | 0.546763 |           0.165961  |          48 |
| bad_top10                   | val     | early_post_top80        | early_post | logreg_balanced | 309 |       0.100324  | 0.544442 |           0.133735  |          80 |
| bad_top10_vehicle_ambiguous | test    | window_post0_3_top48    | early_post | logreg_balanced | 184 |       0.0815217 | 0.662722 |           0.22851   |          48 |
| bad_top10_vehicle_ambiguous | test    | window_post0_2_top48    | early_post | logreg_balanced | 184 |       0.0815217 | 0.65641  |           0.191257  |          48 |
| bad_top10_vehicle_ambiguous | test    | window_post0_2_top48    | early_post | extra_trees_d5  | 184 |       0.0815217 | 0.654832 |           0.236286  |          48 |
| bad_top10_vehicle_ambiguous | test    | all_prepost_top120      | mixed      | logreg_balanced | 184 |       0.0815217 | 0.643393 |           0.193532  |         120 |
| bad_top10_vehicle_ambiguous | test    | early_post_top80        | early_post | extra_trees_d5  | 184 |       0.0815217 | 0.642998 |           0.213665  |          80 |
| bad_top10_vehicle_ambiguous | test    | window_post0_3_top48    | early_post | extra_trees_d5  | 184 |       0.0815217 | 0.616963 |           0.180435  |          48 |
| bad_top10_vehicle_ambiguous | test    | window_pre5_pre2_top48  | pre        | extra_trees_d5  | 184 |       0.0815217 | 0.601183 |           0.212892  |          48 |
| bad_top10_vehicle_ambiguous | test    | window_post1_3_top48    | early_post | extra_trees_d5  | 184 |       0.0815217 | 0.600789 |           0.202917  |          48 |
| bad_top10_vehicle_ambiguous | val     | window_post0_1_top48    | early_post | logreg_balanced | 309 |       0.0873786 | 0.651826 |           0.190607  |          48 |
| bad_top10_vehicle_ambiguous | val     | window_post0_1_top48    | early_post | extra_trees_d5  | 309 |       0.0873786 | 0.589572 |           0.130179  |          48 |
| bad_top10_vehicle_ambiguous | val     | window_post0_2_top48    | early_post | logreg_balanced | 309 |       0.0873786 | 0.553323 |           0.120288  |          48 |
| bad_top10_vehicle_ambiguous | val     | window_post0_3_top48    | early_post | extra_trees_d5  | 309 |       0.0873786 | 0.513265 |           0.114924  |          48 |
| bad_top10_vehicle_ambiguous | val     | window_post0_2_top48    | early_post | extra_trees_d5  | 309 |       0.0873786 | 0.493696 |           0.120232  |          48 |
| bad_top10_vehicle_ambiguous | val     | window_post1_3_top48    | early_post | extra_trees_d5  | 309 |       0.0873786 | 0.478592 |           0.113821  |          48 |
| bad_top10_vehicle_ambiguous | val     | early_post_top80        | early_post | extra_trees_d5  | 309 |       0.0873786 | 0.477673 |           0.092939  |          80 |
| bad_top10_vehicle_ambiguous | val     | window_pre10_pre5_top48 | pre        | extra_trees_d5  | 309 |       0.0873786 | 0.467691 |           0.0896845 |          48 |
| candidate_pool_gain_gt_005  | test    | late_post_top80         | late_post  | logreg_balanced | 184 |       0.483696  | 0.58084  |           0.550191  |          80 |
| candidate_pool_gain_gt_005  | test    | window_post5_10_top48   | late_post  | logreg_balanced | 184 |       0.483696  | 0.575517 |           0.527758  |          48 |
| candidate_pool_gain_gt_005  | test    | window_pre5_pre2_top48  | pre        | logreg_balanced | 184 |       0.483696  | 0.572206 |           0.57189   |          48 |
| candidate_pool_gain_gt_005  | test    | window_post0_1_top48    | early_post | logreg_balanced | 184 |       0.483696  | 0.559314 |           0.552433  |          48 |
| candidate_pool_gain_gt_005  | test    | all_prepost_top120      | mixed      | logreg_balanced | 184 |       0.483696  | 0.552454 |           0.517509  |         120 |
| candidate_pool_gain_gt_005  | test    | pre_top80               | pre        | logreg_balanced | 184 |       0.483696  | 0.55139  |           0.545692  |          80 |
| candidate_pool_gain_gt_005  | test    | window_post0_3_top48    | early_post | logreg_balanced | 184 |       0.483696  | 0.549143 |           0.571967  |          48 |
| candidate_pool_gain_gt_005  | test    | window_pre10_pre5_top48 | pre        | logreg_balanced | 184 |       0.483696  | 0.547132 |           0.551786  |          48 |
| candidate_pool_gain_gt_005  | val     | late_post_top80         | late_post  | logreg_balanced | 309 |       0.414239  | 0.561939 |           0.473963  |          80 |
| candidate_pool_gain_gt_005  | val     | all_prepost_top120      | mixed      | logreg_balanced | 309 |       0.414239  | 0.557666 |           0.457915  |         120 |
| candidate_pool_gain_gt_005  | val     | window_post0_1_top48    | early_post | logreg_balanced | 309 |       0.414239  | 0.552702 |           0.469952  |          48 |
| candidate_pool_gain_gt_005  | val     | early_post_top80        | early_post | logreg_balanced | 309 |       0.414239  | 0.544069 |           0.456085  |          80 |
| candidate_pool_gain_gt_005  | val     | window_post0_5_top48    | early_post | extra_trees_d5  | 309 |       0.414239  | 0.542041 |           0.458918  |          48 |
| candidate_pool_gain_gt_005  | val     | window_post2_5_top48    | late_post  | logreg_balanced | 309 |       0.414239  | 0.531552 |           0.43917   |          48 |
| candidate_pool_gain_gt_005  | val     | window_post2_5_top48    | late_post  | extra_trees_d5  | 309 |       0.414239  | 0.530516 |           0.432812  |          48 |
| candidate_pool_gain_gt_005  | val     | window_pre10_pre5_top48 | pre        | logreg_balanced | 309 |       0.414239  | 0.524776 |           0.438432  |          48 |
| candidate_pool_gain_gt_02   | test    | window_post0_5_top48    | early_post | logreg_balanced | 184 |       0.625     | 0.615627 |           0.745319  |          48 |
| candidate_pool_gain_gt_02   | test    | late_post_top80         | late_post  | logreg_balanced | 184 |       0.625     | 0.606679 |           0.688398  |          80 |
| candidate_pool_gain_gt_02   | test    | early_post_top80        | early_post | logreg_balanced | 184 |       0.625     | 0.60189  |           0.694836  |          80 |
| candidate_pool_gain_gt_02   | test    | window_pre10_pre5_top48 | pre        | logreg_balanced | 184 |       0.625     | 0.600378 |           0.724186  |          48 |
| candidate_pool_gain_gt_02   | test    | window_post5_10_top48   | late_post  | extra_trees_d5  | 184 |       0.625     | 0.596597 |           0.708945  |          48 |
| candidate_pool_gain_gt_02   | test    | window_post5_10_top48   | late_post  | logreg_balanced | 184 |       0.625     | 0.588028 |           0.6824    |          48 |
| candidate_pool_gain_gt_02   | test    | window_pre2_0_top48     | pre        | logreg_balanced | 184 |       0.625     | 0.57971  |           0.722545  |          48 |
| candidate_pool_gain_gt_02   | test    | all_prepost_top120      | mixed      | logreg_balanced | 184 |       0.625     | 0.577316 |           0.68956   |         120 |
| candidate_pool_gain_gt_02   | val     | window_post0_5_top48    | early_post | logreg_balanced | 309 |       0.582524  | 0.548191 |           0.63601   |          48 |
| candidate_pool_gain_gt_02   | val     | window_post0_2_top48    | early_post | extra_trees_d5  | 309 |       0.582524  | 0.533463 |           0.601782  |          48 |
| candidate_pool_gain_gt_02   | val     | window_post0_3_top48    | early_post | logreg_balanced | 309 |       0.582524  | 0.52528  |           0.608939  |          48 |
| candidate_pool_gain_gt_02   | val     | early_post_top80        | early_post | extra_trees_d5  | 309 |       0.582524  | 0.52261  |           0.605164  |          80 |
| candidate_pool_gain_gt_02   | val     | window_post0_1_top48    | early_post | logreg_balanced | 309 |       0.582524  | 0.519165 |           0.606945  |          48 |
| candidate_pool_gain_gt_02   | val     | window_post2_5_top48    | late_post  | logreg_balanced | 309 |       0.582524  | 0.517399 |           0.601293  |          48 |
| candidate_pool_gain_gt_02   | val     | all_prepost_top120      | mixed      | logreg_balanced | 309 |       0.582524  | 0.512016 |           0.581368  |         120 |
| candidate_pool_gain_gt_02   | val     | window_post2_5_top48    | late_post  | extra_trees_d5  | 309 |       0.582524  | 0.50758  |           0.577778  |          48 |
| vehicle_ambiguous           | test    | window_pre10_pre5_top48 | pre        | logreg_balanced | 184 |       0.706522  | 0.540028 |           0.758144  |          48 |
| vehicle_ambiguous           | test    | window_pre2_0_top48     | pre        | extra_trees_d5  | 184 |       0.706522  | 0.53547  |           0.766857  |          48 |
| vehicle_ambiguous           | test    | window_post2_5_top48    | late_post  | extra_trees_d5  | 184 |       0.706522  | 0.509544 |           0.710633  |          48 |
| vehicle_ambiguous           | test    | window_post0_3_top48    | early_post | logreg_balanced | 184 |       0.706522  | 0.495442 |           0.72471   |          48 |
| vehicle_ambiguous           | test    | window_pre2_0_top48     | pre        | logreg_balanced | 184 |       0.706522  | 0.492593 |           0.750679  |          48 |
| vehicle_ambiguous           | test    | pre_top80               | pre        | logreg_balanced | 184 |       0.706522  | 0.487322 |           0.70807   |          80 |
| vehicle_ambiguous           | test    | window_post1_3_top48    | early_post | extra_trees_d5  | 184 |       0.706522  | 0.482051 |           0.701167  |          48 |
| vehicle_ambiguous           | test    | window_post0_3_top48    | early_post | extra_trees_d5  | 184 |       0.706522  | 0.475783 |           0.728032  |          48 |
| vehicle_ambiguous           | val     | window_post0_3_top48    | early_post | extra_trees_d5  | 309 |       0.708738  | 0.573871 |           0.767338  |          48 |
| vehicle_ambiguous           | val     | early_post_top80        | early_post | logreg_balanced | 309 |       0.708738  | 0.568747 |           0.751811  |          80 |
| vehicle_ambiguous           | val     | window_post0_3_top48    | early_post | logreg_balanced | 309 |       0.708738  | 0.550228 |           0.767494  |          48 |
| vehicle_ambiguous           | val     | window_pre5_pre2_top48  | pre        | extra_trees_d5  | 309 |       0.708738  | 0.546829 |           0.752894  |          48 |
| vehicle_ambiguous           | val     | window_post0_1_top48    | early_post | logreg_balanced | 309 |       0.708738  | 0.539168 |           0.73934   |          48 |
| vehicle_ambiguous           | val     | window_pre5_pre2_top48  | pre        | logreg_balanced | 309 |       0.708738  | 0.537747 |           0.744738  |          48 |
| vehicle_ambiguous           | val     | early_post_top80        | early_post | extra_trees_d5  | 309 |       0.708738  | 0.537697 |           0.739289  |          80 |
| vehicle_ambiguous           | val     | window_post0_5_top48    | early_post | extra_trees_d5  | 309 |       0.708738  | 0.534652 |           0.737044  |          48 |

## window/signal train screen

| phase      | window     | signal   |   feature_n |   max_corr |   mean_top10_corr |
|:-----------|:-----------|:---------|------------:|-----------:|------------------:|
| early_post | post1_3    | emg      |           9 |  0.170411  |         0.133944  |
| early_post | post0_5    | emg      |           9 |  0.167144  |         0.129468  |
| early_post | post0_3    | emg      |           9 |  0.145452  |         0.129826  |
| early_post | post0_1    | emg      |           9 |  0.129108  |         0.0958785 |
| early_post | post0_1    | ecg      |           9 |  0.128963  |         0.0947213 |
| early_post | post0_2    | emg      |           9 |  0.12707   |         0.112793  |
| early_post | post1_3    | hr       |           9 |  0.122178  |         0.091782  |
| early_post | post0_5    | ecg      |           9 |  0.122017  |         0.0998271 |
| early_post | post0_1    | resp     |           9 |  0.121039  |         0.0804395 |
| early_post | post0_2    | resp     |           9 |  0.118497  |         0.0649101 |
| early_post | post1_3    | ecg      |           9 |  0.116319  |         0.0906583 |
| early_post | post0_3    | hr       |           9 |  0.112326  |         0.0959495 |
| early_post | post0_1    | hr       |           9 |  0.110022  |         0.0770266 |
| early_post | post0_3    | ecg      |           9 |  0.106423  |         0.0820073 |
| early_post | post0_2    | ecg      |           9 |  0.105694  |         0.0817947 |
| early_post | post0_2    | hr       |           9 |  0.103231  |         0.0813791 |
| early_post | post0_3    | resp     |           9 |  0.10207   |         0.0848688 |
| early_post | post0_5    | resp     |           9 |  0.0969262 |         0.0690207 |
| early_post | post0_5    | hr       |           9 |  0.0914261 |         0.0793884 |
| early_post | post0_3    | eda      |          18 |  0.0897653 |         0.0772448 |
| early_post | post1_3    | eda      |          18 |  0.0870824 |         0.0729322 |
| early_post | post1_3    | resp     |           9 |  0.080254  |         0.0635962 |
| early_post | post0_5    | eda      |          18 |  0.0773639 |         0.0718044 |
| early_post | post0_2    | eda      |          18 |  0.0765512 |         0.0690086 |
| early_post | post0_1    | eda      |          18 |  0.0725447 |         0.0586427 |
| late_post  | post2_5    | emg      |           9 |  0.195497  |         0.144091  |
| late_post  | post5_10   | ecg      |           9 |  0.163734  |         0.109318  |
| late_post  | post2_5    | ecg      |           9 |  0.135569  |         0.102756  |
| late_post  | post5_10   | resp     |           9 |  0.124131  |         0.0774676 |
| late_post  | post2_5    | resp     |           9 |  0.106142  |         0.0645723 |
| late_post  | post2_5    | hr       |           9 |  0.104309  |         0.07418   |
| late_post  | post5_10   | emg      |           9 |  0.0960794 |         0.0737176 |
| late_post  | post5_10   | eda      |          18 |  0.0770417 |         0.0695881 |
| late_post  | post2_5    | eda      |          18 |  0.0744941 |         0.0672121 |
| late_post  | post5_10   | hr       |           9 |  0.0577608 |         0.0414455 |
| pre        | pre2_0     | ecg      |           9 |  0.133064  |         0.0761795 |
| pre        | pre5_pre2  | emg      |           9 |  0.124416  |         0.0665673 |
| pre        | pre2_0     | emg      |           9 |  0.123355  |         0.0976065 |
| pre        | pre5_pre2  | ecg      |           9 |  0.113068  |         0.0843589 |
| pre        | pre2_0     | resp     |           9 |  0.107453  |         0.0740453 |
| pre        | pre10_pre5 | resp     |           9 |  0.101983  |         0.0612866 |
| pre        | pre5_pre2  | resp     |           9 |  0.0970755 |         0.0747451 |
| pre        | pre10_pre5 | emg      |           9 |  0.0963455 |         0.0730712 |
| pre        | pre10_pre5 | hr       |           9 |  0.0944031 |         0.0603768 |
| pre        | pre5_pre2  | hr       |           9 |  0.0943715 |         0.0567037 |
| pre        | pre2_0     | hr       |           9 |  0.092049  |         0.067903  |
| pre        | pre10_pre5 | eda      |          18 |  0.0809724 |         0.0759851 |
| pre        | pre10_pre5 | ecg      |           9 |  0.077596  |         0.055154  |
| pre        | pre5_pre2  | eda      |          18 |  0.075622  |         0.0687723 |
| pre        | pre2_0     | eda      |          18 |  0.0649745 |         0.057915  |

## top screened features

| feature                             | phase      | window    | signal   | metric             |   finite_rate_train |   max_abs_corr_train |
|:------------------------------------|:-----------|:----------|:---------|:-------------------|--------------------:|---------------------:|
| v293_post2_5_emg_z_p95              | late_post  | post2_5   | emg      | z_p95              |            0.888724 |             0.195497 |
| v293_post2_5_emg_z_abs_mean         | late_post  | post2_5   | emg      | z_abs_mean         |            0.888724 |             0.193449 |
| v293_post2_5_emg_line_length_per_s  | late_post  | post2_5   | emg      | line_length_per_s  |            0.888724 |             0.191188 |
| v293_post2_5_emg_z_p05              | late_post  | post2_5   | emg      | z_p05              |            0.888724 |             0.185646 |
| v293_post2_5_emg_z_std              | late_post  | post2_5   | emg      | z_std              |            0.888724 |             0.181624 |
| v293_post2_5_emg_z_range            | late_post  | post2_5   | emg      | z_range            |            0.888724 |             0.175136 |
| v293_post1_3_emg_z_last_minus_first | early_post | post1_3   | emg      | z_last_minus_first |            0.888724 |             0.170411 |
| v293_post1_3_emg_z_slope            | early_post | post1_3   | emg      | z_slope            |            0.888724 |             0.170411 |
| v293_post0_5_emg_z_p95              | early_post | post0_5   | emg      | z_p95              |            0.888724 |             0.167144 |
| v293_post0_5_emg_z_abs_mean         | early_post | post0_5   | emg      | z_abs_mean         |            0.888724 |             0.166721 |
| v293_post0_5_emg_line_length_per_s  | early_post | post0_5   | emg      | line_length_per_s  |            0.888724 |             0.166056 |
| v293_post5_10_ecg_z_abs_mean        | late_post  | post5_10  | ecg      | z_abs_mean         |            0.888724 |             0.163734 |
| v293_post5_10_ecg_z_mean            | late_post  | post5_10  | ecg      | z_mean             |            0.888724 |             0.160752 |
| v293_post0_5_emg_z_p05              | early_post | post0_5   | emg      | z_p05              |            0.888724 |             0.159057 |
| v293_post5_10_ecg_z_p95             | late_post  | post5_10  | ecg      | z_p95              |            0.888724 |             0.155754 |
| v293_post0_5_emg_z_std              | early_post | post0_5   | emg      | z_std              |            0.888724 |             0.147542 |
| v293_post0_3_emg_z_last_minus_first | early_post | post0_3   | emg      | z_last_minus_first |            0.888724 |             0.145452 |
| v293_post0_3_emg_z_slope            | early_post | post0_3   | emg      | z_slope            |            0.888724 |             0.145452 |
| v293_post1_3_emg_z_p95              | early_post | post1_3   | emg      | z_p95              |            0.888724 |             0.142375 |
| v293_post0_3_emg_z_p95              | early_post | post0_3   | emg      | z_p95              |            0.888724 |             0.13781  |
| v293_post0_3_emg_line_length_per_s  | early_post | post0_3   | emg      | line_length_per_s  |            0.888724 |             0.136239 |
| v293_post2_5_ecg_z_p95              | late_post  | post2_5   | ecg      | z_p95              |            0.888724 |             0.135569 |
| v293_post0_3_emg_z_p05              | early_post | post0_3   | emg      | z_p05              |            0.888724 |             0.135424 |
| v293_post1_3_emg_z_p05              | early_post | post1_3   | emg      | z_p05              |            0.888724 |             0.133283 |
| v293_pre2_0_ecg_z_last_minus_first  | pre        | pre2_0    | ecg      | z_last_minus_first |            0.888724 |             0.133064 |
| v293_pre2_0_ecg_z_slope             | pre        | pre2_0    | ecg      | z_slope            |            0.888724 |             0.13306  |
| v293_post1_3_emg_z_abs_mean         | early_post | post1_3   | emg      | z_abs_mean         |            0.888724 |             0.132718 |
| v293_post0_3_emg_z_abs_mean         | early_post | post0_3   | emg      | z_abs_mean         |            0.888724 |             0.132189 |
| v293_post1_3_emg_line_length_per_s  | early_post | post1_3   | emg      | line_length_per_s  |            0.888724 |             0.132181 |
| v293_post0_3_emg_z_std              | early_post | post0_3   | emg      | z_std              |            0.888724 |             0.130871 |
| v293_post1_3_emg_z_std              | early_post | post1_3   | emg      | z_std              |            0.888724 |             0.12985  |
| v293_post0_1_emg_z_range            | early_post | post0_1   | emg      | z_range            |            0.888724 |             0.129108 |
| v293_post0_1_ecg_z_p95              | early_post | post0_1   | ecg      | z_p95              |            0.888724 |             0.128963 |
| v293_post0_2_emg_line_length_per_s  | early_post | post0_2   | emg      | line_length_per_s  |            0.888724 |             0.12707  |
| v293_post0_2_emg_z_p95              | early_post | post0_2   | emg      | z_p95              |            0.888724 |             0.126108 |
| v293_pre5_pre2_emg_z_mean           | pre        | pre5_pre2 | emg      | z_mean             |            0.888724 |             0.124416 |
| v293_post5_10_resp_z_abs_mean       | late_post  | post5_10  | resp     | z_abs_mean         |            0.888724 |             0.124131 |
| v293_post0_1_emg_z_p95              | early_post | post0_1   | emg      | z_p95              |            0.888724 |             0.12381  |
| v293_pre2_0_emg_z_range             | pre        | pre2_0    | emg      | z_range            |            0.888724 |             0.123355 |
| v293_post0_5_emg_z_range            | early_post | post0_5   | emg      | z_range            |            0.888724 |             0.123132 |
| v293_post2_5_ecg_z_slope            | late_post  | post2_5   | ecg      | z_slope            |            0.888724 |             0.122703 |
| v293_post2_5_ecg_z_last_minus_first | late_post  | post2_5   | ecg      | z_last_minus_first |            0.888724 |             0.122703 |
| v293_post5_10_ecg_z_std             | late_post  | post5_10  | ecg      | z_std              |            0.888724 |             0.122442 |
| v293_post1_3_hr_z_range             | early_post | post1_3   | hr       | z_range            |            0.888724 |             0.122178 |
| v293_post5_10_resp_z_p95            | late_post  | post5_10  | resp     | z_p95              |            0.888724 |             0.122128 |
| v293_post0_2_emg_z_abs_mean         | early_post | post0_2   | emg      | z_abs_mean         |            0.888724 |             0.12203  |
| v293_post0_5_ecg_z_slope            | early_post | post0_5   | ecg      | z_slope            |            0.888724 |             0.122017 |
| v293_post0_5_ecg_z_last_minus_first | early_post | post0_5   | ecg      | z_last_minus_first |            0.888724 |             0.122017 |
| v293_post1_3_emg_z_range            | early_post | post1_3   | emg      | z_range            |            0.888724 |             0.12154  |
| v293_post0_1_emg_line_length_per_s  | early_post | post0_1   | emg      | line_length_per_s  |            0.888724 |             0.121434 |

## audit

```json
{
  "event_n": 1167,
  "feature_event_n": 1167,
  "missing_recording_groups": 7,
  "ok_rate": 0.919451585261354,
  "uses_post_observation": true,
  "post_features_are_diagnostic_only": true
}
```

## guardrail

```json
{
  "pass": true,
  "event_n": 1167,
  "feature_n": 540,
  "screen_feature_n": 540,
  "feature_set_n": 14,
  "ok_rate": 0.919451585261354,
  "uses_post_observation": true,
  "post_features_are_diagnostic_only": true,
  "guardrail_core_targets": [
    "bad_top10",
    "bad_top10_vehicle_ambiguous",
    "candidate_pool_gain_gt_005"
  ],
  "pre_route_supported_now": false,
  "pre_weak_subgroup_signal_exists": true,
  "post_wait_route_supported_diagnostic": true,
  "best_pre_badtop10_test_auc": 0.4896331738437001,
  "best_early_post_badtop10_test_auc": 0.7725677830940989,
  "best_pre_badtop10_vehicle_ambiguous_test_auc": 0.6011834319526628,
  "best_early_post_badtop10_vehicle_ambiguous_test_auc": 0.6627218934911242,
  "best_pre_candidate_gain_test_auc": 0.5722057953873447,
  "best_early_post_candidate_gain_test_auc": 0.5593140153755174,
  "test_used_for_feature_screen_or_threshold": false,
  "v292_route_viable_now": false
}
```

## 判断

- observation 后短窗出现比 pre 明显更强的生理可见性，后续应考虑 wait/late-observation 策略，而不是当前锚点前预测。