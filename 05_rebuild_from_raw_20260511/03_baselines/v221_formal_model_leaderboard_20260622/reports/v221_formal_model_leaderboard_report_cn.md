# v221 统一评估框架报告

## 范围

- 本轮只读取 v216/v217/v218/v219 已有 CSV 输出，不训练新模型。
- formal leaderboard 只包含允许候选；v218 强峰值训练、零变化、旧 v110 等仅作为诊断或背景。
- test 只用于最终报告，不用于选择 v222a 的阈值、模型族或残差强度。

## 候选决策摘要

| pool_key         | pool_name   | base_best_overall_test   |   base_best_overall_test_rmse | base_best_tail_test   |   base_best_tail_test_rmse_1to2s | base_lowest_under_test   |   base_lowest_under_test_rate | base_best_strong_rmse_test   |   base_best_strong_mean_sample_rmse | base_best_strong_under_test   |   base_best_strong_under_rate | base_best_normal_curve_test   |   base_best_normal_curve_mean_sample_rmse | least_harm_vs_reference   |   least_harm_large_harm_ratio |
|:-----------------|:------------|:-------------------------|------------------------------:|:----------------------|---------------------------------:|:-------------------------|------------------------------:|:-----------------------------|------------------------------------:|:------------------------------|------------------------------:|:------------------------------|------------------------------------------:|:--------------------------|------------------------------:|
| loose_main_pool  | 可用主池    | avg_joint_focus          |                      0.544879 | joint_equal           |                         0.629631 | peak_floor_090           |                      0.103261 | ridge_residual_peakfloor     |                            0.604504 | peak_floor_090                |                        0.1625 | global_blend                  |                                  0.348379 | global_blend              |                      0.152174 |
| strict_main_pool | 严格主池    | peak_floor_090           |                      0.571775 | peak_floor_090        |                         0.658308 | ridge_residual_peakfloor |                      0.091954 | ridge_residual_peakfloor     |                            0.606821 | ridge_residual_peakfloor      |                        0.15   | avg_joint_focus               |                                  0.3534   | global_blend              |                      0.183908 |

## GPTPro 要求的五个问题

### 1. 哪个模型总体 RMSE 最低？

| pool_key         | pool_name   | model_name               |   steer_rmse |   steer_tail_rmse_1to2s |   steer_severe_under_rate |
|:-----------------|:------------|:-------------------------|-------------:|------------------------:|--------------------------:|
| loose_main_pool  | 可用主池    | avg_joint_focus          |     0.544879 |                0.629739 |                  0.163043 |
| loose_main_pool  | 可用主池    | v219_val_selected        |     0.54978  |                0.635789 |                  0.173913 |
| loose_main_pool  | 可用主池    | global_blend             |     0.54978  |                0.635789 |                  0.173913 |
| loose_main_pool  | 可用主池    | joint_equal              |     0.553929 |                0.629631 |                  0.141304 |
| loose_main_pool  | 可用主池    | ridge_residual_joint     |     0.561455 |                0.659765 |                  0.125    |
| loose_main_pool  | 可用主池    | peak_floor_090           |     0.574285 |                0.65997  |                  0.103261 |
| loose_main_pool  | 可用主池    | ridge_residual_peakfloor |     0.576627 |                0.677724 |                  0.108696 |
| loose_main_pool  | 可用主池    | joint_steer_focus        |     0.586358 |                0.688588 |                  0.179348 |
| loose_main_pool  | 可用主池    | steering_only            |     0.620367 |                0.722493 |                  0.173913 |
| strict_main_pool | 严格主池    | peak_floor_090           |     0.571775 |                0.658308 |                  0.137931 |
| strict_main_pool | 严格主池    | global_blend             |     0.575564 |                0.672472 |                  0.258621 |
| strict_main_pool | 严格主池    | avg_joint_focus          |     0.580616 |                0.678813 |                  0.195402 |
| strict_main_pool | 严格主池    | v219_val_selected        |     0.58148  |                0.686853 |                  0.132184 |
| strict_main_pool | 严格主池    | ridge_residual_joint     |     0.58148  |                0.686853 |                  0.132184 |
| strict_main_pool | 严格主池    | ridge_residual_peakfloor |     0.582243 |                0.685261 |                  0.091954 |
| strict_main_pool | 严格主池    | joint_equal              |     0.596805 |                0.698089 |                  0.247126 |
| strict_main_pool | 严格主池    | steering_only            |     0.604244 |                0.687123 |                  0.264368 |
| strict_main_pool | 严格主池    | joint_steer_focus        |     0.61904  |                0.726776 |                  0.195402 |

### 2. 哪个模型强反应低估率最低？

| pool_key         | pool_name   | model_name               |   n_rows |   mean_sample_rmse |   severe_under_rate |   mean_peak_ratio |
|:-----------------|:------------|:-------------------------|---------:|-------------------:|--------------------:|------------------:|
| loose_main_pool  | 可用主池    | peak_floor_090           |       80 |           0.608899 |              0.1625 |          0.693137 |
| loose_main_pool  | 可用主池    | ridge_residual_peakfloor |       80 |           0.604504 |              0.1875 |          0.713576 |
| loose_main_pool  | 可用主池    | joint_equal              |       80 |           0.628628 |              0.1875 |          0.711811 |
| loose_main_pool  | 可用主池    | avg_joint_focus          |       80 |           0.611672 |              0.2    |          0.674713 |
| loose_main_pool  | 可用主池    | ridge_residual_joint     |       80 |           0.605222 |              0.2125 |          0.717939 |
| loose_main_pool  | 可用主池    | global_blend             |       80 |           0.624938 |              0.25   |          0.655158 |
| loose_main_pool  | 可用主池    | v219_val_selected        |       80 |           0.624938 |              0.25   |          0.655158 |
| loose_main_pool  | 可用主池    | joint_steer_focus        |       80 |           0.647123 |              0.25   |          0.668621 |
| loose_main_pool  | 可用主池    | steering_only            |       80 |           0.674963 |              0.3    |          0.632852 |
| strict_main_pool | 严格主池    | ridge_residual_peakfloor |       80 |           0.606821 |              0.15   |          0.742023 |
| strict_main_pool | 严格主池    | ridge_residual_joint     |       80 |           0.616814 |              0.2125 |          0.725069 |
| strict_main_pool | 严格主池    | v219_val_selected        |       80 |           0.616814 |              0.2125 |          0.725069 |
| strict_main_pool | 严格主池    | peak_floor_090           |       80 |           0.61948  |              0.225  |          0.676368 |
| strict_main_pool | 严格主池    | joint_steer_focus        |       80 |           0.663153 |              0.275  |          0.6551   |
| strict_main_pool | 严格主池    | steering_only            |       80 |           0.672878 |              0.275  |          0.625735 |
| strict_main_pool | 严格主池    | avg_joint_focus          |       80 |           0.649465 |              0.2875 |          0.632111 |
| strict_main_pool | 严格主池    | global_blend             |       80 |           0.642542 |              0.3375 |          0.625227 |
| strict_main_pool | 严格主池    | joint_equal              |       80 |           0.675651 |              0.35   |          0.627531 |

### 3. 哪个模型对普通弯道/普通样本伤害最小？

| pool_key         | pool_name   | model_name               |   n_rows |   mean_sample_rmse |   p90_sample_rmse |
|:-----------------|:------------|:-------------------------|---------:|-------------------:|------------------:|
| loose_main_pool  | 可用主池    | global_blend             |      104 |           0.348379 |          0.549026 |
| loose_main_pool  | 可用主池    | v219_val_selected        |      104 |           0.348379 |          0.549026 |
| loose_main_pool  | 可用主池    | avg_joint_focus          |      104 |           0.357577 |          0.553316 |
| loose_main_pool  | 可用主池    | ridge_residual_joint     |      104 |           0.367965 |          0.592299 |
| loose_main_pool  | 可用主池    | joint_equal              |      104 |           0.379674 |          0.572316 |
| loose_main_pool  | 可用主池    | ridge_residual_peakfloor |      104 |           0.38211  |          0.62486  |
| loose_main_pool  | 可用主池    | joint_steer_focus        |      104 |           0.390955 |          0.650141 |
| loose_main_pool  | 可用主池    | peak_floor_090           |      104 |           0.397668 |          0.622545 |
| loose_main_pool  | 可用主池    | steering_only            |      104 |           0.404115 |          0.695835 |
| strict_main_pool | 严格主池    | avg_joint_focus          |       94 |           0.3534   |          0.607864 |
| strict_main_pool | 严格主池    | global_blend             |       94 |           0.358549 |          0.588703 |
| strict_main_pool | 严格主池    | joint_equal              |       94 |           0.362308 |          0.629778 |
| strict_main_pool | 严格主池    | peak_floor_090           |       94 |           0.371749 |          0.612275 |
| strict_main_pool | 严格主池    | v219_val_selected        |       94 |           0.37676  |          0.634149 |
| strict_main_pool | 严格主池    | ridge_residual_joint     |       94 |           0.37676  |          0.634149 |
| strict_main_pool | 严格主池    | ridge_residual_peakfloor |       94 |           0.388904 |          0.646746 |
| strict_main_pool | 严格主池    | steering_only            |       94 |           0.398473 |          0.596277 |
| strict_main_pool | 严格主池    | joint_steer_focus        |       94 |           0.414498 |          0.666176 |

### 4. 哪个模型在极强峰值 >=3 rad 上表现最差？

| pool_key         | pool_name   | model_name               |   n_rows |   mean_sample_rmse |   severe_under_rate |   mean_peak_ratio |
|:-----------------|:------------|:-------------------------|---------:|-------------------:|--------------------:|------------------:|
| loose_main_pool  | 可用主池    | joint_equal              |        6 |           1.11393  |            0.166667 |          0.622631 |
| loose_main_pool  | 可用主池    | steering_only            |        6 |           1.03699  |            0.166667 |          0.586127 |
| loose_main_pool  | 可用主池    | global_blend             |        6 |           1.03264  |            0.166667 |          0.600836 |
| loose_main_pool  | 可用主池    | v219_val_selected        |        6 |           1.03264  |            0.166667 |          0.600836 |
| loose_main_pool  | 可用主池    | avg_joint_focus          |        6 |           1.03001  |            0.166667 |          0.619919 |
| loose_main_pool  | 可用主池    | peak_floor_090           |        6 |           1.00186  |            0.166667 |          0.625012 |
| loose_main_pool  | 可用主池    | joint_steer_focus        |        6 |           0.996955 |            0.166667 |          0.636828 |
| loose_main_pool  | 可用主池    | ridge_residual_joint     |        6 |           0.956812 |            0.166667 |          0.669852 |
| loose_main_pool  | 可用主池    | ridge_residual_peakfloor |        6 |           0.922369 |            0        |          0.662889 |
| strict_main_pool | 严格主池    | joint_steer_focus        |        6 |           1.16175  |            0.666667 |          0.527518 |
| strict_main_pool | 严格主池    | global_blend             |        6 |           1.10752  |            0.666667 |          0.539651 |
| strict_main_pool | 严格主池    | avg_joint_focus          |        6 |           1.13459  |            0.5      |          0.536063 |
| strict_main_pool | 严格主池    | joint_equal              |        6 |           1.16547  |            0.333333 |          0.548748 |
| strict_main_pool | 严格主池    | peak_floor_090           |        6 |           1.04284  |            0.333333 |          0.578816 |
| strict_main_pool | 严格主池    | ridge_residual_joint     |        6 |           1.04267  |            0.333333 |          0.604165 |
| strict_main_pool | 严格主池    | v219_val_selected        |        6 |           1.04267  |            0.333333 |          0.604165 |
| strict_main_pool | 严格主池    | ridge_residual_peakfloor |        6 |           1.0116   |            0.333333 |          0.60425  |
| strict_main_pool | 严格主池    | steering_only            |        6 |           1.11477  |            0.166667 |          0.595823 |

### 5. 哪些样本每个模型都预测不好？

| pool_key         |   array_index | event_uid                                         | subject   | scene_type   |   best_rmse |   mean_rmse |   true_steer_peak_abs |
|:-----------------|--------------:|:--------------------------------------------------|:----------|:-------------|------------:|------------:|----------------------:|
| strict_main_pool |           487 | rjy_Entity_Recording_2025_09_28_20_02_20_v108_041 | rjy       | 直道事件     |    1.56396  |    1.90356  |              3.31438  |
| loose_main_pool  |           587 | rjy_Entity_Recording_2025_09_28_20_02_20_v108_041 | rjy       | 直道事件     |    1.33029  |    1.46806  |              3.31438  |
| loose_main_pool  |           701 | tyy_Entity_Recording_2025_09_28_14_23_43_v108_002 | tyy       | 直道事件     |    1.2574   |    1.31615  |              2.28726  |
| strict_main_pool |           591 | tyy_Entity_Recording_2025_09_28_14_23_43_v108_014 | tyy       | 直道事件     |    1.25006  |    1.2592   |              2.66808  |
| loose_main_pool  |           703 | tyy_Entity_Recording_2025_09_28_14_23_43_v108_014 | tyy       | 直道事件     |    1.21359  |    1.27524  |              2.66808  |
| strict_main_pool |           455 | rjy_Entity_Recording_2025_09_28_19_51_44_v108_023 | rjy       | 下坡弯道事件 |    1.19035  |    1.25465  |              1.89735  |
| strict_main_pool |           589 | tyy_Entity_Recording_2025_09_28_14_23_43_v108_002 | tyy       | 直道事件     |    1.19032  |    1.25376  |              2.28726  |
| loose_main_pool  |           554 | rjy_Entity_Recording_2025_09_28_19_51_44_v108_023 | rjy       | 下坡弯道事件 |    1.16347  |    1.27902  |              1.89735  |
| loose_main_pool  |           711 | tyy_Entity_Recording_2025_09_28_14_23_43_v108_026 | tyy       | 下坡弯道事件 |    1.15264  |    1.26921  |              2.41169  |
| loose_main_pool  |           556 | rjy_Entity_Recording_2025_09_28_19_51_44_v108_025 | rjy       | 直道事件     |    1.13689  |    1.16017  |              2.31099  |
| strict_main_pool |           590 | tyy_Entity_Recording_2025_09_28_14_23_43_v108_004 | tyy       | 直道事件     |    1.10582  |    1.16762  |              2.55604  |
| strict_main_pool |           486 | rjy_Entity_Recording_2025_09_28_20_02_20_v108_040 | rjy       | 直道事件     |    1.09069  |    1.36751  |              3.37783  |
| strict_main_pool |           457 | rjy_Entity_Recording_2025_09_28_19_51_44_v108_025 | rjy       | 直道事件     |    1.08999  |    1.16106  |              2.31099  |
| strict_main_pool |           599 | tyy_Entity_Recording_2025_09_28_14_23_43_v108_026 | tyy       | 下坡弯道事件 |    1.0847   |    1.21595  |              2.41169  |
| loose_main_pool  |           563 | rjy_Entity_Recording_2025_09_28_19_51_44_v108_039 | rjy       | 直道事件     |    1.0575   |    1.15893  |              1.2706   |
| loose_main_pool  |           586 | rjy_Entity_Recording_2025_09_28_20_02_20_v108_040 | rjy       | 直道事件     |    1.0285   |    1.13033  |              3.37783  |
| strict_main_pool |           463 | rjy_Entity_Recording_2025_09_28_19_51_44_v108_039 | rjy       | 直道事件     |    0.998704 |    1.15424  |              1.2706   |
| loose_main_pool  |           580 | rjy_Entity_Recording_2025_09_28_20_02_20_v108_031 | rjy       | 下坡弯道事件 |    0.977866 |    1.14304  |              3.21193  |
| strict_main_pool |           480 | rjy_Entity_Recording_2025_09_28_20_02_20_v108_031 | rjy       | 下坡弯道事件 |    0.973243 |    1.06592  |              3.21193  |
| strict_main_pool |           451 | rjy_Entity_Recording_2025_09_28_19_51_44_v108_014 | rjy       | 直道事件     |    0.938015 |    0.97397  |              2.57977  |
| strict_main_pool |           468 | rjy_Entity_Recording_2025_09_28_20_02_20_v108_010 | rjy       | 下坡弯道事件 |    0.93294  |    1.33776  |              0.820132 |
| loose_main_pool  |           549 | rjy_Entity_Recording_2025_09_28_19_51_44_v108_014 | rjy       | 直道事件     |    0.925676 |    1.11833  |              2.57977  |
| loose_main_pool  |           713 | tyy_Entity_Recording_2025_09_28_14_23_43_v108_029 | tyy       | 直道事件     |    0.915898 |    1.00972  |              3.04263  |
| strict_main_pool |           482 | rjy_Entity_Recording_2025_09_28_20_02_20_v108_034 | rjy       | 下坡弯道事件 |    0.906685 |    0.957037 |              1.76942  |
| loose_main_pool  |           702 | tyy_Entity_Recording_2025_09_28_14_23_43_v108_004 | tyy       | 直道事件     |    0.885812 |    1.13528  |              2.55604  |
| loose_main_pool  |           537 | rjy_Entity_Recording_2025_09_28_19_33_26_v108_021 | rjy       | 直道事件     |    0.88358  |    0.908583 |              1.71828  |
| strict_main_pool |           601 | tyy_Entity_Recording_2025_09_28_14_23_43_v108_029 | tyy       | 直道事件     |    0.875538 |    0.988406 |              3.04263  |
| strict_main_pool |           439 | rjy_Entity_Recording_2025_09_28_19_33_26_v108_021 | rjy       | 直道事件     |    0.842786 |    0.953207 |              1.71828  |
| loose_main_pool  |           530 | rjy_Entity_Recording_2025_09_28_19_33_26_v108_013 | rjy       | 下坡弯道事件 |    0.838043 |    0.856408 |              1.81619  |
| strict_main_pool |           607 | tyy_Entity_Recording_2025_09_28_14_23_43_v108_037 | tyy       | 直道事件     |    0.830036 |    0.885414 |              2.23699  |
| loose_main_pool  |           719 | tyy_Entity_Recording_2025_09_28_14_23_43_v108_037 | tyy       | 直道事件     |    0.81403  |    0.868    |              2.23699  |
| loose_main_pool  |           728 | tyy_Entity_Recording_2025_09_28_14_40_01_v108_008 | tyy       | 下坡弯道事件 |    0.796411 |    0.829861 |              1.89193  |
| strict_main_pool |           369 | lx_Entity_Recording_2025_09_26_08_58_43_v108_011  | lx        | 直道事件     |    0.784235 |    0.857976 |              3.19605  |
| loose_main_pool  |           454 | lx_Entity_Recording_2025_09_26_09_17_22_v108_019  | lx        | 下坡弯道事件 |    0.76749  |    0.837697 |              2.15199  |
| loose_main_pool  |           582 | rjy_Entity_Recording_2025_09_28_20_02_20_v108_034 | rjy       | 下坡弯道事件 |    0.765815 |    0.885818 |              1.76942  |
| loose_main_pool  |           450 | lx_Entity_Recording_2025_09_26_08_58_43_v108_011  | lx        | 直道事件     |    0.75856  |    0.928924 |              3.19605  |
| strict_main_pool |           101 | cwh_Entity_Recording_2025_09_26_20_06_19_v108_002 | cwh       | 直道事件     |    0.742477 |    0.947761 |              1.98915  |
| loose_main_pool  |           570 | rjy_Entity_Recording_2025_09_28_20_02_20_v108_014 | rjy       | 直道事件     |    0.741843 |    0.789994 |              2.61538  |
| strict_main_pool |           608 | tyy_Entity_Recording_2025_09_28_14_23_43_v108_038 | tyy       | 直道事件     |    0.73977  |    0.869665 |              2.44442  |
| loose_main_pool  |           720 | tyy_Entity_Recording_2025_09_28_14_23_43_v108_038 | tyy       | 直道事件     |    0.730109 |    0.903994 |              2.44442  |

## Do-no-harm 参考

| pool_key         | baseline_model   | model_name               |   n_rows |   mean_delta_vs_baseline |   improved_ratio |   harmed_ratio |   large_harm_ratio_delta_gt_0p05 |   worst_delta |   best_delta |
|:-----------------|:-----------------|:-------------------------|---------:|-------------------------:|-----------------:|---------------:|---------------------------------:|--------------:|-------------:|
| loose_main_pool  | avg_joint_focus  | avg_joint_focus          |      184 |              0           |         0        |       0        |                         0        |      0        |     0        |
| loose_main_pool  | avg_joint_focus  | global_blend             |      184 |              0.000568963 |         0.505435 |       0.494565 |                         0.152174 |      0.463692 |    -0.484351 |
| loose_main_pool  | avg_joint_focus  | v219_val_selected        |      184 |              0.000568963 |         0.505435 |       0.494565 |                         0.152174 |      0.463692 |    -0.484351 |
| loose_main_pool  | avg_joint_focus  | ridge_residual_joint     |      184 |              0.00306689  |         0.478261 |       0.521739 |                         0.195652 |      1.02567  |    -0.438414 |
| loose_main_pool  | avg_joint_focus  | ridge_residual_peakfloor |      184 |              0.0107501   |         0.467391 |       0.532609 |                         0.244565 |      1.44913  |    -0.465572 |
| loose_main_pool  | avg_joint_focus  | peak_floor_090           |      184 |              0.0214543   |         0.472826 |       0.527174 |                         0.26087  |      1.29408  |    -0.191866 |
| loose_main_pool  | avg_joint_focus  | joint_steer_focus        |      184 |              0.0342798   |         0.391304 |       0.608696 |                         0.336957 |      0.509722 |    -0.184575 |
| loose_main_pool  | avg_joint_focus  | joint_equal              |      184 |              0.019862    |         0.375    |       0.625    |                         0.336957 |      0.255423 |    -0.456206 |
| loose_main_pool  | avg_joint_focus  | steering_only            |      184 |              0.0538222   |         0.380435 |       0.619565 |                         0.456522 |      1.56867  |    -0.329376 |
| strict_main_pool | peak_floor_090   | peak_floor_090           |      174 |              0           |         0        |       0        |                         0        |      0        |     0        |
| strict_main_pool | peak_floor_090   | global_blend             |      174 |              0.00347218  |         0.356322 |       0.477011 |                         0.183908 |      0.23287  |    -0.352054 |
| strict_main_pool | peak_floor_090   | avg_joint_focus          |      174 |              0.0038734   |         0.45977  |       0.54023  |                         0.206897 |      0.232808 |    -0.622315 |
| strict_main_pool | peak_floor_090   | ridge_residual_joint     |      174 |              0.00148135  |         0.522989 |       0.477011 |                         0.218391 |      0.335273 |    -0.357421 |
| strict_main_pool | peak_floor_090   | v219_val_selected        |      174 |              0.00148135  |         0.522989 |       0.477011 |                         0.218391 |      0.335273 |    -0.357421 |
| strict_main_pool | peak_floor_090   | ridge_residual_peakfloor |      174 |              0.00344766  |         0.494253 |       0.505747 |                         0.247126 |      0.362618 |    -0.179607 |
| strict_main_pool | peak_floor_090   | joint_equal              |      174 |              0.0207258   |         0.465517 |       0.534483 |                         0.321839 |      0.581897 |    -0.520429 |
| strict_main_pool | peak_floor_090   | steering_only            |      174 |              0.0389881   |         0.396552 |       0.603448 |                         0.448276 |      0.468049 |    -0.93843  |
| strict_main_pool | peak_floor_090   | joint_steer_focus        |      174 |              0.0431742   |         0.321839 |       0.678161 |                         0.471264 |      0.458602 |    -0.694501 |

## 当前结论

- v222a 可以继续作为下一步，但必须基于 v221 的 `v221_candidate_decision_summary.csv` 固定候选和基准。
- 不建议直接进入 v222b/v223；当前应先做轻量软融合和受限残差。
- v218 代表“强峰值 loss 直接内化训练”的诊断，不应作为新主线。